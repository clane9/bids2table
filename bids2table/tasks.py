import logging
import os
import traceback
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from bids2table.context import BIDSContext
from bids2table.handlers import Handler, HandlerMap
from bids2table.record import GroupedRecordTable, Record


@dataclass
class HandlingFailure:
    dirpath: str
    path: str
    pattern: str
    handler: str


def scan_directory_task(
    dirpath: Path,
    handler_map: HandlerMap,
    max_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[HandlingFailure]]:
    """
    Scan a directory for files and process them using handlers from the handler lookup
    table. Group and flatten the resulting records.

    TODO:
        [x] How to handle the various corner cases arising from:
            - missing files for a pattern/unmatched handlers
            - multiple files for a pattern/overmatched handlers
            - multiple metadata groups, possibly with non-overlapping data

            One simplifying assumption would be to say: within a scan task, every
            metadata group must share the same overall schema. I.e. do multiple scans,
            one per table. This seems fine.
        [x] TODO: threading to overlap io and compute
            - is there any risk of deadlock if e.g. queues start to block before we
              start to read off results? I hope not here. We're assuming the data for a
              session fits in memory? A bit risky. Maybe we no longer need to make it if
              we have the assumption above.

              No. I think it's important to make this assumption. If one dir of data
              doesn't fit in memory, we're trying to put too much data in the table.
        [ ] Quit early after maximum number of errors?
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count())

    # TODO: BIDSContext will handle the inference of metadata like subject, session,
    # modality, etc. It can read global info from the directory, as well as local info
    # for each path.
    context = BIDSContext(dirpath)

    def taskfn(args):
        record, err = _handler_match_task(*args, dirpath=dirpath, context=context)
        return args + (record, err)

    matches = _scan_directory_for_matches(dirpath, handler_map)

    # TODO: Should this be some abstraction? It's a bit of an awkward data structure
    #   - it's like a 2d dict; analogue to 2d array, list-of-list
    #   - indexed by key, and column group
    #   - `GroupedRecordTable` ? sounds fine
    #       - needs to depend on the handler table, because the handlers define the
    #         column groups. (aside: is it fine to couple handlers to columns? I think
    #         so, can't have a column (group) without a handler)
    record_table = GroupedRecordTable(handler_map.handlers())
    errors = []

    with ThreadPool(max_workers) as pool:
        # NOTE: bc pool map does not operate lazily, this will need to complete the full
        # scan before it starts processing
        for val in pool.imap_unordered(taskfn, matches):
            *_, handler, record, err = val
            if record is not None:
                record_table.add(record, handler)
            if err is not None:
                errors.append(err)

    record_table = record_table.to_pandas()
    return record_table, errors


def _scan_directory_for_matches(
    dirpath: Path, handler_map: HandlerMap
) -> Iterable[Tuple[str, str, Handler]]:
    """
    Scan a directory for files matching the patterns in ``handler_map``. Yields tuples
    of ``(pattern, path, handler)```.
    """
    with os.scandir(dirpath) as it:
        for entry in it:
            for pattern, handler in handler_map.lookup(entry.path):
                yield pattern, entry.path, handler


def _handler_match_task(
    pattern: str,
    path: str,
    handler: Handler,
    *,
    dirpath: Path,
    context: BIDSContext,
) -> Tuple[Optional[Record], Optional[HandlingFailure]]:
    """
    Process a single file match with its corresponding handler.
    """
    # TODO: safe to assume this won't crash (except in catastrophes)?
    #   - yes
    metadata = context.get_metadata(path)
    if metadata is None:
        return None, None

    try:
        data = handler(path)
        err = None
    except Exception:
        # TODO: in this case how to we re-run just the files that failed?
        # In general, robustly being able to update the table is a critical
        # feature I guess could re-try just the subjects with failures.
        logging.warning(
            "Handler failed to process a file\n"
            f"\tdirpath: {dirpath}\n"
            f"\tpath: {path}\n"
            f"\tpattern: {pattern}\n"
            f"\thandler: {handler.name}\n\n" + traceback.format_exc() + "\n"
        )
        data = None
        err = HandlingFailure(dirpath, path, pattern, handler.name)

    record = None if data is None else Record(metadata, data)
    return record, err
