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


class Crawler:
    def __init__(
        self,
        dirpath: Path,
        handler_map: HandlerMap,
        max_workers: Optional[int] = None,
    ):
        self.dirpath = dirpath
        self.handler_map = handler_map
        if max_workers is None:
            # default from ThreadPoolExecutor
            max_workers = min(32, os.cpu_count() + 4)
        self.max_workers = max_workers

        self.context: Optional[BIDSContext] = None

    def __call__(self) -> Tuple[pd.DataFrame, List[HandlingFailure]]:
        """
        Crawl the directory, scanning for files matching the list of patterns, and
        processing each with its associated handler(s). Returns a ``DataFrame`` of
        extracted and transformed data and a list of failures, if any.

        TODO:
            [ ] Discuss how records are grouped into rows by metadata key in the
                appropriate place.
            [ ] Exit early after a maximum number of errors?
        """
        self._init_context()

        def taskfn(args):
            record, err = self._process_one(*args)
            return args + (record, err)

        # A 2d dict similar to a matlab cell array where rows are indexed by a record
        # key generated from metadata (``Record.key()``), and columns are indexed by
        # handler name. Each cell contains a ``Record``. To convert to a pandas table,
        # the records within a row are flattened together.
        record_table = GroupedRecordTable(self.handler_map.handlers())
        errors = []

        with ThreadPool(self.max_workers) as pool:
            # Lazy streaming evaluation using ``imap_unordered`` applied to a generator.
            # NOTE: Using ``ThreadPool`` instead of the more modern
            # ``ThreadPoolExecutor`` bc it doesn't support this lazy evaluation.
            for val in pool.imap_unordered(taskfn, self._scan_for_matches()):
                *_, handler, record, err = val
                if record is not None:
                    record_table.add(record, handler)
                if err is not None:
                    errors.append(err)

        record_table = record_table.to_pandas()
        return record_table, errors

    def _init_context(self):
        """
        Initialize the directory context.
        """
        self.context = BIDSContext(self.dirpath)

    def _scan_for_matches(self) -> Iterable[Tuple[str, str, Handler]]:
        """
        Scan a directory for files matching the patterns in ``handler_map``. Yields tuples
        of ``(pattern, path, handler)```.
        """
        # TODO: A sub-class or might generalize this to scan S3.
        with os.scandir(self.dirpath) as it:
            for entry in it:
                for pattern, handler in self.handler_map.lookup(entry.path):
                    yield pattern, entry.path, handler

    def _process_one(
        self, pattern: str, path: str, handler: Handler
    ) -> Tuple[Optional[Record], Optional[HandlingFailure]]:
        """
        Process a single file match with its corresponding handler.
        """
        assert self.context is not None, "crawler context is uninitialized"
        # TODO: safe to assume this won't crash (except in catastrophes)?
        #   - let's say yes
        metadata = self.context.get_metadata(path)
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
                f"\tdirpath: {self.dirpath}\n"
                f"\tpath: {path}\n"
                f"\tpattern: {pattern}\n"
                f"\thandler: {handler.name}\n\n" + traceback.format_exc() + "\n"
            )
            data = None
            err = HandlingFailure(self.dirpath, path, pattern, handler.name)

        record = None if data is None else Record(metadata, data)
        return record, err
