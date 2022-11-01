import logging
import os
import traceback
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import pyarrow as pa

from bids2table import Key, RecordDict, StrOrPath
from bids2table.context import Context
from bids2table.handlers import Handler, HandlerLUT
from bids2table.schema import Schema
from bids2table.table import Table


class HandlingFailure(NamedTuple):
    dirpath: str
    path: str
    pattern: str
    handler: str


class MatchResult(NamedTuple):
    path: str
    group: str
    handler: Handler


# TODO:
#   [x] dirpath should be an arg to __call__
#   [x] context factory should be arg to init
#   [x] pyarrow table should be the return format
#       - interchangeable with pandas
#       - contains schema


class Crawler:
    def __init__(
        self,
        handlers_map: Dict[str, List[Handler]],
        context: Context,
        max_threads: Optional[int] = 8,
        max_failures: Optional[int] = None,
    ):
        self.handlers_map = handlers_map
        self.context = context
        if max_threads is None:
            # default from ThreadPoolExecutor
            max_threads = min(32, (os.cpu_count() or 1) + 4)
        self.max_threads = max_threads
        self.max_failures = max_failures
        self._handler_lut = HandlerLUT(handlers_map)

    def __call__(
        self, dirpath: StrOrPath
    ) -> Tuple[Dict[str, pa.Table], List[HandlingFailure]]:
        """
        Crawl the directory, scanning for files matching the handler patterns, and
        processing each with the assigned handler(s). Returns a PyArrow ``Table`` of
        extracted and transformed data, and a list of failures, if any.

        TODO:
            [x] Exit early after a maximum number of errors?
        """
        self.context.set_root(dirpath)
        taskfn = self._make_taskfn(self.context)

        column_groups = self._column_groups_from_handlers(self.handlers_map)
        tables = {
            group: Table(cg, self.context.index_names())
            for group, cg in column_groups.items()
        }
        errors = []

        with ThreadPool(self.max_threads) as pool:
            # Lazy streaming evaluation using ``imap_unordered`` applied to a generator.
            # NOTE: Using ``ThreadPool`` instead of the more modern
            # ``ThreadPoolExecutor`` bc it doesn't support this lazy evaluation.
            for val in pool.imap_unordered(
                taskfn, self._scan_for_matches(dirpath, self._handler_lut)
            ):
                group, handler, data, err = val
                if data is not None:
                    key, record = data
                    tables[group].put(key, handler.name, record)
                if err is not None:
                    errors.append(err)
                    if self.max_failures and len(errors) >= self.max_failures:
                        raise RuntimeError(
                            f"Max number of failures {self.max_failures} exceeded"
                        )

        tables = {group: table.to_pyarrow() for group, table in tables.items()}
        return tables, errors

    @staticmethod
    def _scan_for_matches(
        dirpath: StrOrPath,
        handler_lut: HandlerLUT,
    ) -> Iterator[MatchResult]:
        """
        Scan a directory for files matching the handler patterns. Yields tuples of
        ``(path, group, handler)```.
        """
        # TODO: A sub-class might generalize this to scan S3.
        with os.scandir(str(dirpath)) as it:
            for entry in it:
                for group, handler in handler_lut.lookup(entry.path):
                    yield MatchResult(entry.path, group, handler)

    @staticmethod
    def _make_taskfn(context: Context):
        """
        Return the actual task function that is mapped by the thread pool over the scan
        results. Consumes ``args`` tuples from ``scan_for_matches`` and calls
        ``_process_one``.
        """

        def taskfn(match: MatchResult):
            path, group, handler = match
            data, err = Crawler._process_one(context, path, handler)
            return group, handler, data, err

        return taskfn

    @staticmethod
    def _process_one(
        context: Context, path: str, handler: Handler
    ) -> Tuple[Optional[Tuple[Key, RecordDict]], Optional[HandlingFailure]]:
        """
        Process a single file match with its corresponding handler.
        """
        key = context.get_key(path)
        if key is None:
            return None, None

        try:
            record = handler(path)
            err = None
        except Exception:
            # TODO: in this case how to we re-run just the files that failed?
            # In general, robustly being able to update the table is a critical
            # feature I guess could re-try just the subjects with failures.
            logging.warning(
                "Handler failed to process a file\n"
                f"\tdirpath: {context.root}\n"
                f"\tpath: {path}\n"
                f"\tpattern: {handler.pattern}\n"
                f"\thandler: {handler.name}\n\n" + traceback.format_exc() + "\n"
            )
            record = None
            err = HandlingFailure(
                str(context.root), path, handler.pattern, handler.name
            )

        data = None if record is None else (key, record)
        return data, err

    @staticmethod
    def _column_groups_from_handlers(
        handlers_map: Dict[str, List[Handler]]
    ) -> Dict[str, Dict[str, Schema]]:
        """
        Extract column groups from a bunch of handlers, using the ``handler.name`` as
        the column group key.
        """
        column_groups: Dict[str, Dict[str, Schema]] = defaultdict(dict)
        for group, handlers in handlers_map.items():
            for handler in handlers:
                if handler.name in column_groups[group]:
                    raise RuntimeError(
                        f"Duplicate handler with name {handler.name} in group {group}"
                    )
                column_groups[group][handler.name] = handler.schema
        return column_groups
