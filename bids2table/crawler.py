import logging
import os
import traceback
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa

from bids2table import RecordDict, StrOrPath
from bids2table.handlers import HandlerLUT, HandlerTuple
from bids2table.indexers import Indexer
from bids2table.table import IncrementalTable


@dataclass
class HandlingFailure:
    path: str
    pattern: str
    handler: str
    exception: str

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class CrawlCounts:
    count: int
    error_count: int

    @property
    def error_rate(self) -> float:
        return self.error_count / self.count


class Crawler:
    def __init__(
        self,
        indexers_map: Dict[str, Indexer],
        handlers_map: Dict[str, List[HandlerTuple]],
        max_threads: Optional[int] = 8,
        max_failures: Optional[int] = None,
    ):
        if not list(handlers_map.keys()) == list(indexers_map.keys()):
            raise ValueError("handlers_map and indexers_map should have identical keys")

        self.handlers_map = handlers_map
        self.indexers_map = indexers_map
        if max_threads is None:
            # default from ThreadPoolExecutor
            max_threads = min(32, (os.cpu_count() or 1) + 4)
        self.max_threads = max_threads
        self.max_failures = max_failures

        self._handler_lut = HandlerLUT(
            [h for handlers in handlers_map.values() for h in handlers]
        )

    def __call__(
        self, dirpath: StrOrPath
    ) -> Tuple[Dict[str, pa.Table], List[HandlingFailure], CrawlCounts]:
        """
        Crawl the directory, scanning for files matching the handler patterns, and
        processing each with the assigned handler(s). Returns a PyArrow ``Table`` of
        extracted and transformed data, and a list of failures, if any.
        """
        for indexer in self.indexers_map.values():
            indexer.set_root(dirpath)
        taskfn = self._make_taskfn(self.indexers_map)

        constants = {"_dir": str(dirpath)}
        tables = self._new_tables(constants)

        count, err_count = 0, 0
        errors = []

        with ThreadPool(self.max_threads) as pool:
            # Lazy streaming evaluation using ``imap_unordered`` applied to a generator.
            # NOTE: Using ``ThreadPool`` instead of the more modern
            # ``ThreadPoolExecutor`` bc it doesn't support this lazy evaluation.
            for val in pool.imap_unordered(
                taskfn, self._scan_for_matches(dirpath, self._handler_lut)
            ):
                handler, data, err = val
                if data is not None:
                    key, record = data
                    tables[handler.group].put(key, record, handler.label)
                    count += 1
                if err is not None:
                    errors.append(err)
                    err_count += 1
                    if self.max_failures and len(errors) >= self.max_failures:
                        raise RuntimeError(
                            f"Max number of failures {self.max_failures} exceeded"
                        )

        tables = {group: table.as_table() for group, table in tables.items()}
        return tables, errors, CrawlCounts(count, err_count)

    def _new_tables(
        self,
        constants: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, IncrementalTable]:
        """
        Create new ``IncrementalTables`` based on the defined indexer and handlers, and
        optional additional constants.
        """
        tables = {}
        for group, indexer in self.indexers_map.items():
            handlers = self.handlers_map[group]
            tables[group] = IncrementalTable(
                index_schema=indexer.schema,
                schema={h.label: h.handler.schema for h in handlers},
                constants=constants,
            )
        return tables

    @staticmethod
    def _scan_for_matches(
        dirpath: StrOrPath,
        handler_lut: HandlerLUT,
    ) -> Iterator[Tuple[str, HandlerTuple]]:
        """
        Scan a directory for files matching the handler patterns.
        """
        # TODO: A sub-class might generalize this to scan S3.
        with os.scandir(str(dirpath)) as it:
            for entry in it:
                for handler in handler_lut.lookup(entry.path):
                    yield entry.path, handler

    @staticmethod
    def _make_taskfn(indexers_map: Dict[str, Indexer]):
        """
        Return the actual task function that is mapped by the thread pool over the scan
        results. Consumes ``args`` tuples from ``scan_for_matches`` and calls
        ``_process_one``.
        """

        def taskfn(match: Tuple[str, HandlerTuple]):
            path, handler = match
            indexer = indexers_map[handler.group]
            data, err = Crawler._process_one(indexer, path, handler)
            return handler, data, err

        return taskfn

    @staticmethod
    def _process_one(
        indexer: Indexer, path: str, handler: HandlerTuple
    ) -> Tuple[Optional[Tuple[RecordDict, RecordDict]], Optional[HandlingFailure]]:
        """
        Process a single file match with its corresponding handler.
        """
        try:
            key = indexer(path)
            err = None
        except Exception as exc:
            logging.warning(
                "Indexer failed to process a file\n"
                f"\tdirpath: {indexer.root}\n"
                f"\tpath: {path}\n"
                f"\tpattern: {handler.pattern}\n"
                f"\tindexer: {indexer}\n\n" + traceback.format_exc() + "\n"
            )
            key = None
            err = HandlingFailure(path, handler.pattern, repr(indexer), repr(exc))

        if key is None:
            return None, err

        try:
            record = handler.handler(path)
            err = None
        except Exception as exc:
            # TODO: in this case how to we re-run just the files that failed?
            # In general, robustly being able to update the table is a critical
            # feature I guess could re-try just the subjects with failures.
            logging.warning(
                "Handler failed to process a file\n"
                f"\tdirpath: {indexer.root}\n"
                f"\tpath: {path}\n"
                f"\tpattern: {handler.pattern}\n"
                f"\thandler: {handler.handler}\n\n" + traceback.format_exc() + "\n"
            )
            record = None
            err = HandlingFailure(
                path, handler.pattern, repr(handler.handler), repr(exc)
            )

        data = None if record is None else (key, record)
        return data, err
