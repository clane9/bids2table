import logging
import os
import traceback
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import pyarrow as pa

from bids2table.handlers import HandlerTuple
from bids2table.indexers import Indexer
from bids2table.table import IncrementalTable
from bids2table.types import RecordDict, StrOrPath
from bids2table.utils import PatternLUT


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
    total: int = 0
    process: int = 0
    error: int = 0

    @property
    def error_rate(self) -> float:
        return self.error / max(self.total, 1)

    def to_dict(self):
        return self.__dict__.copy()

    def update(self, other: "CrawlCounts"):
        """
        Increment counts with counts from ``other``.
        """
        self.total += other.total
        self.process += other.process
        self.error += other.error


class Crawler:
    """
    Crawl a directory, scan for files matching the handler patterns, and process each
    with the assigned handler(s). Insert each record at the index generated by the
    assigned indexer.

    Args:
        indexers_map: mapping of table names to indexers.
        handlers_map: mapping of table names to lists of handlers (more precisely
            ``HandlerTuple``s).
        max_threads: maximum number of threads for concurrent I/O and computation.
        max_failures: maximum number of handling failures to tolerate.
    """

    def __init__(
        self,
        indexers_map: Mapping[str, Indexer],
        handlers_map: Mapping[str, List[HandlerTuple]],
        max_threads: int = 4,
        max_failures: Optional[int] = None,
    ):
        if list(handlers_map.keys()) != list(indexers_map.keys()):
            raise ValueError("handlers_map and indexers_map should have identical keys")

        self.handlers_map = handlers_map
        self.indexers_map = indexers_map
        self.max_threads = max_threads
        self.max_failures = max_failures

        self._handler_lut = PatternLUT(
            [(h.pattern, h) for handlers in handlers_map.values() for h in handlers]
        )
        self._pool = ThreadPool(self.max_threads)

    def crawl(
        self, dirpath: StrOrPath
    ) -> Tuple[Dict[str, pa.Table], List[HandlingFailure], CrawlCounts]:
        """
        Crawl a directory, scan for files matching the handler patterns, and process
        each with the assigned handler(s). Insert each record at the index generated by
        the assigned indexer.

        Args:
            dirpath: path to directory to crawl

        Returns:
            A mapping of table names to PyArrow ``Tables``, a list of handling failures
            (if any), and some count stats.
        """
        for indexer in self.indexers_map.values():
            indexer.set_root(dirpath)
        taskfn = self._make_taskfn(self.indexers_map)

        constants = {"_dir": str(dirpath)}
        tables = self._new_tables(constants)

        counts = CrawlCounts()
        errors = []

        # Lazy streaming evaluation using ``imap_unordered`` applied to a generator.
        # NOTE: Using ``ThreadPool`` instead of the more modern
        # ``ThreadPoolExecutor`` bc it doesn't support this lazy evaluation.
        for val in self._pool.imap_unordered(
            taskfn, self._scan_for_matches(dirpath, self._handler_lut)
        ):
            handler, data, err = val
            counts.total += 1
            if data is not None:
                key, record = data
                tables[handler.group].put(key, record, handler.label)
                counts.process += 1
            if err is not None:
                errors.append(err)
                counts.error += 1
                if self.max_failures and len(errors) >= self.max_failures:
                    raise RuntimeError(
                        f"Max number of failures {self.max_failures} exceeded"
                    )

        tables = {group: table.as_table() for group, table in tables.items()}
        return tables, errors, counts

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
        handler_lut: PatternLUT[HandlerTuple],
    ) -> Iterator[Tuple[str, HandlerTuple]]:
        """
        Scan a directory for files matching the handler patterns.
        """
        # TODO: A sub-class might generalize this to scan S3.
        for subdir, _, files in os.walk(dirpath, followlinks=True):
            for f in files:
                path = Path(subdir) / f
                for _, handler in handler_lut.lookup(path):
                    yield str(path), handler

    @staticmethod
    def _make_taskfn(indexers_map: Mapping[str, Indexer]):
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
                f"\tindexer: {repr(indexer)}\n\n" + traceback.format_exc() + "\n"
            )
            key = None
            err = HandlingFailure(path, handler.pattern, repr(indexer), repr(exc))

        if key is None:
            return None, err

        try:
            record = handler.handler(path)
            err = None
        except Exception as exc:
            logging.warning(
                "Handler failed to process a file\n"
                f"\tdirpath: {indexer.root}\n"
                f"\tpath: {path}\n"
                f"\tpattern: {handler.pattern}\n"
                f"\thandler: {repr(handler.handler)}\n\n"
                + traceback.format_exc()
                + "\n"
            )
            record = None
            err = HandlingFailure(
                path, handler.pattern, repr(handler.handler), repr(exc)
            )

        data = None if record is None else (key, record)
        return data, err

    def close(self):
        """
        Close the crawler's thread pool
        """
        self._pool.close()

    def __enter__(self) -> "Crawler":
        return self

    def __exit__(self, *args):
        self.close()
