import logging
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional

from bids2table import RecordDict, StrOrPath, find_file
from bids2table.loaders import Loader
from bids2table.schema import PandasType, Schema
from bids2table.utils import Catalog, combined_suffix, set_overlap

__all__ = [
    "HANDLER_CATALOG",
    "Handler",
    "HandlerLUT",
    "LookupResult",
]

HANDLER_CATALOG: Catalog["Handler"] = Catalog()


class Handler:
    """
    An abstract handler to extract contents from files and transform them into a
    ``dict`` record mapping column names to data values.

    TODO:
        [ ] retry logic?
    """

    def __init__(
        self,
        loader: Loader,
        name: str,
        pattern: str,
        *,
        fields: Optional[Dict[str, PandasType]] = None,
        example: Optional[StrOrPath] = None,
        label: Optional[str] = None,
        metadata: Dict[str, Any] = {},
        rename_map: Dict[str, str] = {},
        convert_dtypes: bool = True,
        register: bool = True,
    ):
        if pattern[-1] in "]*?":
            raise ValueError(
                f"Pattern '{pattern}' should not end in a special character"
            )

        self.loader = loader
        self.name = name
        self.pattern = pattern
        self.label = name if label is None else label
        self.rename_map = rename_map

        if fields is not None:
            self.schema = Schema(fields, metadata)
        elif example is not None:
            record = self.load_example(loader, example)
            record = self.apply_renaming(rename_map, record)
            if record is None:
                raise ValueError(f"Example {example} is invalid for loader {loader}")
            self.schema = Schema.from_record(
                record, metadata=metadata, convert=convert_dtypes
            )
        else:
            raise ValueError("One of 'fields' or 'example' is required")

        if register:
            self.register()

    def __call__(self, path: StrOrPath) -> Optional[RecordDict]:
        record = self.loader(path)
        record = self.apply_renaming(self.rename_map, record)
        if record is not None:
            overlap = set_overlap(self.schema.columns(), record.keys())
            if overlap < 1:
                logging.warning(
                    f"Field overlap between record and schema is only {overlap:.2f}\n"
                    f"\tpath: {path}\n"
                    f"\thandler: {self.name}"
                )
        return record

    def register(self):
        """
        Register the handler in the shared catalog.
        """
        HANDLER_CATALOG.register(self.name, self)

    @classmethod
    def load_example(cls, loader: Loader, path: StrOrPath) -> Optional[RecordDict]:
        if resources.is_resource(__package__, str(path)):
            with resources.path(__package__, str(path)) as p:
                return loader(p)
        else:
            p = find_file(path)
            if p is None:
                raise FileNotFoundError(f"Example {path} not found")
            return loader(p)

    @staticmethod
    def apply_renaming(
        rename_map: Dict[str, str], record: Optional[RecordDict]
    ) -> Optional[RecordDict]:
        if record is None:
            return record
        else:
            return {rename_map.get(k, k): v for k, v in record.items()}


class LookupResult(NamedTuple):
    group: str
    handler: Handler


class HandlerLUT:
    """
    Lookup table mapping glob file patterns to handlers.
    """

    def __init__(self, handlers_map: Dict[str, List[Handler]]):
        self.handlers_map = handlers_map

        # Organize handlers by suffix for faster lookup.
        self._handlers_by_suffix: Dict[str, List[LookupResult]] = defaultdict(list)
        for group, handlers in handlers_map.items():
            for handler in handlers:
                suffix = combined_suffix(handler.pattern)
                self._handlers_by_suffix[suffix].append(LookupResult(group, handler))

    def lookup(self, path: StrOrPath) -> Iterator[LookupResult]:
        """
        Lookup one or more handlers for a path by glob pattern matching. Yields tuples
        of ``(group, handler)``.
        """
        path = Path(path)
        suffix = combined_suffix(path)
        for result in self._handlers_by_suffix[suffix]:
            # TODO: could consider generalizing this pattern matching to:
            #   - tuples of globs
            #   - arbitrary regex
            # But better to keep things simple for now.
            if path.match(result.handler.pattern):
                yield result
