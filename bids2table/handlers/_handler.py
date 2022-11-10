import logging
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

from bids2table import RecordDict, StrOrPath, find_file
from bids2table._utils import Catalog, set_overlap
from bids2table.loaders import Loader
from bids2table.schema import PandasType, Schema

__all__ = ["Handler", "HandlerLUT", "LookupResult"]

HANDLER_CATALOG: Catalog["Handler"] = Catalog()


class Handler:
    """
    An abstract handler to extract contents from files and transform them into a
    ``dict`` record mapping column names to data values.

    TODO:
        [x] Make pattern internal to the handler. These logically go together. I think
            the user would like to define them in the same place.
        [x] initializing schema from an example
            - I think this would be a convenient feature. Leverages pandas type
              inference, which is pretty good, to automate tedious schema definition.
              Makes it easy to update schemas in response to upstream changes (just
              update the example). Also should help avoid typos.

              Not to mention it makes it easy for users to do the right thing and
              include types up front. This way we can fully avoid the issue of varying
              types between batches.
        [ ] retry logic?
    """

    def __init__(
        self,
        loader: Loader,
        *,
        name: str,
        pattern: str,
        fields: Optional[Dict[str, PandasType]] = None,
        example: Optional[StrOrPath] = None,
        label: Optional[str] = None,
        metadata: Dict[str, Any] = {},
        rename_map: Dict[str, str] = {},
        convert_dtypes: bool = True,
        register: bool = True,
    ):
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
            HANDLER_CATALOG.register(self.name, self)

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

    def __init__(
        self,
        handlers_map: Dict[str, List[Handler]],
    ):
        self.handlers_map = handlers_map

    def lookup(self, path: StrOrPath) -> Iterator[LookupResult]:
        """
        Lookup one or more handlers for a path by glob pattern matching. Yields tuples
        of ``(group, handler)``.
        """
        path = Path(path)
        # TODO: this will be expensive if you have a lot of files and handlers. One way
        # to save work is to group handlers by extenstion, and then match a file to the
        # unique extension. Most files will likely match nothing.
        for group, handler in self._iterate():
            # TODO: could consider generalizing this pattern matching to:
            #   - tuples of globs
            #   - arbitrary regex
            # But better to keep things simple for now.
            if path.match(handler.pattern):
                yield LookupResult(group, handler)

    def _iterate(self) -> Iterator[Tuple[str, Handler]]:
        return (
            (group, handler)
            for group, handlers in self.handlers_map.items()
            for handler in handlers
        )
