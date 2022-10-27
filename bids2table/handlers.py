import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from bids2table import PREFIX_SEPARATOR as SEP
from bids2table.schema import Schema

from ._utils import set_overlap

# TODO:
#   [x] abstract handler
#       - must define a schema with names, types, description.
#           - but what if I want a generic nifti handler to be used in multiple places.
#             then the name and description won't be specific?
#           - multiple handlers can be generated for the same loader
#   [x] handler decorator
#   [x] match strategy: all or first
#   [x] handlers can return None if they don't want to emit a record
#   [x] coercing to schema
#   [x] prefixing by handler name for uniqueness


class Handler:
    """
    An abstract handler to extract contents from files and transform them into a
    ``dict`` record mapping column names to data values.

    Can be used as a decorator, e.g.::

        @Handler(
            schema={"age": float, "height": float},
        )
        def json_handler(path):
            with open(path) as f:
                data = json.load(f)
            return data

    As a side-effect the handler is automatically registered to the ``HandlerCatalog``.

    TODO:
        [ ] retry logic?
        [x] rename map, in case you want to change the column names from what's in the
            file, and you don't want to customize the handler func.
        [x] optional prefix in case you want to have multiple handlers with the same
            prefix. On you if you get name clashes!
    """

    def __init__(
        self,
        func: Callable[[Path], Dict[str, Any]],
        *,
        schema: Union[Schema, Dict[str, Any]],
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        rename_map: Dict[str, str] = {},
        strict: bool = False,
    ):
        if not isinstance(schema, Schema):
            schema = Schema.from_dict(schema)
        if name is None:
            name = func.__name__
        if prefix is None:
            prefix = name

        self.func = func
        self.schema = schema
        self.name = name
        self.prefix = prefix
        self.rename_map = rename_map
        self.strict = strict

        HandlerCatalog.register_handler(self, name=name)

    def __call__(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        path = Path(path)
        record = self.func(path)
        if record is not None:
            if set_overlap(self.schema.columns, record.keys()) < 0.5:
                logging.warning(
                    "Less than half of columns in record match the schema\n"
                    f"\tpath: {path}\n"
                    f"\thandler: {self.name}"
                )
            record = {self.rename_map.get(k, k): v for k, v in record.items()}
            # TODO: for columns that can be all None in a batch, will need to do
            # something to ensure that the correct types are set. This could happen:
            #   - in `schema.coerce`
            #   - in `RecordBatchTable.to_pandas` (this seems best with `df.astype`)
            #   - somewhere else?
            #
            # Also have to worry about metadata columns that are all None. Could have
            # the context also define a metadata schema.
            #
            # Another option is to coerce the schema to the first batch in the writer.
            # This would suppress errors but have consequences for the data. In the all
            # None case, the column would be represented as object.
            #   - Implemented this as an option in ``ParquetWriter``. I think worth
            #     having if the user just wants the process to run and deal with column
            #     types later. Can always recover from an object type.
            record = self.schema.coerce(record, strict=self.strict)
            record = self._prepend_prefix(record)
        return record

    def null_record(self) -> Dict[str, None]:
        """
        Generate a null record
        """
        return self._prepend_prefix(self.schema.null_record())

    def _prepend_prefix(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepend the handler name as a prefix to all keys in the record.
        """
        record = {f"{self.prefix}{SEP}{k}": v for k, v in record.items()}
        return record


class HandlerCatalog:
    """
    Catalog of registered handlers.

    TODO:
        [x] update to reflect that handler is a ``Handler`` instance.
    """

    __REGISTERED_HANDLERS: Dict[str, Handler] = {}

    @staticmethod
    def register_handler(handler: Handler):
        """
        Register a handler.
        """
        if not isinstance(handler, Handler):
            raise TypeError("handlers must be of type `Handler`")
        if HandlerCatalog.has(handler.name):
            logging.warning(f"Handler '{handler.name}' already registered; overwriting")
        HandlerCatalog.__REGISTERED_HANDLERS[handler.name] = handler

    @staticmethod
    def get(name: str) -> Optional[Handler]:
        """
        Lookup a handler by name. Returns ``None`` if the handler is not registered.
        """
        return HandlerCatalog.__REGISTERED_HANDLERS.get(name)

    @staticmethod
    def has(name: str) -> bool:
        """
        Check if a handler is registered.
        """
        return name in HandlerCatalog.__REGISTERED_HANDLERS


class HandlerMap:
    """
    Lookup table mapping glob file patterns to handlers.

    TODO:
        [x] call this something else to distinguish from the more typical notion of
            table as in a pandas table. e.g. HandlerLUT, HandlerMap
    """

    def __init__(
        self,
        handler_config: Dict[str, str],
        match_strategy: str = "first",
    ):
        assert match_strategy in {
            "first",
            "all",
        }, f"match_strategy '{match_strategy}' should be 'first' or 'all'"

        self.handler_config = handler_config
        self.match_strategy = match_strategy

        # TODO: more descriptive name, handlers used as list elsewhere
        #   - done
        self._handler_lut: Dict[str, Handler] = {}
        for name in handler_config.values():
            if not HandlerCatalog.has(name):
                raise ValueError(f"Handler '{name}' is not registered")
            self._handler_lut[name] = HandlerCatalog.get(name)

    def lookup(self, path: Union[str, Path]) -> Iterable[Tuple[str, Handler]]:
        """
        Lookup one or more handlers for a path by glob pattern matching. Yields tuples
        of ``(pattern, handler)``.

        If ``match_strategy`` is ``'first'``, only the first match is yielded. Otherwise
        all matches are yielded.

        TODO:
            [x] Do we also want to yield the pattern?
                - yes
        """
        path = Path(path)
        for pattern, name in self.handler_config.items():
            if path.match(pattern):
                yield pattern, self._handler_lut[name]
                if self.match_strategy == "first":
                    break

    def handler_names(self) -> List[str]:
        """
        Return a list of handler names (in the original config order).
        """
        return list(self._handler_lut.keys())

    def handlers(self) -> List[Handler]:
        """
        Return a list of handler names (in the original config order).
        """
        return list(self._handler_lut.values())
