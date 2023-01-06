import logging
from typing import Any, Dict, Optional, Tuple, Union

import pyarrow as pa

from bids2table.extensions import PaPyExtensionType
from bids2table.schema import concat_schemas
from bids2table.types import RecordDict


class IncrementalTable:
    """
    A pyarrow table that you can build up incrementally.

    Args:
        index_schema: Schema for the index columns.
        schema: A schema or group of schemas for the data columns.
        constants: A ``dict`` mapping column labels to constant values.
    """

    SEP = "__"
    INDEX_PREFIX = "_index"

    def __init__(
        self,
        index_schema: pa.Schema,
        schema: Union[pa.Schema, Dict[str, pa.Schema]],
        constants: Optional[Dict[str, Any]] = None,
    ):
        self.index_schema = index_schema
        self.schema = schema
        self.constants = constants

        if isinstance(schema, dict):
            self.groups = set(schema.keys())
            combined_schema = concat_schemas(schema, sep=self.SEP)
        else:
            self.groups = set()
            combined_schema = schema

        if constants is not None:
            constants_schema = pa.schema(
                {k: pa.from_numpy_dtype(type(v)) for k, v in constants.items()}
            )
            combined_schema = concat_schemas([constants_schema, combined_schema])
        self._combined_schema = concat_schemas(
            [index_schema, combined_schema],
            [self.INDEX_PREFIX, None],
            sep=self.SEP,
        )
        self._columns = set(self._combined_schema.names)
        self._table: Dict[Tuple[Any, ...], RecordDict] = {}

    def put(self, key: RecordDict, record: RecordDict, group: Optional[str] = None):
        """
        Insert ``record`` at the row indexed by ``key`` under ``group``.
        """
        key_tup = tuple(key[k] for k in self.index_schema.names)
        row = self._table.get(key_tup, {})
        if len(row) == 0:
            # initialize the index entries and constants on row creation
            row.update(self._prepend_prefix(key, self.INDEX_PREFIX))
            row.update(self.constants)

        if group is not None:
            if group not in self.groups:
                raise ValueError(f"Unrecognized group '{group}'")
            record = self._prepend_prefix(record, group)

        cols = set(record.keys())
        outlier_cols = cols.difference(self._columns)
        if outlier_cols:
            raise ValueError(
                "Record contains columns not in the schema:\n"
                f"\t{', '.join(outlier_cols)}"
            )
        updating_cols = cols.intersection(row.keys())
        if updating_cols:
            logging.warning(
                f"Overwriting the following table columns at row: {key}\n"
                f"\t{', '.join(updating_cols)}"
            )

        record = self._pack_ext_data(record, self._combined_schema)

        row.update(record)
        self._table[key_tup] = row

    @classmethod
    def _prepend_prefix(cls, record: RecordDict, prefix: str) -> RecordDict:
        """
        Prepend a prefix to all the record keys.
        """
        return {f"{prefix}{cls.SEP}{k}": v for k, v in record.items()}

    @staticmethod
    def _pack_ext_data(record: RecordDict, schema: pa.Schema) -> RecordDict:
        """
        For any fields with extension types, pack the data for consumption in pyarrow.
        """
        record_ = {}
        for k, v in record.items():
            typ = schema.field(k).type
            if isinstance(typ, PaPyExtensionType):
                v = typ.pack(v)
            record_[k] = v
        return record_

    def as_table(self) -> pa.Table:
        """
        Convert the table to a pyarrow ``Table``.
        """
        table = pa.Table.from_pylist(
            list(self._table.values()), schema=self._combined_schema
        )
        sort_by = self._prepend_prefix(
            {name: "ascending" for name in self.index_schema.names}, self.INDEX_PREFIX
        )
        table = table.sort_by(list(sort_by.items()))
        return table
