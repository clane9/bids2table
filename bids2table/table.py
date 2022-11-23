import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import pyarrow as pa

from bids2table import RecordDict
from bids2table.schema import concat_schemas


class IncrementalTable:
    """
    A table consisting of one or more column groups that you build incrementally.

    Args:
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
        self._table: Dict[Tuple[Any, ...], RecordDict] = defaultdict(dict)

    def put(self, key: RecordDict, record: RecordDict, group: Optional[str] = None):
        """
        Insert ``record`` at the row indexed by ``key`` under ``group``.
        """
        key_tup = tuple(key.values())
        row = self._table[key_tup]
        if len(row) == 0:
            # initialize the index entries and constants on row creation
            row.update(self._prepend_prefix(key, self.INDEX_PREFIX))
            row.update(self.constants)

        if group is not None:
            assert group in self.groups, f"Unrecognized group '{group}'"
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
        row.update(record)

    @classmethod
    def _prepend_prefix(cls, record: RecordDict, prefix: str) -> RecordDict:
        """
        Prepend a prefix to all the record keys.
        """
        return {f"{prefix}{cls.SEP}{k}": v for k, v in record.items()}

    def as_table(self) -> pa.Table:
        """
        Convert the table to a pyarrow ``Table``.
        """
        table = pa.Table.from_pylist(
            [self._table[k] for k in sorted(self._table.keys())],
            schema=self._combined_schema,
        )
        return table
