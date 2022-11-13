from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa

from bids2table import Key
from bids2table.schema import Schema


class Table:
    """
    A table consisting of one or more column groups that you build incrementally.

    Args:
        column_groups: A ``dict`` mapping column group labels to an associated schema
        index_names: Name(s) for the index column(s).
        constants: A ``dict`` mapping column labels to constant values.
    """

    def __init__(
        self,
        column_groups: Dict[str, Schema],
        index_names: List[str],
        constants: Optional[Dict[str, Any]] = None,
    ):
        if len(column_groups) == 0:
            raise ValueError("At least one column group required")
        if len(index_names) == 0:
            raise ValueError("At least one index name required")

        self.column_groups = column_groups
        self.index_names = index_names
        self.constants = constants
        self._table: Dict[str, Dict[Key, List[Any]]] = defaultdict(dict)

    def put(self, key: Key, column_group: str, record: Dict[str, Any]):
        """
        Insert a ``record`` at the row ``key`` under ``column_group``.
        """
        key = self._check_key(key)
        schema = self.column_groups[column_group]
        row = [record.get(col) for col in schema.columns()]
        self._table[column_group][key] = row

    def _check_key(self, key: Key) -> Key:
        if isinstance(key, tuple):
            if len(key) != len(self.index_names):
                raise ValueError(
                    f"Invalid key {key}; expected same length as index_names "
                    f"{self.index_names}"
                )
        else:
            if len(self.index_names) != 1:
                raise ValueError(
                    f"Invalid key {key}; a tuple is required unless there is a "
                    "single index column"
                )
            key = (key,)

        if not all(isinstance(ki, (int, str)) for ki in key):
            raise TypeError(f"Invalid key {key}; expected only str and int")

        if len(self.index_names) == 1:
            key = key[0]
        return key

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the table to a pandas ``DataFrame``.

        - Any missing rows for any of the column groups are filled with nulls.
        - The hierarchical index is reset as a column group with the label ``"_index"``.
        - Any constant columns specified by ``constants`` are added.
        """
        # construct table for each column group
        column_tables = []
        for column_group, schema in self.column_groups.items():
            df = pd.DataFrame.from_dict(
                self._table[column_group], orient="index", columns=schema.columns()
            )
            df = df.astype(schema.fields)
            df.index.names = self.index_names
            column_tables.append(df)
        # concatenate column group tables
        df = pd.concat(
            column_tables, axis=1, keys=list(self.column_groups.keys()), sort=True
        )
        # reset hierarchical index to columns
        df = df.reset_index(col_level=1, col_fill="_index")
        df.loc[:, ("_index", slice(None))] = df.loc[
            :, ("_index", slice(None))
        ].convert_dtypes()
        # add constant columns
        for col in reversed(list(self.constants.keys())):
            df.insert(0, col, self.constants[col])
            df.loc[col] = df.loc[col].convert_dtypes()
        return df

    def metadata(self) -> Dict[str, Any]:
        """
        Combined metadata for each column group.
        """
        metadata = {
            f"{col}.{k}": v
            for col, schema in self.column_groups.items()
            for k, v in schema.metadata.items()
        }
        return metadata

    def to_pyarrow(self) -> pa.Table:
        """
        Convert the table to a PyArrow ``Table``.

        The table is first converted to a pandas ``DataFrame`` via ``to_pandas()``.
        """
        tab = self.to_pandas()
        tab = pa.Table.from_pandas(tab)
        tab = tab.replace_schema_metadata(self.metadata())
        return tab
