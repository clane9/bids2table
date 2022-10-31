from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import pyarrow as pa

from bids2table.schema import Schema

IndexType = Union[str, int]
KeyType = Union[IndexType, Tuple[IndexType, ...]]


class Table:
    def __init__(
        self,
        column_groups: Dict[str, Schema],
        index_names: List[str],
    ):
        if len(column_groups) == 0:
            raise ValueError("At least one column group required")
        if len(index_names) == 0:
            raise ValueError("At least one index name required")

        self.column_groups = column_groups
        self.index_names = index_names
        self._table = defaultdict(dict)

    def put(self, key: KeyType, column_group: str, record: Dict[str, Any]):
        key = self._check_key(key)
        schema = self.column_groups[column_group]
        row = [record.get(col) for col in schema.columns()]
        self._table[column_group][key] = row

    def _check_key(self, key: KeyType) -> KeyType:
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
        column_tables = []
        for column_group, schema in self.column_groups.items():
            df = pd.DataFrame.from_dict(
                self._table[column_group], orient="index", columns=schema.columns()
            )
            df = df.astype(schema.fields)
            df.index.names = self.index_names
            column_tables.append(df)
        df = pd.concat(
            column_tables, axis=1, keys=list(self.column_groups.keys()), sort=True
        )
        return df

    def metadata(self) -> Dict[str, Any]:
        metadata = {
            f"{col}.{k}": v
            for col, schema in self.column_groups.items()
            for k, v in schema.metadata.items()
        }
        return metadata

    def to_pyarrow(self) -> pa.Table:
        tab = self.to_pandas()
        tab = pa.Table.from_pandas(tab)
        tab = tab.replace_schema_metadata(self.metadata())
        return tab
