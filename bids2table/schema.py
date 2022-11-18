from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from bids2table import RecordDict

PandasType = Union[str, type, np.dtype]


class Schema:
    """
    A table schema patterned after `pyarrow Schema`_

    .. _pyarrow Schema: https://arrow.apache.org/docs/python/data.html#schemas

    TODO:
        [x] check/coerce that a record conforms to a schema?
        [x] record which fields need serialization?
            - implicit with object types
        [x] cast a record to pyarrow?
            - maybe, but somewhere else
        [x] allow loose typing, e.g. with Any?
            - no. types are required. but you can infer from a valid example.

    TODO:
        [x] `from_pandas` interface to infer from a pandas df example.

    TODO: How do I plan to get the metadata into the parquet? Would need to carry the
    pyarrow schema all the way through. Maybe I should convert to pyarrow format sooner.
    Perhaps in the ``GroupedRecordTable``.
    """

    def __init__(
        self,
        fields: Dict[str, PandasType],
        metadata: Dict[str, Any] = {},
    ):
        self.fields = fields
        self.metadata = metadata

    def columns(self) -> List[str]:
        return list(self.fields.keys())

    def dtypes(self) -> pd.Series:
        pd.Series(self.fields)

    def empty(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=self.columns())
        df = self.cast(df)
        return df

    def to_pyarrow(self) -> pa.Schema:
        schema = pa.Schema.from_pandas(self.empty())
        schema = schema.with_metadata(self.metadata)
        return schema

    def matches(self, df: pd.DataFrame) -> bool:
        """
        Check if a ``DataFrame`` matches the schema.
        """
        return (df.dtypes == self.dtypes()).all()

    def cast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast a ``DataFrame`` to match the schema's types.

        Unlike ``coerce``, this will raise an error if ``df`` does not match the
        schema's columns.
        """
        return df.astype(self.fields)

    def coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce a ``DataFrame`` to match the schema.
        """
        if self.matches(df):
            return df
        df = df.copy()
        columns = np.array(self.columns())
        missing_cols = np.setdiff1d(columns, df.columns)
        if len(missing_cols) > 0:
            df.loc[:, missing_cols] = None
        df = df.loc[:, columns]
        df = self.cast(df)
        return df

    @classmethod
    def from_record(
        cls,
        record: RecordDict,
        metadata: Dict[str, Any] = {},
        convert: bool = True,
    ) -> "Schema":
        """
        Infer a schema from a record dict.

        If ``convert`` is ``True``, columns will be converted to their "best possible"
        (nullable) data types.
        """
        df = pd.DataFrame.from_records([record])
        return cls.from_pandas(df, metadata=metadata, convert=convert)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        metadata: Dict[str, Any] = {},
        convert: bool = True,
    ) -> "Schema":
        """
        Infer a schema from a pandas dataframe.

        If ``convert`` is ``True``, columns will be converted to their "best possible"
        (nullable) data types.

        NOTE: ``convert=True`` requires pandas>=1.2.0
        """
        if convert:
            df = df.convert_dtypes()
        return cls(df.dtypes.to_dict(), metadata=metadata)

    @classmethod
    def from_pyarrow(
        cls,
        *,
        schema: Optional[pa.Schema] = None,
        table: Optional[pa.Table] = None,
    ):
        if schema is not None:
            table = schema.empty_table()
        elif table is None:
            raise ValueError("One of 'schema' or 'table' is required")
        df = table.to_pandas()
        return cls.from_pandas(df, metadata=table.schema.metadata, convert=False)
