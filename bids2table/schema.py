from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.dtypes.base import ExtensionDtype

from bids2table import RecordDict

PandasType = Union[str, type, np.dtype, ExtensionDtype]
Fields = Mapping[str, PandasType]

# https://arrow.apache.org/docs/python/pandas.html#nullable-types
PYARROW_PANDAS_DTYPE_MAP = {
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.uint8(): pd.UInt8Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
    pa.float32(): pd.Float32Dtype(),
    pa.float64(): pd.Float64Dtype(),
    pa.string(): pd.StringDtype(),
}


class Schema:
    """
    A table schema patterned after `pyarrow Schema`_

    Args:
        fields: A mapping of column names to datatypes. Datatypes can be any that
            `pandas can handle`_. E.g.

            - primitive types (e.g. ``str``, ``int``, ``float``) and their string
              equivalents (e.g. ``"str"``).

            - numpy dtypes (e.g. ``np.float32``, ``np.uint8``, ``np.datetime64``) and
              their string equivalents.

            - pandas dtypes (including extended nullable dtypes) (e.g.
                ``pd.StringDtype()``, ``pd.Float32Dtype()``, ``pd.CategoricalDtype()``)
                and their string equivalents (``"string"``, ``"Float32"``,
                ``"category"``).

        metadata: Optional metadata. Per column metadata is specified using the column
            name as the key.

    .. _pyarrow Schema: https://arrow.apache.org/docs/python/data.html#schemas
    .. _pandas can handle: https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
    """

    def __init__(
        self,
        fields: Fields,
        metadata: Dict[str, Any] = {},
    ):
        self.fields = {k: self.canonicalize(v) for k, v in fields.items()}
        self.metadata = metadata

    @property
    def columns(self) -> List[str]:
        """
        Schema columns
        """
        return list(self.fields.keys())

    @property
    def dtypes(self) -> pd.Series:
        """
        Schema dtypes, as in ``pd.DataFrame.dtypes``.
        """
        return pd.Series(self.fields)

    def empty(self) -> pd.DataFrame:
        """
        Return an empty dataframe matching the schema.
        """
        df = pd.DataFrame(columns=self.columns)
        df = self.cast(df)
        return df

    def to_pyarrow(self) -> pa.Schema:
        """
        Return an equivalent pyarrow schema, with metadata.
        """
        schema = pa.Schema.from_pandas(self.empty())
        schema = schema.with_metadata(self.metadata)
        return schema

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

        .. note::
            Nullable extension types (such as ``pd.Int64Dtype()`` rather than ``int``)
            are required for any columns with missing data.
        """
        if self.matches(df):
            return df
        df = df.copy()
        columns = np.array(self.columns)
        missing_cols = np.setdiff1d(columns, df.columns)
        if len(missing_cols) > 0:
            df.loc[:, missing_cols] = None
        df = df.loc[:, columns]
        df = self.cast(df)
        return df

    def convert_dtypes(self) -> "Schema":
        """
            Convert dtypes to their "best possible" (nullable) data types.

        .. warning::
            Not all dtypes are converted correctly. E.g. ``str`` -> ``object``, and (oddly)
            ``np.float32`` -> ``pd.Int64Dtype()``.
        """
        fields = self.empty().convert_dtypes().dtypes.to_dict()
        return Schema(fields, self.metadata.copy())

    def matches(
        self,
        other: Union[
            "Schema", List[RecordDict], RecordDict, pd.DataFrame, pa.Table, pa.Schema
        ],
    ) -> bool:
        """
        Check if a ``DataFrame`` matches the schema.
        """
        if isinstance(other, Schema):
            pass
        elif isinstance(other, (list, dict)):
            other = self.from_records(other)
        elif isinstance(other, pd.DataFrame):
            other = self.from_pandas(other)
        elif isinstance(other, pa.Table):
            other = Schema.from_pyarrow(table=other)
        elif isinstance(other, pa.Schema):
            other = Schema.from_pyarrow(schema=other)
        else:
            raise TypeError(f"Invalid `other` for matching schema ({type(other)})")
        return self.dtypes.equals(other.dtypes)

    @classmethod
    def from_records(
        cls,
        records: Union[RecordDict, List[RecordDict]],
        metadata: Dict[str, Any] = {},
        convert: bool = False,
    ) -> "Schema":
        """
        Infer a schema from a record dict or list of record dicts.

        If ``convert`` is ``True``, columns will be converted to their "best possible"
        (nullable) data types.
        """
        if isinstance(records, dict):
            records = [records]
        df = pd.DataFrame.from_records(records)
        return cls.from_pandas(df, metadata=metadata, convert=convert)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        metadata: Dict[str, Any] = {},
        convert: bool = False,
    ) -> "Schema":
        """
        Infer a schema from a pandas dataframe.

        If ``convert`` is ``True``, columns will be converted to their "best possible"
        (nullable) data types.

        .. note::
            ``convert=True`` requires pandas>=1.2.0
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
        convert: bool = False,
    ):
        """
        Infer a schema from a pyarrow table or schema.

        If ``convert`` is ``True``, columns will be converted to their "best possible"
        (nullable) data types.
        """
        if schema is not None:
            table = schema.empty_table()
        elif table is None:
            raise ValueError("One of 'schema' or 'table' is required")
        df = table.to_pandas(types_mapper=(pyarrow_types_mapper if convert else None))
        return cls.from_pandas(df, metadata=table.schema.metadata, convert=convert)

    @staticmethod
    def canonicalize(dtype: PandasType) -> PandasType:
        """
        Get the canonical pandas dtype object for any valid ``dtype`` representation.
        """
        # TODO: bit of hack. Is there a pandas util for this?
        return pd.Series([], dtype=dtype).dtype

    def __str__(self) -> str:
        return (
            "Schema:\n"
            f"\tfields: {str(self.fields)}\n"
            f"\tmetadata: {str(self.metadata)}"
        )

    def __repr__(self) -> str:
        return f"Schema(fields={repr(self.fields)}, metadata={repr(self.metadata)})"


def pyarrow_types_mapper(dtype: pa.DataType) -> Optional[PandasType]:
    """
    Mapping between pyarrow and pandas dtypes.
    """
    return PYARROW_PANDAS_DTYPE_MAP.get(dtype)
