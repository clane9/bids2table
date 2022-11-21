from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.dtypes.base import ExtensionDtype

from bids2table import RecordDict

PandasType = Union[str, type, np.dtype, ExtensionDtype]
Fields = Mapping[str, PandasType]
TableLike = Union[RecordDict, List[RecordDict], pd.DataFrame, pa.Table]

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


# TODO: we have a challenge with getting some complex data into pyarrow
#   - arbitrary objects are not supported (even though they could be with pickle)
#   - multidimensional arrays are not supported (even though they could probably be)
#   - lists and 1d arrays both map to pa list_ types
#   - dicts map to structs. although they often could map to map_ types
#
# The challenge is all these are represented in pandas as objects. So the pyarrow dtype
# is not determined by the schema alone.
#
# I think balancing all these different representations is over-complicated. Our
# ultimate target is a pyarrow table. Should just commit to using pyarrow internally,
# and have the API be convenient/familiar/general enough.
#
# The api needs to let users specify schemas manually or by example and help users do
# the right thing. Should also be extensible to support more types later on. I think
# most of this can be satisfied using pyarrow Schema and Table functionality. We can get
# rid of all this custom Schema logic.


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

            - pandas extended (nullable) dtypes (e.g.  ``pd.StringDtype()``,
                ``pd.Float32Dtype()``, ``pd.CategoricalDtype()``) and their string
                equivalents (``"string"``, ``"Float32"``, ``"category"``).

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
        df = self._cast(df)
        return df

    def to_pyarrow(self) -> pa.Schema:
        """
        Return an equivalent pyarrow schema, with metadata.
        """
        schema = pa.Schema.from_pandas(self.empty())
        # pyarrow attempts to infer better types for object columns based on contents.
        # In the case of an empty table, this results in columns assigned nulls. Replace
        # this with binary.
        # TODO:
        # for ii, (name, dtype) in enumerate(self.fields.items()):
        #     schema = schema.set(ii, schema.field(ii).with_type(pa.binary()))
        schema = schema.with_metadata(self.metadata)
        return schema

    def _cast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast a ``DataFrame``'s columns to match the schema's types.

        Raises a ``KeyError`` if any of the schema's columns are missing from ``df``.
        """
        return df.astype(self.fields)

    def coerce(self, other: TableLike) -> TableLike:
        """
        Coerce a record dict, list of records, pandas ``DataFrame``, or pyarrow
        ``Table`` to match the schema.

        .. note::
            Nullable extension types (such as ``pd.Int64Dtype()`` rather than ``int``)
            are required for any columns with missing data.
        """
        coerced: TableLike
        if isinstance(other, dict):
            coerced = self._coerce_record(other)
        elif isinstance(other, list):
            coerced = self._coerce_records(other)
        elif isinstance(other, pd.DataFrame):
            coerced = self._coerce_pandas(other)
        elif isinstance(other, pa.Table):
            coerced = self._coerce_pyarrow(other)
        else:
            raise TypeError(f"Invalid `other` for coercing to schema ({type(other)})")
        return coerced

    def _coerce_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce a ``DataFrame`` to match the schema.
        """
        if self.matches(df, strict=True):
            return df
        df = df.copy()
        columns = np.array(self.columns)
        missing_cols = np.setdiff1d(columns, df.columns)
        if len(missing_cols) > 0:
            df.loc[:, missing_cols] = None
        df = df.loc[:, columns]
        df = self._cast(df)
        return df

    def _coerce_record(self, record: RecordDict) -> RecordDict:
        return self._coerce_records([record])[0]

    def _coerce_records(self, records: List[RecordDict]) -> List[RecordDict]:
        df = pd.DataFrame.from_records(records)
        df = self._coerce_pandas(df)
        records = df.to_dict(orient="records")
        return records

    def _coerce_pyarrow(self, table: pa.Table) -> pa.Table:
        df = table.to_pandas(types_mapper=pyarrow_types_mapper)
        df = self._coerce_pandas(df)
        table = pa.Table.from_pandas(df, preserve_index=True, schema=self.to_pyarrow())
        return table

    def convert_dtypes(self) -> "Schema":
        """
        Convert dtypes to their "best possible" (nullable) data types.

        .. warning::
            Not all dtypes are currently converted correctly. E.g. ``str`` ->
            ``object``, and (oddly) ``np.float32`` -> ``pd.Int64Dtype()``.
        """
        fields = self.empty().convert_dtypes().dtypes.to_dict()
        return Schema(fields, self.metadata.copy())

    def matches(
        self,
        other: Union["Schema", TableLike, pa.Schema],
        strict: bool = True,
    ) -> bool:
        """
        Check if another ``Schema``, record dict, list of records, pandas ``DataFrame``,
        or pyarrow ``Table`` or ``Schema`` match this schema.

        If ``strict`` is ``True``. The ``other``'s columns and dtypes must match the
        schema *exactly*. Otherwise, we test only if the columns match and their dtypes
        can be *cast* to this schema's dtypes.
        """
        schema: Schema
        if isinstance(other, Schema):
            schema = other
        elif isinstance(other, dict):
            schema = self.from_record(other)
        elif isinstance(other, list):
            schema = self.from_records(other)
        elif isinstance(other, pd.DataFrame):
            schema = self.from_pandas(other)
        elif isinstance(other, pa.Table):
            schema = Schema.from_pyarrow(table=other)
        elif isinstance(other, pa.Schema):
            schema = Schema.from_pyarrow(schema=other)
        else:
            raise TypeError(f"Invalid `other` for matching schema ({type(other)})")

        if strict:
            matches = self.dtypes.equals(schema.dtypes)
        else:
            matches = self.columns == schema.columns
            if matches:
                try:
                    self._cast(schema.empty())
                except Exception:
                    matches = False
        return matches

    @classmethod
    def from_record(
        cls,
        record: RecordDict,
        metadata: Dict[str, Any] = {},
        convert: bool = False,
    ) -> "Schema":
        """
        Infer a schema from a record dict.

        If ``convert`` is ``True``, columns will be converted to their "best possible"
        (nullable) data types.
        """
        if not isinstance(record, dict):
            raise TypeError(f"Expected a record dict, got {type(record)}")
        return cls.from_records([record], metadata=metadata, convert=convert)

    @classmethod
    def from_records(
        cls,
        records: List[RecordDict],
        metadata: Dict[str, Any] = {},
        convert: bool = False,
    ) -> "Schema":
        """
        Infer a schema from a list of record dicts.

        If ``convert`` is ``True``, columns will be converted to their "best possible"
        (nullable) data types.
        """
        if not isinstance(records, list):
            raise TypeError(f"Expected a list of record dicts, got {type(records)}")
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
        #   - yes `pandas.core.dtypes.common.pandas_dtype`
        return pd.Series([], dtype=dtype).dtype

    def __repr__(self) -> str:
        return f"Schema(fields={repr(self.fields)}, metadata={repr(self.metadata)})"


def pyarrow_types_mapper(dtype: pa.DataType) -> Optional[PandasType]:
    """
    Mapping between pyarrow and pandas dtypes.
    """
    return PYARROW_PANDAS_DTYPE_MAP.get(dtype)
