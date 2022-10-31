from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pyarrow as pa

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

    def dtypes(self) -> List[type]:
        return list(self.fields.values())

    def empty(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=self.columns())
        df = df.astype(self.fields)
        return df

    def to_pyarrow(self) -> pa.Schema:
        schema = pa.Schema.from_pandas(self.empty())
        schema = schema.with_metadata(self.metadata)
        return schema

    @classmethod
    def from_record(
        cls,
        record: Dict[str, Any],
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
        fields = list(df.dtypes.to_dict().items())
        return cls(fields, metadata=metadata)
