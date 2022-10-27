import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import pyarrow as pa


class Field(NamedTuple):
    name: str
    dtype: Optional[type] = None


class Schema:
    """
    A table schema following `pyarrow Schema`_

    .. _pyarrow Schema: https://arrow.apache.org/docs/python/data.html#schemas

    TODO:
        [x] check/coerce that a record conforms to a schema?
        [ ] record which fields need serialization?
        [ ] cast a record to pyarrow?
        [x] allow loose typing, e.g. with Any?
    """

    def __init__(
        self,
        fields: List[Union[str, Tuple[str, type], Field]],
        metadata: Dict[str, Any],
    ):
        self.fields: List[Field] = []
        for field in fields:
            if not isinstance(field, tuple):
                field = (field,)
            self.fields.append(Field(*field))

        self.metadata = metadata

    def columns(self) -> List[str]:
        return [field.name for field in self.fields]

    def dtypes(self) -> List[type]:
        return [field.dtype for field in self.fields]

    def to_pyarrow(self) -> pa.Schema:
        fields = []
        for field in self.fields:
            name, dtype = field
            if dtype is None:
                raise RuntimeError(
                    "All schema dtypes must be defined to convert to pyarrow"
                )
            if not isinstance(dtype, pa.DataType):
                try:
                    dtype = pa.from_numpy_dtype(dtype)
                except pa.ArrowNotImplementedError:
                    # TODO: how and where should we remember that these columns need
                    # serialization?
                    dtype = pa.binary()
            fields.append((name, dtype))
        schema = pa.schema(fields, metadata=self.metadata)
        return schema

    def null_record(self) -> Dict[str, None]:
        """
        Generate a null record
        """
        return {field.name: None for field in self.fields}

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> "Schema":
        return cls(**schema_dict)

    def coerce(self, record: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
        """
        Coerce a record to match the schema.
        """
        coerced = {}
        for field in self.fields:
            value = record.get(field.name)
            if not (field.dtype is None or isinstance(value, field.dtype)):
                if strict:
                    # TODO: should we complain if the field.dtype is None in this case?
                    value = field.dtype(value)
                else:
                    logging.warning(
                        "Value for field '%s' has dtype '%s'; expected '%s'",
                        field.name,
                        type(value),
                        field.dtype,
                    )
            coerced[field.name] = value
        return coerced
