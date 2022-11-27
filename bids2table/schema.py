import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

from bids2table import RecordDict

__all__ = [
    "create_schema",
    "get_dtype",
    "get_fields",
    "cast_to_schema",
    "concat_schemas",
]

DataType = Union[str, pa.DataType, np.dtype]
Fields = Union[List[pa.Field], List[Tuple[str, DataType]], Dict[str, DataType]]


def create_schema(
    fields: Fields, metadata: Optional[Dict[str, str]] = None
) -> pa.Schema:
    """
    Similar to ``pa.schema()`` but with extended type inference.

    See ``get_dtype()`` for details about the supported type inference.

    TODO: It would be nice if there could be some round trip functionality for:
        - config ->
        - schema ->
        - string ->
        - config

    This way you could e.g. infer a complex schema from an example offline, and then fix
    it in the config.
    """
    if isinstance(fields, dict):
        fields = list(fields.items())
    schema = pa.schema(map(_as_field, fields), metadata=metadata)
    return schema


def _as_field(f: Union[pa.Field, Tuple[str, DataType]]) -> pa.Field:
    if not isinstance(f, pa.Field):
        name, dtype = f
        f = pa.field(name, get_dtype(dtype))
    return f


_NESTED_TYPES = [
    (re.compile(r"^(?:list|array)\[(.+)\]$"), pa.list_),
]


def get_dtype(dtype: DataType) -> pa.DataType:
    """
    Attempt to infer the PyArrow dtype from string type alias or numpy dtype.

    The list of available pyarrow type aliases is available `here`_.

    The following nested type aliases are also supported:

    - ``"list[<type>]"`` -> ``pa.list_(<type>)``
    - ``"array[<type>]"`` -> ``pa.list_(<type>)``

    .. _here: https://github.com/apache/arrow/blob/go/v10.0.0/python/pyarrow/types.pxi#L3159

    TODO: Could go crazy and supported struct types with nested field sequences. Not now
    though. Although these could happen when schemas are initialized by example.
    """
    try:
        dtype = pa.lib.ensure_type(dtype)
        return dtype
    except Exception:
        pass

    try:
        dtype = pa.from_numpy_dtype(dtype)
        return dtype
    except Exception:
        pass

    if not isinstance(dtype, str):
        raise ValueError(f"Unsupported dtype {dtype}")

    for pattern, dtype_ in _NESTED_TYPES:
        match = pattern.match(dtype)
        if match is not None:
            args = tuple(map(get_dtype, match.groups()))
            dtype = dtype_(*args)
            return dtype

    raise ValueError(f"Unsupported dtype '{dtype}'")


def get_fields(schema: pa.Schema) -> Dict[str, pa.DataType]:
    """
    Extract a dict mapping column names to pyarrow dtypes from a pyarrow schema.
    """
    fields = {name: schema.field(name).type for name in schema.names}
    return fields


def cast_to_schema(
    record: RecordDict, schema: pa.Schema, safe: bool = True, with_null: bool = True
) -> RecordDict:
    """
    Cast a record dict to match a pyarrow schema.

    - keys are re-ordered
    - extra keys are discarded
    - data values are cast to target types (safely if ``safe`` is ``True``).

    If ``with_null`` is ``True``, mising fields are entered as ``None``, and left as
    missing otherwise.

    Raises ``ArrowInvalid`` if a data value cannot be converted.
    """
    record_: RecordDict = {}
    for name, dtype in get_fields(schema).items():
        if record.get(name) is not None:
            # TODO: Will need to extend here for any extension types
            if not safe or isinstance(dtype, pa.ListType):
                val = pa.scalar(record[name], type=dtype)
            else:
                val = pa.scalar(record[name]).cast(dtype)
            record_[name] = _scalar_as_py(val)
        elif with_null:
            record_[name] = None
    return record_


def _scalar_as_py(scalar: pa.Scalar) -> Any:
    """
    Recursively convert a pyarrow scalar to a standard python object, converting structs
    to dicts and lists to numpy arrays. All other types are converted using pyarrow's
    builtin ``Scalar.as_py()`` method.

    .. note::
        We use this rather than simply call ``Scalar.as_py()`` for all in order to
        ensure pyarrow lists convert to numpy arrays.
    """
    if isinstance(scalar, pa.StructScalar):
        py_scalar = {k: _scalar_as_py(v) for k, v in scalar.items()}
    elif isinstance(scalar, pa.ListScalar):
        py_scalar = scalar.values.to_numpy()
    else:
        py_scalar = scalar.as_py()
    return py_scalar


def concat_schemas(
    schemas: Union[List[pa.Schema], Dict[str, pa.Schema]],
    keys: Optional[List[str]] = None,
    sep: str = "__",
) -> pa.Schema:
    """
    Concatenate pyarrow schemas. If ``keys`` are passed, these are used to identify each
    schema group. The key for each schema will be prepended to each of the schema's
    fields and metadata entries, separated by ``sep``. If ``schemas`` is a dict, the
    dict keys are used as the ``keys``.
    """
    if isinstance(schemas, dict):
        keys = list(schemas.keys())
        schemas = list(schemas.values())
    if keys is not None and len(keys) != len(schemas):
        raise ValueError("One key per schema is required")

    fields: Dict[str, DataType] = {}
    metadata: Dict[str, str] = {}

    for ii, schema in enumerate(schemas):
        key = keys[ii] if keys is not None else None

        schema_fields = get_fields(schema)
        for k, v in schema_fields.items():
            if key is not None:
                k = f"{key}{sep}{k}"
            if k in fields:
                raise ValueError(f"Duplicate column '{k}'")
            fields[k] = v

        k_: bytes
        if schema.metadata is not None:
            for k_, v in schema.metadata.items():
                k = k_.decode()
                if key is not None:
                    k = f"{key}{sep}{k}"
                if k in metadata:
                    raise ValueError(f"Duplicate metadata key '{k}'")
                metadata[k] = v
    schema = pa.schema(fields, metadata=metadata)
    return schema
