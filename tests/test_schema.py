from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pytest import FixtureRequest

from bids2table.schema import Fields, Schema, TableLike


@pytest.fixture(params=["prim", "prim_str", "np", "np_str", "pd", "pd_str", "convert"])
def field_type(request: FixtureRequest) -> str:
    return request.param


def _fields(_field_type: str) -> Fields:
    fields: Fields
    if _field_type == "prim":
        fields = {"a": str, "b": int, "c": float, "d": object}
    elif _field_type == "prim_str":
        fields = {"a": "str", "b": "int", "c": "float", "d": "object"}
    elif _field_type == "np":
        fields = {
            "a": np.str_,
            "b": np.int32,
            "c": np.float32,
            "d": np.dtype("datetime64[ns]"),
        }
    elif _field_type == "np_str":
        fields = {"a": "U", "b": "int32", "c": "float32", "d": "datetime64[ns]"}
    elif _field_type == "pd":
        fields = {
            "a": pd.StringDtype(),
            "b": pd.Int32Dtype(),
            "c": pd.Float32Dtype(),
            "d": pd.CategoricalDtype(),
        }
    elif _field_type == "pd_str":
        fields = {"a": "string", "b": "Int32", "c": "Float32", "d": "category"}
    elif _field_type == "convert":
        # NOTE: I would like to not use pandas types for some of these, but the pandas
        # type conversion doesn't work for the corresponding numpy or primitive types,
        # at least in v1.3.4.
        fields = {"a": "string", "b": "int32", "c": "Float32", "d": "category"}
    else:
        raise NotImplementedError(f"field_type '{field_type}' not implemented")
    return fields


@pytest.fixture
def fields(field_type: str) -> Fields:
    return _fields(field_type)


@pytest.fixture(params=["record", "records", "pandas", "pyarrow"])
def table_type(request: FixtureRequest) -> str:
    return request.param


def _table_and_schemas(_table_type: str) -> Tuple[TableLike, Schema, Schema, Schema]:
    schema = Schema({"a": "object", "b": "int", "c": "float", "d": "object"})
    schema_convert = Schema(
        {"a": "string", "b": "Int64", "c": "Float64", "d": "object"}
    )
    schema_coerce = Schema(
        # Includes:
        #   - reordering
        #   - type conversion (int -> Int64)
        #   - reduced precision (float -> float16)
        #   - new columns
        {"c": "float16", "b": "Int64", "d": "object", "e": "datetime64[ns]"}
    )
    records = [
        {"a": "abc", "b": 123, "c": 9.99, "d": np.zeros(3)},
        {"a": "def", "b": 456, "c": 1.11, "d": np.ones(3)},
    ]
    table: TableLike
    if _table_type == "record":
        table = records[0]
    elif _table_type == "records":
        table = records
    elif _table_type == "pandas":
        table = pd.DataFrame.from_records(records)
    elif _table_type == "pyarrow":
        table = pa.Table.from_pylist(records)
    else:
        raise NotImplementedError(f"table_type '{_table_type}' not implemented")
    return table, schema, schema_convert, schema_coerce


@pytest.fixture
def table_and_schemas(table_type: str) -> Tuple[TableLike, Schema, Schema, Schema]:
    return _table_and_schemas(table_type)


def test_schema_empty(fields: Fields):
    schema = Schema(fields)
    empty = schema.empty()
    assert empty.columns.equals(pd.Index(schema.columns)), "empty columns mismatch"
    assert empty.dtypes.equals(schema.dtypes), "empty dtypes mismatch"
    assert schema.matches(empty), "empty mismatch"


@pytest.mark.parametrize(
    "_field_type",
    ["prim_str", "np_str", "pd_str"],
)
def test_schema_string_equivalence(_field_type: str):
    fields = _fields(_field_type)
    schema = Schema(fields)
    fields2 = _fields(_field_type[:-4])
    schema2 = Schema(fields2)
    assert schema.matches(schema2)


def test_schema_conversion():
    fields = _fields("convert")
    fields2 = _fields("pd")
    schema = Schema(fields).convert_dtypes()
    schema2 = Schema(fields2)
    assert schema.matches(schema2)


@pytest.mark.parametrize("convert", [False, True])
def test_schema_constructors(
    table_and_schemas: Tuple[TableLike, Schema, Schema, Schema], convert: bool
):
    table, schema, schema_convert, _ = table_and_schemas
    schema = schema_convert if convert else schema
    schema3 = None
    if isinstance(table, dict):
        schema2 = Schema.from_record(table, convert=convert)
    elif isinstance(table, list):
        schema2 = Schema.from_records(table, convert=convert)
    elif isinstance(table, pd.DataFrame):
        schema2 = Schema.from_pandas(table, convert=convert)
    elif isinstance(table, pa.Table):
        schema2 = Schema.from_pyarrow(table=table, convert=convert)
        schema3 = Schema.from_pyarrow(schema=table.schema, convert=convert)
    else:
        raise TypeError(f"table has unexpected type {type(table)}")
    assert schema.matches(schema2)
    assert schema3 is None or schema.matches(schema3)


def test_schema_matches(table_and_schemas: Tuple[TableLike, Schema, Schema, Schema]):
    table, schema, schema_convert, schema_coerce = table_and_schemas
    assert schema.matches(table)
    assert not schema_convert.matches(table)
    assert schema_convert.matches(table, strict=False)
    assert not schema_coerce.matches(table, strict=False)
    if isinstance(table, pa.Table):
        assert schema.matches(table.schema)


def test_coerce(table_and_schemas: Tuple[TableLike, Schema, Schema, Schema]):
    table, _, _, schema_coerce = table_and_schemas
    coerced = schema_coerce.coerce(table)
    assert schema_coerce.matches(coerced, strict=False)


def test_to_pyarrow(fields: Fields):
    schema = Schema(fields)
    pa_schema = schema.to_pyarrow()
    schema2 = Schema.from_pyarrow(schema=pa_schema)
    assert schema.matches(schema2), "pyarrow roundtrip mismatch"


@pytest.mark.parametrize("_field_type", ["prim", "pd"])
@pytest.mark.parametrize("metadata", [{}, {"a": "some data"}])
def test_schema_str_repr(_field_type: str, metadata: Optional[Dict[str, Any]]):
    fields = _fields(_field_type)
    schema = Schema(fields=fields, metadata=metadata)
    print(str(schema))
    print(repr(schema))


if __name__ == "__main__":
    pytest.main([__file__])
