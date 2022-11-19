from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pytest import FixtureRequest

from bids2table import RecordDict
from bids2table.schema import Fields, Schema

TableType = Union[List[RecordDict], pd.DataFrame, pa.Table]


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


@pytest.fixture(params=["records", "pandas", "pyarrow"])
def table_type(request: FixtureRequest) -> str:
    return request.param


def _table_and_schemas(_table_type: str) -> Tuple[TableType, Schema, Schema]:
    schema = Schema({"a": "object", "b": "int", "c": "float", "d": "object"})
    schema_convert = Schema(
        {"a": "string", "b": "Int64", "c": "Float64", "d": "object"}
    )
    records = [
        {"a": "abc", "b": 123, "c": 9.99, "d": np.zeros(3)},
        {"a": "def", "b": 456, "c": 1.11, "d": np.ones(3)},
    ]
    if _table_type == "records":
        table = records
    elif _table_type == "pandas":
        table = pd.DataFrame.from_records(records)
    elif _table_type == "pyarrow":
        table = pa.Table.from_pylist(records)
    else:
        raise NotImplementedError(f"table_type '{_table_type}' not implemented")
    return table, schema, schema_convert


@pytest.fixture
def table_and_schemas(table_type: str) -> Tuple[TableType, Schema, Schema]:
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
    table_and_schemas: Tuple[TableType, Schema, Schema], convert: bool
):
    table, schema, schema_convert = table_and_schemas
    schema = schema_convert if convert else schema
    if isinstance(table, list):
        schema2 = Schema.from_records(table, convert=convert)
        schema3 = Schema.from_records(table[0], convert=convert)
    elif isinstance(table, pd.DataFrame):
        schema2 = Schema.from_pandas(table, convert=convert)
        schema3 = None
    elif isinstance(table, pa.Table):
        schema2 = Schema.from_pyarrow(table=table, convert=convert)
        schema3 = Schema.from_pyarrow(schema=table.schema, convert=convert)
    else:
        raise TypeError(f"table has unexpected type {type(table)}")
    assert schema.matches(schema2)
    assert schema3 is None or schema.matches(schema3)


def test_schema_matches(table_and_schemas: Tuple[TableType, Schema, Schema]):
    table, schema, _ = table_and_schemas
    assert schema.matches(table)
    if isinstance(table, pa.Table):
        assert schema.matches(table.schema)


def test_coerce():
    # NOTE: extension dtypes are required to handle missing data
    schema = Schema({"a": "string", "b": "Int32", "c": "Float32", "d": "object"})
    df = pd.DataFrame.from_records([{"d": np.ones(3), "c": 1, "a": None, "e": None}])
    expected_df = pd.DataFrame.from_records(
        [{"a": None, "b": None, "c": 1.0, "d": np.ones(3)}]
    ).astype(schema.fields)
    coerced_df = schema.coerce(df)
    assert coerced_df.equals(expected_df)
    assert schema.matches(coerced_df)


@pytest.mark.parametrize("_field_type", ["prim", "pd"])
@pytest.mark.parametrize("metadata", [{}, {"a": "some data"}])
def test_schema_str_repr(_field_type: str, metadata: Optional[Dict[str, Any]]):
    fields = _fields(_field_type)
    schema = Schema(fields=fields, metadata=metadata)
    print(str(schema))
    print(repr(schema))


if __name__ == "__main__":
    pytest.main([__file__])
