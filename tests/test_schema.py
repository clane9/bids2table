import json
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pytest
import yaml
from pytest import FixtureRequest

from bids2table import RecordDict
from bids2table.schema import (
    DataType,
    cast_to_schema,
    concat_schemas,
    create_schema,
    format_schema,
    get_dtype,
    get_fields,
)


@pytest.fixture
def fields() -> Dict[str, Tuple[DataType, pa.DataType]]:
    return {
        "a": ("int", pa.int64()),
        "b": ("int16", pa.int16()),
        "c": ("str", pa.string()),
        "d": (np.float32, pa.float32()),
        "e": ("datetime64[ns]", pa.timestamp("ns")),
        "f": ("list<float32>", pa.list_(pa.float32())),
        "g": (
            "struct < A: int, B: str >",
            pa.struct({"A": pa.int64(), "B": pa.string()}),
        ),
    }


@pytest.fixture
def record() -> RecordDict:
    return {
        "b": 99,
        "a": 12,
        "c": "abc",
        "f": np.ones(10),
        "e": datetime.now(),
    }


@pytest.fixture
def metadata() -> Dict[str, str]:
    return {
        "a": "some int column",
        "c": "some str column",
        "details": json.dumps(
            {
                "year": 2022,
                "month": "November",
                "list": [0, 1, 2, 3],
            },
        ),
    }


@pytest.fixture(params=[object, "object", "map<int32, str>", "list", "struct"])
def unsupported_dtype(request: FixtureRequest) -> DataType:
    return request.param


def test_create_schema(
    fields: Dict[str, Tuple[DataType, pa.DataType]],
    metadata: Dict[str, str],
):
    input_fields = {name: f[0] for name, f in fields.items()}
    expected_fields = {name: f[1] for name, f in fields.items()}

    expected_schema = pa.schema(expected_fields, metadata=metadata)
    input_schema = create_schema(input_fields, metadata=metadata)
    assert input_schema.equals(expected_schema, check_metadata=True)


def test_unsupported_get_dtype(unsupported_dtype: DataType):
    with pytest.raises(ValueError):
        get_dtype(unsupported_dtype)


def test_get_fields(fields: Dict[str, Tuple[DataType, pa.DataType]]):
    expected_fields = {name: f[1] for name, f in fields.items()}
    schema = pa.schema(expected_fields)
    extracted_fields = get_fields(schema)
    assert expected_fields == extracted_fields
    assert list(expected_fields.keys()) == list(extracted_fields.keys())


def test_format_schema(
    fields: Dict[str, Tuple[DataType, pa.DataType]],
    metadata: Dict[str, str],
):
    fields_ = {name: f[1] for name, f in fields.items()}
    schema = pa.schema(fields_, metadata=metadata)

    schema_fmt = format_schema(schema)
    schema_cfg = yaml.safe_load(schema_fmt)
    schema2 = create_schema(schema_cfg["fields"], metadata=schema_cfg["metadata"])
    assert schema.equals(schema2)


def test_cast_to_schema(
    fields: Dict[str, Tuple[DataType, pa.DataType]],
    record: RecordDict,
):
    fields_ = {name: f[1] for name, f in fields.items()}
    schema = pa.schema(fields_)
    cast_record = cast_to_schema(record, schema, safe=True, with_null=True)
    cast_schema = pa.RecordBatch.from_pylist([cast_record]).schema
    assert cast_schema.names == schema.names


def test_concat_schemas(
    fields: Dict[str, Tuple[DataType, pa.DataType]],
    metadata: Dict[str, str],
):
    fields_ = {name: f[1] for name, f in fields.items()}
    schema = pa.schema(fields_, metadata=metadata)

    schemas = {"A": schema, "B": schema}
    expected_names = [f"{k}__{name}" for k, s in schemas.items() for name in s.names]
    expected_metadata_keys = [
        f"{k}__{j.decode()}".encode()
        for k, s in schemas.items()
        for j in (s.metadata or {})
    ]

    combined_schema = concat_schemas(schemas=schemas, sep="__")
    assert expected_names == combined_schema.names
    assert expected_metadata_keys == list(combined_schema.metadata.keys())


if __name__ == "__main__":
    pytest.main([__file__])
