import csv
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from pytest import FixtureRequest

from bids2table.loaders import text


@pytest.fixture(params=["flat", "nested"])
def record_type(request: FixtureRequest) -> str:
    return request.param


@pytest.fixture
def record(record_type: str) -> Dict[str, Any]:
    base_record = {"a": 1, "b": "abc", "c": 99.99}
    if record_type == "flat":
        record = base_record
    elif record_type == "nested":
        record = base_record
        record["d"] = [1, 2, 3]
        record["e"] = {"A": 101, "B": 102}
    else:
        raise NotImplementedError(f"record_type '{record_type}' not implemented")
    return record


@pytest.fixture
def single_row_df(record: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame.from_records([record])
    return df


@pytest.fixture
def single_row_tsv(tmp_path: Path, single_row_df: pd.DataFrame) -> Path:
    path = tmp_path / "table.tsv"
    single_row_df.to_csv(
        path, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\"
    )
    return path


@pytest.fixture
def json_dict(tmp_path: Path, record: Dict[str, Any]) -> Path:
    path = tmp_path / "record.json"
    with open(path, "w") as f:
        json.dump(record, f)
    return path


@pytest.fixture(params=[(10, 12), (10, 1), (1, 10), (10,)])
def array_shape(request: FixtureRequest) -> Tuple[int, int]:
    return request.param


@pytest.fixture
def random_array(array_shape: Tuple[int, int]) -> np.ndarray:
    return np.random.randn(*array_shape)


@pytest.fixture
def array_tsv(tmp_path: Path, random_array: np.ndarray) -> Path:
    path = tmp_path / "array.tsv"
    np.savetxt(path, random_array, delimiter="\t")
    return path


def test_load_single_row_tsv(single_row_tsv: Path, record: Dict[str, Any]):
    with open(single_row_tsv) as f:
        print(repr(f.read()))
    loaded_record = text.load_single_row_tsv(single_row_tsv)
    assert loaded_record == record


def test_load_json_dict(json_dict: Path, record: Dict[str, Any]):
    loaded_record = text.load_json_dict(json_dict)
    assert loaded_record == record


def test_load_json_dict_no_nesting(json_dict: Path, record: Dict[str, Any]):
    record = {k: v for k, v in record.items() if not isinstance(v, (list, dict, tuple))}
    loaded_record = text.load_json_dict(json_dict, nested=False)
    assert loaded_record == record


def test_load_array_tsv(array_tsv: Path, random_array: np.ndarray):
    loaded_record = text.load_array_tsv(array_tsv, name="array")
    loaded_array = loaded_record["array"]
    random_array = np.squeeze(random_array)
    assert loaded_array.shape == random_array.shape
    assert np.all(loaded_array == random_array)


if __name__ == "__main__":
    pytest.main()
