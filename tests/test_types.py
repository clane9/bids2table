import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import parquet as pq

from bids2table.types import PaPickleArray, PaPickleType, PdPickleArray, PdPickleDtype


@pytest.fixture
def objarr() -> np.ndarray:
    objs = [{"a": [1, 2, 3]}, ["abc", "def"], "abcdef", 0, None]
    return np.array(objs, dtype=object)


def test_pd_pickle_series(objarr: np.ndarray):
    ser = pd.Series(objarr, dtype=PdPickleDtype())
    ser2 = pd.Series(objarr, dtype="pickle")
    # check expected dtype
    assert ser.dtype == PdPickleDtype()
    assert str(ser.dtype) == "pickle"
    # check expected type of underlying array
    assert isinstance(ser.values, PdPickleArray)
    # check two dtype specs produce same result
    assert ser.equals(ser2)
    # check series to numpy comparison
    # TODO: series and array compare not equal at na values. why?
    eqmask = ser == objarr
    namask = ser.isna()
    assert eqmask[~namask].all()
    assert not eqmask[namask].any()
    # check item access
    assert ser[0] == objarr[0]


def test_pa_pickle_array(objarr: np.ndarray):
    bufs = [pickle.dumps(obj) for obj in objarr]
    arr = pa.array(bufs, type=PaPickleType())
    storage = pa.array(bufs)
    arr2 = PaPickleArray.from_storage(PaPickleType(), storage)
    # check both construction methods produce same result
    assert arr == arr2

    # NOTE: values must be pickled *before* constructing array
    with pytest.raises(pa.ArrowTypeError):
        pa.array(objarr, type=PaPickleType())

    # check python scalar conversion (with implicit deserialization)
    assert arr[0].as_py() == objarr[0]

    # check numpy conversion
    objarr2 = arr.to_numpy()
    assert np.all(objarr == objarr2)

    # check python conversion
    assert arr.to_pylist() == objarr.tolist()


def test_pd_pa_pickle_array_conversion(objarr: np.ndarray):
    ser = pd.Series(objarr, dtype="pickle")
    arr = pa.array(ser)
    arr2 = pa.Array.from_pandas(ser)
    assert arr == arr2

    ser2 = arr.to_pandas()
    assert ser.equals(ser2)

    ser3 = pd.Series(arr).astype("pickle")
    assert ser.equals(ser3)


def test_pd_pa_pickle_df_conversion(objarr: np.ndarray):
    ser = pd.Series(objarr, dtype="pickle")
    data = {"ind": np.arange(len(ser)), "x": np.ones(len(ser)), "obj": ser}
    df = pd.DataFrame(data)

    # basic pandas -> arrow conversion
    tab = pa.Table.from_pandas(df)

    # constructing table from data dict
    # note that since the "obj" column is represented as a series, pickle serialization
    # happens implicitly inside __arrow_array__.
    schema = pa.schema({"ind": pa.int64(), "x": pa.float64(), "obj": PaPickleType()})
    tab2 = pa.Table.from_pydict(data, schema=schema)

    # constructing table from records list
    # this is how the table is constructed inside the Crawler
    # note that we have to serialize manually in this case
    data2 = df.to_dict(orient="records")
    for record in data2:
        record["obj"] = pickle.dumps(record["obj"])
    tab3 = pa.Table.from_pylist(data2, schema=schema)

    df2 = tab.to_pandas()
    df3 = tab2.to_pandas()
    df4 = tab3.to_pandas()
    assert df.equals(df2)
    assert df.equals(df3)
    assert df.equals(df4)

    # check conversion of a table column to pandas series
    # TODO: the table column (type ChunkedArray) does not convert to pandas correctly.
    # The __from_arrow__ method never gets called. Why? Note that converting the first
    # chunk works fine.
    col = tab["obj"]
    ser2 = col.to_pandas()
    assert not ser.equals(ser2)
    ser3 = col.chunk(0).to_pandas()
    assert ser.equals(ser3)


def test_pickle_parquet(objarr: np.ndarray, tmp_path: Path):
    ser = pd.Series(objarr, dtype="pickle")
    data = {"ind": np.arange(len(ser)), "x": np.ones(len(ser)), "obj": ser}
    df = pd.DataFrame(data)

    # check round trip to parquet within pandas
    df.to_parquet(tmp_path / "data_pd.parquet")
    df2 = pd.read_parquet(tmp_path / "data_pd.parquet")
    assert df.equals(df2)

    # check round trip to parquet starting from pyarrow
    tab = pa.Table.from_pandas(df)
    pq.write_table(tab, tmp_path / "data_pa.parquet")
    df3 = pd.read_parquet(tmp_path / "data_pa.parquet")
    assert df.equals(df3)


if __name__ == "__main__":
    pytest.main([__file__])
