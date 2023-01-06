from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pyarrow import parquet as pq

from bids2table.extensions import (
    PaExtensionArray,
    PaNDArrayArray,
    PaNDArrayType,
    PaPickleArray,
    PaPickleType,
    PaPyExtensionType,
    PdNDArrayArray,
    PdNDArrayDtype,
    PdPickleArray,
    PdPickleDtype,
)
from bids2table.extensions.ndarray import _array_of_arrays


@dataclass
class ExtData:
    name: str
    array: np.ndarray
    pa_ext_type: PaPyExtensionType
    pa_ext_array_cls: Type[PaExtensionArray]
    pd_ext_type: ExtensionDtype
    pd_ext_array_cls: Type[ExtensionArray]


@pytest.fixture(params=["pickle", "ndarray"])
def ext_data(request: pytest.FixtureRequest) -> ExtData:
    if request.param == "pickle":
        array = np.array(
            [{"a": [1, 2, 3]}, ["abc", "def"], "abcdef", 0, None], dtype=object
        )
        ext_data = ExtData(
            name=request.param,
            array=array,
            pa_ext_type=PaPickleType(),
            pa_ext_array_cls=PaPickleArray,
            pd_ext_type=PdPickleDtype(),
            pd_ext_array_cls=PdPickleArray,
        )
    elif request.param == "ndarray":
        array = _array_of_arrays([np.random.randn(5, 4) for ii in range(5)])
        ext_data = ExtData(
            name=request.param,
            array=array,
            pa_ext_type=PaNDArrayType(pa.float64()),
            pa_ext_array_cls=PaNDArrayArray,
            pd_ext_type=PdNDArrayDtype(),
            pd_ext_array_cls=PdNDArrayArray,
        )
    else:
        raise ValueError(f"name {request.param} not implemented")

    return ext_data


def test_pd_series(ext_data: ExtData):
    ser = pd.Series(ext_data.array, dtype=ext_data.pd_ext_type)
    ser2 = pd.Series(ext_data.array, dtype=ext_data.name)
    # check expected dtype
    assert ser.dtype == ext_data.pd_ext_type
    assert str(ser.dtype) == ext_data.name
    # check expected type of underlying array
    assert isinstance(ser.values, ext_data.pd_ext_array_cls)
    # check two dtype specs produce same result
    assert ser.equals(ser2)
    # check series to numpy comparison
    if ext_data.name != "ndarray":
        eqmask = ser == ext_data.array
        namask = ser.isna()
        assert eqmask[~namask].all()
        # TODO: series and array compare not equal at na values. why?
        assert not eqmask[namask].any()
    # check item access
    assert _equals(ser[0], ext_data.array[0])


def test_pa_array(ext_data: ExtData):
    data = [ext_data.pa_ext_type.pack(v) for v in ext_data.array]
    arr = pa.array(data, type=ext_data.pa_ext_type)
    storage = pa.array(data)
    arr2 = ext_data.pa_ext_array_cls.from_storage(ext_data.pa_ext_type, storage)
    arr3 = ext_data.pa_ext_array_cls.from_sequence(ext_data.array)

    # check all construction methods produce same result
    assert arr == arr2
    assert arr == arr3

    # NOTE: values must be packed *before* constructing array
    with pytest.raises((pa.ArrowTypeError, pa.ArrowInvalid)):
        pa.array(ext_data.array, type=ext_data.pa_ext_type)

    # check python scalar conversion (with implicit deserialization)
    assert _equals(arr[0].as_py(), ext_data.array[0])

    # check numpy conversion
    if ext_data.name != "ndarray":
        assert np.all(ext_data.array == arr.to_numpy())

    # check python conversion
    if ext_data.name != "ndarray":
        assert arr.to_pylist() == ext_data.array.tolist()


def test_pd_pa_array_conversion(ext_data: ExtData):
    ser = pd.Series(ext_data.array, dtype=ext_data.name)
    arr = pa.array(ser)
    arr2 = pa.Array.from_pandas(ser)
    assert arr == arr2

    ser2 = arr.to_pandas()
    assert ser.equals(ser2)

    ser3 = pd.Series(arr).astype(ext_data.name)
    assert ser.equals(ser3)


def test_pd_pa_df_conversion(ext_data: ExtData):
    ser = pd.Series(ext_data.array, dtype=ext_data.name)
    data = {"ind": np.arange(len(ser)), "x": np.ones(len(ser)), ext_data.name: ser}
    df = pd.DataFrame(data)

    # basic pandas -> arrow conversion
    tab = pa.Table.from_pandas(df)

    # constructing table from data dict
    # note that since the "obj" column is represented as a series, pickle serialization
    # happens implicitly inside __arrow_array__.
    schema = pa.schema(
        {"ind": pa.int64(), "x": pa.float64(), ext_data.name: ext_data.pa_ext_type}
    )
    tab2 = pa.Table.from_pydict(data, schema=schema)

    # constructing table from records list
    # this is how the table is constructed inside the Crawler
    # note that we have to serialize manually in this case
    data2 = df.to_dict(orient="records")
    for record in data2:
        record[ext_data.name] = ext_data.pa_ext_type.pack(record[ext_data.name])
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
    col = tab[ext_data.name]
    ser2 = col.to_pandas()
    assert not ser.equals(ser2)
    ser3 = col.chunk(0).to_pandas()
    assert ser.equals(ser3)


def test_to_parquet(ext_data: ExtData, tmp_path: Path):
    ser = pd.Series(ext_data.array, dtype=ext_data.name)
    data = {"ind": np.arange(len(ser)), "x": np.ones(len(ser)), ext_data.name: ser}
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


def _equals(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray):
        return np.all(a == b)
    elif isinstance(a, (pd.Series, pd.DataFrame)):
        return a.equals(b)
    else:
        return a == b


if __name__ == "__main__":
    pytest.main(["-x", __file__])
