import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from bids2table.extensions import PaNDArrayType, PaPickleType
from bids2table.table import IncrementalTable
from bids2table.types import RecordDict


# some custom data for the pickle type
@dataclass
class Point:
    x: float
    y: float
    z: float

    @property
    def length(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)


def test_incremental_table():
    """
    Complete test of ``IncrementalTable``

        - incremental insertion
            - with missing fields
            - with overwrites
        - pyarrow conversion
        - pandas conversion
        - error handling

    NOTE: writing this as a single test partly out of laziness, but also bc I want an
    end-to-end test of the full functionality.
    """
    index_schema = pa.schema(
        {"ii": pa.string(), "jj": pa.int64()},
        metadata={"ii": "index string"},
    )

    schema_A = pa.schema(
        {"a": pa.string(), "b": pa.int64()},
        metadata={"a": "some string"},
    )
    schema_B = pa.schema(
        {
            "c": pa.float32(),
            "d": pa.list_(pa.float64()),
            "e": pa.struct({"aa": pa.string()}),
        },
        metadata={"e": "some struct"},
    )
    # schema with extension types,
    schema_C = pa.schema(
        {
            "f": PaPickleType(),
            "g": PaNDArrayType(pa.float32()),
        },
        metadata={"f": "pickled object"},
    )
    schema = {"A": schema_A, "B": schema_B, "C": schema_C}

    constants = {"Y": 2022}

    table = IncrementalTable(
        index_schema=index_schema, schema=schema, constants=constants
    )

    rng = np.random.default_rng(2022)

    def drop_keys(record: RecordDict, missing_prob: float = 0.0):
        ks = list(record.keys())
        for k in ks:
            if rng.random() < missing_prob:
                record.pop(k)
        return record

    def random_key():
        key = {
            "ii": rng.choice(["01", "02", "03"]),
            "jj": rng.integers(0, 4),
        }
        return key

    def random_record_A(missing_prob: float = 0.0):
        key = random_key()
        record = {
            "a": rng.choice(["abc", "def", "ghi"]),
            "b": rng.integers(0, 10),
        }
        drop_keys(record, missing_prob)
        return key, record, "A"

    def random_record_B(missing_prob: float = 0.0):
        key = random_key()
        record = {
            "c": rng.random(),
            "d": rng.normal(size=(rng.integers(10, 20))),
            "e": {"aa": "aabbcc"},
        }
        drop_keys(record, missing_prob)
        return key, record, "B"

    def random_record_C(missing_prob: float = 0.0):
        key = random_key()
        record = {
            "f": Point(*rng.random(size=(3,))),
            # Note the type here is float64 but we asked for float32. Type casting will
            # happen automatically.
            "g": rng.normal(size=(5, 4)),
        }
        drop_keys(record, missing_prob)
        return key, record, "C"

    for _ in range(20):
        for fun in [random_record_A, random_record_B, random_record_C]:
            key, record, group = fun(missing_prob=0.2)
            table.put(key, record, group)

    table_pa = table.as_table()
    table_pd: pd.DataFrame = table_pa.to_pandas()
    logging.info("Table:\n%s", table_pd)

    # check expected shape and columns
    expected_columns = np.array(
        "_index__ii _index__jj Y A__a A__b B__c B__d B__e C__f C__g".split()
    )
    assert table_pd.shape == (12, 10)
    assert np.all(table_pd.columns.values == expected_columns)

    # check that the rows are sorted by index
    index_columns = [c for c in table_pd.columns if c.startswith("_index")]
    index = table_pd.loc[:, index_columns]
    assert index.equals(index.sort_values(by=index_columns))

    # check invalid group
    with pytest.raises(ValueError):
        table.put(random_key(), {}, "D")

    # check extra columns
    with pytest.raises(ValueError):
        table.put(random_key(), {"c": 1.0}, "A")


if __name__ == "__main__":
    pytest.main([__file__])
