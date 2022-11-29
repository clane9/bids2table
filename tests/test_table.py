import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from bids2table.table import IncrementalTable


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
    schema = {"A": schema_A, "B": schema_B}

    constants = {"Y": 2022}

    table = IncrementalTable(
        index_schema=index_schema, schema=schema, constants=constants
    )

    rng = np.random.default_rng(2022)

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
        ks = list(record.keys())
        for k in ks:
            if rng.random() < missing_prob:
                record.pop(k)
        return key, record, "A"

    def random_record_B(missing_prob: float = 0.0):
        key = random_key()
        record = {
            "c": rng.random(),
            "d": rng.normal(size=(rng.integers(10, 20))),
            "e": {"aa": "aabbcc"},
        }
        ks = list(record.keys())
        for k in ks:
            if rng.random() < missing_prob:
                record.pop(k)
        return key, record, "B"

    for _ in range(20):
        for fun in [random_record_A, random_record_B]:
            key, record, group = fun(missing_prob=0.2)
            table.put(key, record, group)

    table_pa = table.as_table()
    table_pd: pd.DataFrame = table_pa.to_pandas()
    logging.info("Table:\n%s", table_pd)

    # check expected shape and columns
    expected_columns = np.array(
        ["_index__ii", "_index__jj", "Y", "A__a", "A__b", "B__c", "B__d", "B__e"],
    )
    assert table_pd.shape == (12, 8)
    assert np.all(table_pd.columns.values == expected_columns)

    # check that the rows are sorted by index
    index_columns = [c for c in table_pd.columns if c.startswith("_index")]
    index = table_pd.loc[:, index_columns]
    assert index.equals(index.sort_values(by=index_columns))

    # check invalid group
    with pytest.raises(ValueError):
        table.put(random_key(), {}, "C")

    # check extra columns
    with pytest.raises(ValueError):
        table.put(random_key(), {"c": 1.0}, "A")


if __name__ == "__main__":
    pytest.main([__file__])
