import string
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pyarrow import parquet as pq

from bids2table.writer import BufferedParquetWriter


def _random_batch(rng: np.random.Generator, batch_size: int = 128) -> pa.Table:
    batch = pd.DataFrame(
        {
            "a": rng.integers(0, 10, batch_size),
            "b": rng.random(batch_size),
            "c": [_random_string(rng, 5) for _ in range(batch_size)],
            "d": [rng.normal(size=rng.integers(0, 100)) for _ in range(batch_size)],
        }
    )
    return pa.Table.from_pandas(batch)


def _random_string(rng: np.random.Generator, length: int):
    return "".join(rng.choice(list(string.ascii_letters), length))


def test_buffered_parquet_writer(tmp_path: Path):
    rng = np.random.default_rng(2022)
    table_path = str(tmp_path / "table")
    batch_size = 128
    num_batches = 100
    with BufferedParquetWriter(table_path, partition_size="1 MB") as writer:
        for _ in range(num_batches):
            batch = _random_batch(rng, batch_size=batch_size)
            writer.write(batch)
    table = pq.read_table(table_path)
    assert table.shape == (num_batches * batch_size, 4)


if __name__ == "__main__":
    pytest.main([__file__])
