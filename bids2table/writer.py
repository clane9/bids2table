import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from bids2table.schema import Schema
from bids2table.utils import atomicopen, parse_size

TableBatch = Union[pd.DataFrame, pa.Table]


class BufferedParquetWriter:
    """
    Write a stream of pandas ``DataFrame`` or `PyArrow`_ ``Table`` batches to a parquet
    file using `PyArrow`_

    Example::

        with BufferedParquetWriter("table") as writer:
            for batch in batches:
                writer.write(batch)

    .. note::
        The schema for the output file is initialized based on the first batch. This
        can have consequences if the batches have different schemas, or even if the
        first batch is just missing data for some columns.

        By default, PyArrow will raise an error in this case. Setting ``coerce = True``
        quiets this by coercing all batches to the schema of the first batch. In the
        first-batch-missing-data case, the corresponding columns will have the
        ``object`` dtype.

    TODO: would this work for paths on s3?

    .. _PyArrow: https://arrow.apache.org/docs/python
    """

    def __init__(
        self,
        prefix: str,
        coerce: bool = False,
        partition_size: Union[str, int] = "64 MiB",
    ):
        self.prefix = prefix
        self.coerce = coerce
        self.partition_size = partition_size

        if isinstance(partition_size, str):
            self._partition_size_bytes = parse_size(partition_size)
        else:
            self._partition_size_bytes = partition_size

        self._schema: Optional[Schema] = None
        self._table: Optional[pa.Table] = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._part = 0
        self._part_batch_start = 0
        self._batch_count = 0

    def write(self, batch: TableBatch) -> str:
        """
        Write a batch to the stream. If this is the first batch, the schema and stream
        will be initialized. Returns the path to the partition containing this batch.
        """
        if self._schema is None:
            self._init_schema(batch)
        batch = self._to_pyarrow(batch)
        if self._table is None:
            self._table = batch
        else:
            self._table = pa.concat_tables([self._table, batch])
        self._batch_count += 1

        partition = self.path()
        if self._table.get_total_buffer_size() > self._partition_size_bytes:
            self.flush()
        return partition

    def _init_schema(self, first_batch: TableBatch):
        """
        Initialize the output schema from the first batch.
        """
        if isinstance(first_batch, pd.DataFrame):
            self._schema = Schema.from_pandas(first_batch)
        else:
            self._schema = Schema.from_pyarrow(table=first_batch)

    def path(self) -> str:
        """
        Return the path for the current partition.
        """
        prefix = Path(self.prefix)
        if prefix.is_dir():
            path = str(prefix / f"{self._part:04d}.parquet")
        else:
            path = f"{self.prefix}-{self._part:04d}.parquet"
        return path

    def _to_pyarrow(self, batch: TableBatch) -> pa.Table:
        """
        Convert a table batch to a PyArrow ``Table`` (if necessary).
        """
        if isinstance(batch, pd.DataFrame):
            if not self._schema.matches(batch):
                logging.warning(
                    "Parquet writer batch %d has a different schema\n\tprefix: %s",
                    self._batch_count,
                    self.prefix,
                )
                if self.coerce:
                    batch = self._schema.coerce(batch)
            batch = pa.Table.from_pandas(batch)
            batch = batch.replace_schema_metadata(self._schema.metadata)
        return batch

    def flush(self, blocking: bool = False):
        """
        Flush the table buffer.
        """
        if self._table is not None:
            parent = Path(self.path()).parent
            parent.mkdir(parents=True, exist_ok=True)

            if self._future is not None and self._future.running():
                logging.info("Waiting for previous partition to finish writing")
                self._future.result()

            table_bytes = self._table.get_total_buffer_size()
            logging.info(
                "Flushing to partition\n"
                f"\tbatches: [{self._part_batch_start}, {self._batch_count}]\n"
                f"\tMB: {table_bytes / 1e6}\n"
                f"\tpath: {self.path()}"
            )
            self._future = self._pool.submit(
                self._flush_task,
                self._table,
                self.path(),
                2 * self._partition_size_bytes,
            )

            self._table = None
            self._part += 1
            self._part_batch_start = self._batch_count

            if blocking:
                self._future.result()

    @staticmethod
    def _flush_task(table: pa.Table, path: str, row_group_size: int):
        with atomicopen(path, "wb") as f:
            pq.write_table(table, f, row_group_size=row_group_size)

    def __enter__(self) -> "BufferedParquetWriter":
        return self

    def __exit__(self):
        self.flush()
