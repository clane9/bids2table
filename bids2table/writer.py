import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa
from pyarrow import parquet as pq

from bids2table.types import StrOrPath
from bids2table.utils import atomicopen, parse_size


class BufferedParquetWriter:
    """
    Write a stream of `PyArrow`_ ``Table`` batches to a parquet directory.

    Batches are first buffered and then written out to partitions of approximately a
    fixed size. Writes are designed to be all-or-nothing, so there should be no
    remaining partial files in case of interruption.

    All batches must share the same schema, except for possibly the metadata. The
    metadata are inherited from the first batch in each partition.

    Example::

        with BufferedParquetWriter("table") as writer:
            for batch in batches:
                writer.write(batch)


    TODO: would this work for paths on s3?

    .. _PyArrow: https://arrow.apache.org/docs/python
    """

    def __init__(
        self,
        prefix: StrOrPath,
        partition_size: Union[str, int] = "64 MiB",
    ):
        self.prefix = Path(prefix)
        self.partition_size = partition_size

        if isinstance(partition_size, str):
            self._partition_size_bytes = parse_size(partition_size)
        else:
            self._partition_size_bytes = partition_size

        self._table: Optional[pa.Table] = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._part = 0
        self._part_batch_start = 0
        self._batch_count = 0

    def write(self, batch: pa.Table) -> str:
        """
        Write a batch to the stream. If this is the first batch, the schema and stream
        will be initialized. Returns the path to the partition containing this batch.
        """
        if self._table is None:
            self._table = batch
        else:
            self._table = pa.concat_tables([self._table, batch], promote=False)
        self._batch_count += 1
        partition = self.path()
        if self._table.get_total_buffer_size() > self._partition_size_bytes:
            self.flush(blocking=False)
        return partition

    def path(self) -> str:
        """
        Return the path for the current partition.
        """
        return str(self.prefix / f"{self._part:04d}.parquet")

    def flush(self, blocking: bool = True):
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
                f"\tbatches: [{self._part_batch_start}, {self._batch_count})\n"
                f"\tMB: {table_bytes / 1e6:.2f}\n"
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

    def __exit__(self, *args):
        self.flush()
        logging.info("Writer shutting down")
