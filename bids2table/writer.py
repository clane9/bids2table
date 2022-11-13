import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from bids2table._utils import parse_size
from bids2table.schema import Schema

TableBatch = Union[pd.DataFrame, pa.Table]


class ParquetWriter:
    """
    Write a stream of pandas ``DataFrame`` or `PyArrow`_ ``Table`` batches to a parquet
    file using `PyArrow`_

    Example::

        with ParquetWriter("table") as writer:
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
        buffer_size: Union[str, int] = "16 MiB",
        partition_size: Union[str, int] = "64 MiB",
    ):
        self.prefix = prefix
        self.coerce = coerce
        if isinstance(buffer_size, str):
            buffer_size = parse_size(buffer_size)
        if isinstance(partition_size, str):
            partition_size = parse_size(partition_size)
        self.buffer_size = buffer_size
        self.partition_size = partition_size

        self._schema: Optional[Schema] = None
        self._table: Optional[pa.Table] = None
        self._stream: Optional[pq.ParquetWriter] = None
        self._part = 0
        self._batch_count = 0
        self._bytes_written = 0

    def write(self, batch: TableBatch):
        """
        Write a batch to the stream. If this is the first batch, the schema and stream
        will be initialized.
        """
        if self._schema is None:
            self._init_schema(batch)
        if self._stream is None:
            self._init_stream()
        batch = self._to_pyarrow(batch)
        if self._table is None:
            self._table = batch
        else:
            self._table = pa.concat_tables([self._table, batch])
        if self._table.get_total_buffer_size() > self.buffer_size:
            self._flush()
        if self._bytes_written > self.partition_size:
            self.close()
        self._batch_count += 1

    def _init_schema(self, first_batch: TableBatch):
        """
        Initialize the output schema from the first batch.
        """
        if isinstance(first_batch, pd.DataFrame):
            self._schema = Schema.from_pandas(first_batch)
        else:
            self._schema = Schema.from_pyarrow(table=first_batch)

    def _init_stream(self):
        # TODO: expose kwargs as config options
        self._stream = pq.ParquetWriter(
            self._temp_path(), schema=self._schema.to_pyarrow()
        )

    def _path(self) -> str:
        return f"{self.prefix}-{self._part:04d}.parquet"

    def _temp_path(self) -> str:
        path = Path(self.prefix)
        temp_path = str(path.parent / f".tmp-{path.name}-{self._part:04d}.parquet")
        return temp_path

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

    def _flush(self):
        """
        Flush the internal buffer.
        """
        if self._table is not None:
            self._bytes_written += self._table.get_total_buffer_size()
            logging.info(
                "Flushing to partition\n"
                f"\tpath: {self._path()}\n"
                f"\tbatches: {self._batch_count}\n"
                f"\tMB: {self._bytes_written / 1e6}"
            )
            self._stream.write_table(
                self._table, row_group_size=self._table.get_total_buffer_size()
            )
            self._table = None

    def close(self):
        """
        Close the stream.
        """
        if self._stream is not None:
            self._flush()
            logging.info(
                "Closing partition\n"
                f"\tpath: {self._path()}\n"
                f"\tbatches: {self._batch_count}\n"
                f"\tMB: {self._bytes_written / 1e6}"
            )
            self._stream.close()
            self._stream = None
            os.rename(self._temp_path(), self._path())
            self._bytes_written = 0
            self._part += 1

    def __enter__(self) -> "ParquetWriter":
        return self

    def __exit__(self):
        self.close()
