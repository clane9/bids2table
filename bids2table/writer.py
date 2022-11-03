import logging
from typing import Optional, Union

import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from bids2table.schema import Schema

TableBatch = Union[pd.DataFrame, pa.Table]


class ParquetWriter:
    """
    Write a stream of pandas ``DataFrame`` or `PyArrow`_ ``Table`` batches to a parquet
    file using `PyArrow`_

    Example::

        with ParquetWriter("table.parquet") as writer:
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
        path: str,
        coerce: bool = False,
        buffer_size: int = 64 * 1024 * 1024,
    ):
        self.path = path
        self.coerce = coerce
        self.buffer_size = buffer_size
        self.schema: Optional[Schema] = None
        self._table: Optional[pa.Table] = None
        self._stream: Optional[pq.ParquetWriter] = None
        # TODO: do something with these values
        self._batch_count = 0
        self._total_bytes_written = 0

    def _init_stream(self, first_batch: TableBatch):
        """
        Initialize the stream, using the first batch to set the schema.
        """
        if self._stream is not None:
            raise RuntimeError("parquet stream is already initialized")
        if isinstance(first_batch, pd.DataFrame):
            self.schema = Schema.from_pandas(first_batch)
        else:
            self.schema = Schema.from_pyarrow(table=first_batch)
        # TODO: expose kwargs as config options
        self._stream = pq.ParquetWriter(self.path, self.schema.to_pyarrow())

    def close(self):
        """
        Close the stream.
        """
        if self._stream is not None:
            self._stream.close()

    def write(self, batch: TableBatch):
        """
        Write a batch to the stream. If this is the first batch, the stream is
        initialized.
        """
        self._batch_count += 1
        if self._stream is None:
            self._init_stream(batch)
        batch = self._to_pyarrow(batch)
        if self._table is None:
            self._table = batch
        else:
            self._table = pa.concat_tables([self._table, batch])
        if self._table.get_total_buffer_size() > self.buffer_size:
            self.flush()

    def _to_pyarrow(self, batch: TableBatch) -> pa.Table:
        """
        Convert a table batch to a PyArrow ``Table`` (if necessary).
        """
        assert self.schema is not None, "writer schema not initialized"
        if isinstance(batch, pd.DataFrame):
            if not self.schema.matches(batch):
                logging.warning(
                    "Parquet writer batch %d has a different schema\n\tpath: %s",
                    self._batch_count,
                    self.path,
                )
                if self.coerce:
                    batch = self.schema.coerce(batch)
            batch = pa.Table.from_pandas(batch)
            batch = batch.replace_schema_metadata(self.schema.metadata)
        return batch

    def flush(self):
        """
        Flush the internal buffer.
        """
        if self._table is not None:
            self._total_bytes_written += self._table.get_total_buffer_size()
            self._stream.write_table(self._table)
            self._table = None

    def __enter__(self) -> "ParquetWriter":
        return self

    def __exit__(self):
        self.flush()
        self.close()
