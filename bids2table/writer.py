import logging
from typing import Optional

import pandas as pd
import pyarrow as pa

from ._utils import df_as_other, df_matches_other


class ParquetWriter:
    """
    Write a stream of pandas ``DataFrame`` batches to a parquet file using `PyArrow`_

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

    def __init__(self, path: str, coerce: bool = False):
        self.path = path
        self.coerce = coerce
        self.schema: Optional[pa.Schema] = None
        self._first_batch: Optional[pd.DataFrame] = None
        self._stream: Optional[pa.RecordBatchFileWriter] = None
        self._batch_count = 0

    def _init_stream(self, first_batch: pd.DataFrame):
        """
        Initialize the stream, using the first batch to set the schema.
        """
        if self._stream is not None:
            raise RuntimeError("parquet stream is already initialized")
        self.schema = pa.Schema.from_pandas(first_batch)
        self._first_batch = first_batch
        self._stream = pa.RecordBatchFileWriter(self.path, self.schema)

    def close(self):
        """
        Close the stream.
        """
        if self._stream is not None:
            self._stream.close()

    def write(self, batch: pd.DataFrame):
        """
        Write a batch to the stream. If this is the first batch, the stream is
        initialized.
        """
        self._batch_count += 1
        if self._stream is None:
            self._init_stream(batch)

        if not df_matches_other(batch, self._first_batch):
            logging.warning(
                "Parquet writer batch %d has a different schema\n\tpath: %s",
                self._batch_count,
                self.path,
            )
            if self.coerce:
                logging.info(
                    "Coercing batch %d to match the first batch's schema",
                    self._batch_count,
                )
                batch = df_as_other(batch, self._first_batch)

        batch = pa.Table.from_pandas(batch)
        self._stream.write_table(batch)

    def __enter__(self) -> "ParquetWriter":
        return self

    def __exit__(self):
        self.close()
