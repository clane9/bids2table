# TODO: Check that loading parquet files in different libraries works as expected:
#
#   [x] Dask. Seems to work fine out of the box. This produces a pandas dataframe with
#       the converted extension type::
#
#           ddf.read_parquet("table.parquet", engine="pyarrow").compute()
#
#   [ ] DuckDB. Doesn't seem work right away. Extension type columns are unboxed to the
#       storage type. Conversion to pandas doesn't automatically transform the data::
#
#           ddb.from_parquet("table.parquet").df()
#           ddb.from_parquet("table.parquet").arrow().to_pandas()

from .base import *  # noqa
from .ndarray import *  # noqa
from .pickle import *  # noqa
