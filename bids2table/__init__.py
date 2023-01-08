"""
bids2table is a scalable data pipeline for transforming a structured directory of
scientific data files (e.g. a BIDS directory) into a Parquet database.
"""

__version__ = "0.1.0-dev"

from bids2table.engine import launch  # noqa
from bids2table.path import *  # noqa
from bids2table.types import *  # noqa
