from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from bids2table.handlers import HandlerConfig, WrapHandlerConfig
from bids2table.indexers import BIDSIndexerConfig, IndexerConfig

__all__ = [
    "PathsConfig",
    "TableConfig",
    "Config",
]


@dataclass
class PathsConfig:
    # List of session directories to process. Can be one of the following:
    #   - A list of directory paths and/or glob patterns.
    #   - A single path pointing to '*.txt' or '*.npy' file containing an
    #     array of directory paths/glob patterns.
    list: Optional[List[str]] = None
    list_path: Optional[str] = None

    # If true, the pattern '**' will match any files and zero or more directories and
    # subdirectories.
    glob_recursive: bool = True

    # Exclude paths that have alreadly been successfully processed.
    filter_completed: bool = True

    # Paths that were processed previously but whose error rates exceed this threshold
    # will be re-processed. Set to null to disable.
    redo_error_rate_threshold: float = 0.25

    # Minimum number of paths to assign to each worker.
    min_per_worker: Optional[int] = None


@dataclass
class TableConfig:
    # Table name. Defaults to the table key in cfg.tables.
    name: Optional[str] = None

    # Indexer config
    indexer: IndexerConfig = MISSING

    # Handler configs, mapping arbitrary handler key to a handler config. The key is
    # only used to enable composition.
    handlers: Dict[str, HandlerConfig] = MISSING


@dataclass
class Config:
    # Top-level database directory. Table partitions are organized as:
    #   {db_dir} / {table_name} / {collection_id} / {worker_id} / {part}.parquet
    db_dir: str = MISSING

    # Logging directory, shared across runs. Individual run logs go in a subdirectory
    # according to collection_id.
    log_dir: str = MISSING

    # Unique ID for the current bids2table collection.
    collection_id: str = MISSING

    # Session directories are assigned to workers to process based on their ID.
    # TODO: It might be worth inferring these from slurm. But then that might be a bit
    # too much hiding stuff from the user.
    worker_id: int = 0

    # Total number of workers
    num_workers: int = 1

    # List of session directories to process
    paths: PathsConfig = PathsConfig()

    # Mapping of table names to table configs
    tables: Dict[str, TableConfig] = MISSING

    # Crawler kwargs
    crawler: Optional[Dict[str, Any]] = None

    # BufferedParquetWriter kwargs
    writer: Optional[Dict[str, Any]] = None

    # List of python modules to import for e.g. Handler and Indexer definitions.
    # TODO:
    #   - still needed?
    #   - what about example path?
    include_modules: Optional[List[str]] = None

    # Whether to flush table buffers when exiting on error
    flush_on_error: bool = False

    # Configure the processing run but don't run anything
    dry_run: bool = False

    # Logging level
    log_level: str = "INFO"
    # Log crawler progress after every `log_frequency` directories
    log_frequency: int = 10


config_store = ConfigStore.instance()
config_store.store(name="base", node=Config)

config_store.store(
    name="base_bids_indexer", node=BIDSIndexerConfig, group="tables/indexer"
)
config_store.store(
    name="base_wrap_handler", node=WrapHandlerConfig, group="tables/handlers"
)
