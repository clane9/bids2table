import json
import logging
import sys
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from bids2table.crawler import CrawlCounts, HandlingFailure
from bids2table.types import StrOrPath

__all__ = ["ProcessedLog", "setup_logging", "format_worker_id"]


class ProcessedLog:
    """
    A log of processed directories. Supports reading, writing, and filtering candidate
    paths.

    Args:
        db_dir: path to database directory
    """

    PREFIX = "_proc"
    FIELDS = {
        "timestamp": np.datetime64,
        "collection_id": str,
        "worker_id": int,
        "path": str,
        "counts": object,
        "error_rate": float,
        "partitions": object,
        "errors": object,
    }

    def __init__(self, db_dir: StrOrPath):
        self.db_dir = Path(db_dir)
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        """
        Processed log dataframe.
        """
        if self._df is None:
            self._df = self._load(self.db_dir)
        return self._df

    def _load(self, db_dir: StrOrPath) -> pd.DataFrame:
        """
        Load the log of processed directories from the database directory ``db_dir``.
        """
        log_path = Path(db_dir) / self.PREFIX
        if not log_path.exists():
            return self._empty()

        logging.info("Loading processed log JSONs")
        batches = []
        for path in sorted(log_path.glob("**/*.proc.json")):
            batches.append(pd.read_json(path, lines=True))
        if len(batches) == 0:
            return self._empty()

        logging.info("De-duplicating processed log (keeping last)")
        df = pd.concat(batches, ignore_index=True)
        df = df.astype(self.FIELDS)
        df = df.drop_duplicates(subset="path", keep="last")
        return df

    @classmethod
    def _empty(cls) -> pd.DataFrame:
        df = pd.DataFrame(columns=list(cls.FIELDS.keys()))
        df = df.astype(cls.FIELDS)
        return df

    def write(
        self,
        collection_id: str,
        worker_id: int,
        path: StrOrPath,
        counts: CrawlCounts,
        partitions: List[str],
        errors: List[HandlingFailure],
    ):
        """
        Append a record to the processed log table.
        """
        path = str(Path(path).absolute())
        partitions = [str(Path(part).relative_to(self.db_dir)) for part in partitions]
        # TODO: possible fields to add:
        #   - start/end/elapsed
        #   - table shapes
        #   - list of handlers applied
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "collection_id": collection_id,
            "worker_id": worker_id,
            "path": path,
            "counts": counts.to_dict(),
            "error_rate": counts.error_rate,
            "partitions": partitions,
            "errors": [err.to_dict() for err in errors],
        }
        log_dir = self.db_dir / self.PREFIX / collection_id
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        json_path = log_dir / (format_worker_id(worker_id) + ".proc.json")
        with open(json_path, "a") as f:
            print(json.dumps(record), file=f)

    def filter_paths(
        self, paths: np.ndarray, error_rate_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Filter a list of paths against those we've already processed successfully.
        """
        # Join current todo list with previously processed paths.
        todo_df = pd.DataFrame(
            {"path": paths, "indices": np.arange(len(paths))}, index=paths
        )
        log_df = self.df.set_index("path")
        todo_df = todo_df.join(log_df, how="left")

        # If all paths are new, nothing to filter.
        new_mask = todo_df["collection_id"].isna().values
        if new_mask.sum() == len(paths):
            return paths

        # Initialize the redo mask to be the full overlap. We'll filter out successfully
        # processed paths.
        redo_mask = ~new_mask
        maybe_redo_df = todo_df.loc[redo_mask, :]

        # Find partitions that were successfully written out.
        success_mask = (
            maybe_redo_df["partitions"]
            .apply(partial(_all_exist, root=self.db_dir))
            .values
        )

        # And the paths that were processed without too many errors.
        if error_rate_threshold is not None:
            success_mask = success_mask & (
                maybe_redo_df["error_rate"].values <= error_rate_threshold
            )

        # Only redo paths that previously failed.
        redo_mask[redo_mask] = ~success_mask

        # Construct final todo paths. Use indices to guarantee original order.
        todo_mask = new_mask | redo_mask
        todo_df = todo_df.loc[todo_mask, :]
        paths = paths[todo_df["indices"]]
        return paths


def _all_exist(ps: List[str], root: Optional[StrOrPath] = None) -> bool:
    return all(_exists(p, root=root) for p in ps)


@lru_cache(maxsize=2**14)
def _exists(p: str, root: Optional[StrOrPath] = None) -> bool:
    return (Path(root) / p).exists() if root else Path(p).exists()


def setup_logging(
    worker_id: int,
    log_dir: Optional[StrOrPath] = None,
    level: Union[int, str] = "INFO",
):
    """
    Setup root logger.
    """
    worker_id_str = format_worker_id(worker_id)
    fmt = (
        f"({worker_id_str}) [%(levelname)s %(asctime)s %(filename)s:%(lineno)4d]: "
        "%(message)s"
    )
    formatter = logging.Formatter(fmt, datefmt="%y-%m-%d %H:%M:%S")

    logger = logging.getLogger()
    logger.setLevel(level)
    # clean up any pre-existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_path = log_dir / f"log-{worker_id_str}.txt"
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Redefining the root logger is not strictly best practice.
    # https://stackoverflow.com/a/7430495
    # But I want the convenience to just call e.g. `logging.info()`.
    logging.root = logger  # type: ignore


def format_worker_id(worker_id: int) -> str:
    """
    Format a worker ID as a zero-padded string.
    """
    return f"{worker_id:04d}"
