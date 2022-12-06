import json
import logging
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from bids2table import StrOrPath
from bids2table.crawler import CrawlCounts, HandlingFailure

__all__ = ["ProcessedLog", "setup_logging", "format_task_id"]


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
        "run_id": str,
        "task_id": int,
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
        df = df.drop_duplicates(subset="dir", keep="last")
        return df

    @classmethod
    def _empty(cls) -> pd.DataFrame:
        df = pd.DataFrame(columns=list(cls.FIELDS.keys()))
        df = df.astype(cls.FIELDS)
        return df

    def write(
        self,
        run_id: str,
        task_id: int,
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
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "task_id": task_id,
            "path": path,
            "counts": counts.to_dict(),
            "error_rate": counts.error_rate,
            "partitions": partitions,
            "errors": [err.to_dict() for err in errors],
        }
        log_dir = self.db_dir / self.PREFIX / run_id
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        json_path = log_dir / (format_task_id(task_id) + ".proc.json")
        with open(json_path, "a") as f:
            print(json.dumps(record), file=f)

    def filter_paths(
        self, paths: np.ndarray, error_rate_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Filter a list of paths against those we've already processed successfully.
        """
        # Pick out the processed paths that are also in the current todo list.
        overlap_mask = np.isin(self.df["path"], paths)
        if overlap_mask.sum() == 0:
            return paths
        maybe_redo = self.df.loc[overlap_mask, :]

        # Find partitions that were successfully written out.
        exists_mask = maybe_redo["partitions"].apply(_all_exist)

        # And the paths that were processed without too many errors.
        success_mask = exists_mask
        if error_rate_threshold is not None:
            success_mask = success_mask & (
                maybe_redo["error_rate"] <= error_rate_threshold
            )

        # Filter out paths that were already completed successfully.
        success_paths = maybe_redo.loc[success_mask, "path"].values
        paths = np.setdiff1d(paths, success_paths)
        return paths


def _all_exist(ps: List[str]) -> bool:
    return all(_exists(p) for p in ps)


@lru_cache(maxsize=2**14)
def _exists(p: str) -> bool:
    return Path(p).exists()


def setup_logging(
    task_id: int,
    log_dir: Optional[StrOrPath] = None,
    level: Union[int, str] = "INFO",
):
    """
    Setup root logger.
    """
    task_id_str = format_task_id(task_id)
    FORMAT = (
        f"({task_id_str}) [%(levelname)s %(asctime)s %(filename)s:%(lineno)4d]: "
        "%(message)s"
    )
    formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

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
        log_path = log_dir / f"log-{task_id_str}.txt"
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Redefining the root logger is not strictly best practice.
    # https://stackoverflow.com/a/7430495
    # But I want the convenience to just call e.g. `logging.info()`.
    logging.root = logger  # type: ignore


def format_task_id(task_id: int) -> str:
    """
    Format a task ID as a zero-padded string.
    """
    return f"{task_id:05d}"
