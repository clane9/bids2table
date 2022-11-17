import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from bids2table import StrOrPath
from bids2table.engine import _format_task_id
from bids2table.schema import Schema


class ProcessedLog:
    SCHEMA = Schema(
        {
            "run_id": str,
            "task_id": int,
            # TODO: paths can get quite long. Also, datasets can move around. Do we want a
            # more robust way to identify repeat chunks? Maybe an `Indexer`?
            #   - for now don't worry about it
            "dir": str,
            "error_rate": float,
            # TODO: need to handle multiple partitions, one per table. Or have a separate
            # row per table?
            #   - allow list of partitions
            "partitions": object,
        }
    )

    def __init__(self, db_dir: StrOrPath):
        self.db_dir = Path(db_dir)
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load(self.db_dir)
        return self._df

    @staticmethod
    def _load(db_dir: StrOrPath) -> pd.DataFrame:
        """
        Load the log of processed directories from the database directory ``db_dir``.
        """
        # TODO: should these go in the db_dir or the log_dir? Or should there even be
        # two separate dirs?
        log_path = Path(db_dir) / "_log"
        if not log_path.exists():
            df = ProcessedLog.SCHEMA.empty()
        else:
            batches = []
            for path in sorted(log_path.glob("**/*.json")):
                batches.append(pd.read_json(path, lines=True))
            if len(batches) == 0:
                df = ProcessedLog.SCHEMA.empty()
            else:
                df = pd.concat(batches, ignore_index=True)
                df = df.drop_duplicates(subset="path", keep="last")
        return df

    def write(
        self,
        run_id: str,
        task_id: int,
        dirpath: StrOrPath,
        error_rate: float,
        partitions: List[str],
    ):
        """
        Append a record to the processed log table.
        """
        dirpath = str(Path(dirpath).absolute())
        partitions = [str(Path(part).relative_to(self.db_dir)) for part in partitions]
        record = {
            "run_id": run_id,
            "task_id": task_id,
            "dir": dirpath,
            "error_rate": error_rate,
            "partitions": partitions,
        }
        log_dir = self.db_dir / "_log" / run_id
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        json_path = log_dir / (_format_task_id(task_id) + ".json")
        with open(json_path, "a") as f:
            print(json.dumps(record), file=f)

    def filter_paths(
        self, paths: np.ndarray, error_rate_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Filter a list of paths against those we've already processed successfully.
        """
        # Pick out the processed paths that are also in the current todo list.
        overlap_mask = np.isin(self.df["dir"], paths)
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
