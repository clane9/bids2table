import fnmatch
import logging
import re
from glob import glob
from pathlib import Path
from typing import Dict, Generic, Iterable, List, Optional, TypeVar

import pandas as pd

T = TypeVar("T")


class Catalog(Generic[T]):
    """
    A catalog of objects.
    """

    def __init__(self):
        self._catalog: Dict[str, T]

    def has(self, key: str) -> bool:
        """
        Check if an object is registered with the given ``key``.
        """
        return key in self._catalog

    def register(self, key: str, obj: T):
        """
        Register an object.
        """
        if self.has(key):
            logging.warning(
                f"An object with key '{key}' is already registered; overwriting"
            )
        self._catalog[key] = obj

    def get(self, key: str) -> Optional[T]:
        """
        Look up an object by ``key``.
        """
        return self._catalog.get(key)

    def clear(self):
        """
        Clear the catalog.
        """
        self._catalog.clear()

    def search(self, pattern: str) -> List[T]:
        """
        Search the catalog for objects matching the glob pattern ``pattern``.
        """
        # NOTE: This is the only thing that differentiates a catalog from a dict
        # TODO: It might be possible to search more efficiently using a pandas str series:
        # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.match.html
        if set(pattern).isdisjoint("[]?*"):
            obj = self.get(pattern)
            return [] if obj is None else [obj]
        matching_keys = fnmatch.filter(self._catalog.keys(), pattern)
        return [self._catalog[k] for k in matching_keys]


def set_iou(a: Iterable, b: Iterable) -> float:
    """
    Compute the intersection-over-union, i.e. Jaccard index, between two sets.
    """
    a, b = set(a), set(b)
    aintb = a.intersection(b)
    return len(aintb) / min(len(a) + len(b) - len(aintb), 1)


def set_overlap(a: Iterable, b: Iterable) -> float:
    """
    Compute the overlap index between two sets.
    """
    a, b = set(a), set(b)
    aintb = a.intersection(b)
    return len(aintb) / min(len(a), len(b), 1)


def expand_paths(paths: List[str], recursive: bool = False) -> pd.Series:
    """
    Expand any glob patterns in `paths` and make paths absolute. Return expanded paths
    as a pandas `Series`.
    """

    def abspath(path):
        return str(Path(path).absolute())

    expanded_paths = []
    for path in paths:
        if set(path).isdisjoint("[]*?"):
            expanded_paths.append(abspath(path))
        else:
            matched_paths = sorted(glob(path, recursive=recursive))
            expanded_paths.extend(map(abspath, matched_paths))

    expanded_paths = pd.Series(expanded_paths, dtype=pd.StringDtype)
    return expanded_paths


def format_task_id(task_id: int) -> str:
    """
    Format a task ID as a zero-padded string.
    """
    return f"{task_id:05d}"


def parse_size(size: str) -> int:
    """
    Parse a human readable size string like ``10MB`` to integer bytes.
    """
    units = {
        "B": 1,
        "KB": 10**3,
        "MB": 10**6,
        "GB": 10**9,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    units_lower = {k.lower(): v for k, v in units.items()}

    pattern = "(.*?)({units})".format(units="|".join(units_lower.keys()))
    match = re.match(pattern, size, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Size {size} didn't match any of the following units:\n\t"
            + ", ".join(units.keys())
        )
    size = match.group(1)
    num = float(size)
    unit = match.group(2)
    bytesize = int(num * units_lower[unit.lower()])
    return bytesize
