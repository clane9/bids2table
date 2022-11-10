import fnmatch
import logging
from typing import Dict, Generic, Iterable, List, Optional, TypeVar

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
