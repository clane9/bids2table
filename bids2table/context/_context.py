from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from bids2table import Key, StrOrPath

__all__ = ["Context"]


class Context(ABC):
    """
    A ``Context`` generates the row ``key`` for a given path.

    Sub-classes should implement ``index_names()`` and ``get_key()`` and optionally
    override ``set_root()``.
    """

    def __init__(self) -> None:
        self.root: Optional[Path] = None

    @abstractmethod
    def set_root(self, dirpath: StrOrPath) -> None:
        """
        (Re-)Initialize the root directory.
        """
        self.root = Path(dirpath)

    @abstractmethod
    def index_names(self) -> List[str]:
        """
        Return a list of column names for the index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_key(self, path: StrOrPath) -> Optional[Key]:
        """
        Return the associated key for a given path.
        """
        raise NotImplementedError
