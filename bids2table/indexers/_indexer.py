from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from bids2table import Key, StrOrPath
from bids2table.utils import Catalog

__all__ = [
    "INDEXER_CATALOG",
    "Indexer",
    "register_indexer",
]

# Adapted this logic for registering plugins from VISSL:
# https://github.com/facebookresearch/vissl/blob/main/vissl/models/heads/__init__.py
INDEXER_CATALOG: Catalog[Type["Indexer"]] = Catalog()


class Indexer(ABC):
    """
    An ``Indexer`` generates the row ``key`` for a given path.

    Sub-classes should implement ``index_names()`` and ``get_key()`` and optionally
    override ``set_root()``.
    """

    def __init__(self) -> None:
        self.root: Optional[Path] = None

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

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Indexer":
        return cls(**cfg)


def register_indexer(name: str):
    """
    A decorator to register a type of ``Indexer``.

    .. code-block:: python
        @register_indexer("my_indexer")
        class MyIndexer(Indexer):
            ...
    """

    def decorator(cls: Type[Indexer]) -> Type[Indexer]:
        INDEXER_CATALOG.register(name, cls)
        return cls

    return decorator
