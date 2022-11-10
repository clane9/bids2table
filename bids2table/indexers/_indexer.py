import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Type

from omegaconf import DictConfig

from bids2table import Key, StrOrPath

__all__ = [
    "Indexer",
    "INDEXER_REGISTRY",
    "register_indexer",
    "get_indexer",
]


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
    def from_config(cls, cfg: DictConfig) -> "Indexer":
        return cls(**cfg)


# Copied this logic for registering plugins from VISSL:
# https://github.com/facebookresearch/vissl/blob/main/vissl/models/heads/__init__.py
INDEXER_REGISTRY = {}


def register_indexer(name: str):
    """
    A decorator to register a type of ``Indexer``.

    .. code-block:: python
        @register_indexer("my_indexer")
        class MyIndexer(Indexer):
            ...

    To get an ``Indexer`` from a configuration file, see :func:`get_indexer`.
    """

    def register_indexer_cls(cls: Type[Indexer]) -> Type[Indexer]:
        if name in INDEXER_REGISTRY:
            logging.warning(
                f"An indexer with name '{name}' is already registered; overwriting"
            )
        INDEXER_REGISTRY[name] = cls
        return cls

    return register_indexer_cls


def get_indexer(name: str) -> Type[Indexer]:
    """
    Get the ``Indexer`` class referred to by ``name``.
    """
    if name not in INDEXER_REGISTRY:
        raise KeyError(f"Indexer {name} is not registered")
    return INDEXER_REGISTRY[name]
