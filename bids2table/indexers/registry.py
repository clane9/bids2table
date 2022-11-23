import logging
from typing import Dict, Optional, Type

from .indexer import Indexer

__all__ = ["register_indexer", "get_indexer"]

_INDEXER_REGISTRY: Dict[str, Type[Indexer]] = {}


def register_indexer(
    cls: Optional[Type[Indexer]] = None, *, name: Optional[str] = None
):
    """
    A decorator to register a ``Indexer``

    .. code-block:: python
        @register_indexer(name="my_indexer")
        class MyIndexer(Indexer):
            ...

    If no ``name`` is passed, the indexer ``__name__`` is used.

    To retrieve a ``Indexer``, see :func:`get_indexer`.
    """

    def decorator(cls_: Type[Indexer]) -> Type[Indexer]:
        name_ = getattr(cls_, "__name__", None) if name is None else name
        if name_ is None:
            raise RuntimeError(
                "A name is required to register an object without a __name__ attribute"
            )
        if name_ in _INDEXER_REGISTRY:
            logging.warning(f"Indexer '{name_}' is already registered; overwriting")
        _INDEXER_REGISTRY[name_] = cls_
        return cls_

    return decorator if cls is None else decorator(cls)


def get_indexer(name: str) -> Type[Indexer]:
    """
    Get the indexer registered under ``name``.
    """
    if name not in _INDEXER_REGISTRY:
        raise KeyError(f"Indexer '{name}' is not registered")
    return _INDEXER_REGISTRY[name]
