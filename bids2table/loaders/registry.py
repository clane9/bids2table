import logging
from typing import Dict, Optional

from .loader import Loader

__all__ = ["register_loader", "get_loader"]

_LOADER_REGISTRY: Dict[str, Loader] = {}


def register_loader(fun: Optional[Loader] = None, *, name: Optional[str] = None):
    """
    A decorator to register a ``Loader``

    .. code-block:: python
        @register_loader
        def my_loader(path):
            ...

    If no ``name`` is passed, the loader ``__name__`` is used.

    To retrieve a ``Loader``, see :func:`get_loader`.
    """

    def decorator(fun_: Loader) -> Loader:
        name_ = getattr(fun_, "__name__", None) if name is None else name
        if name_ is None:
            raise RuntimeError(
                "A name is required to register an object without a __name__ attribute"
            )
        if name_ in _LOADER_REGISTRY:
            logging.warning(f"Loader '{name_}' is already registered; overwriting")
        _LOADER_REGISTRY[name_] = fun_
        return fun_

    return decorator if fun is None else decorator(fun)


def get_loader(name: str) -> Loader:
    """
    Get the loader registered under ``name``.
    """
    if name not in _LOADER_REGISTRY:
        raise KeyError(f"Loader '{name}' is not registered")
    return _LOADER_REGISTRY[name]
