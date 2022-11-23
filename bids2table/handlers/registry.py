import logging
from typing import Dict, Optional, Type

from .handler import Handler

__all__ = ["register_handler", "get_handler"]

# Adapted this logic for registering plugins from VISSL:
# https://github.com/facebookresearch/vissl/blob/main/vissl/models/heads/__init__.py
_HANDLER_REGISTRY: Dict[str, Type[Handler]] = {}


def register_handler(
    cls: Optional[Type[Handler]] = None, *, name: Optional[str] = None
):
    """
    A decorator to register a ``Handler``

    .. code-block:: python
        @register_handler(name="my_handler")
        class MyHandler(Handler):
            ...

    If no ``name`` is passed, the handler ``__name__`` is used.

    To retrieve a ``Handler``, see :func:`get_handler`.
    """

    def decorator(cls_: Type[Handler]) -> Type[Handler]:
        name_ = getattr(cls_, "__name__", None) if name is None else name
        if name_ is None:
            raise RuntimeError(
                "A name is required to register an object without a __name__ attribute"
            )
        if name_ in _HANDLER_REGISTRY:
            logging.warning(f"Handler '{name_}' is already registered; overwriting")
        _HANDLER_REGISTRY[name_] = cls_
        return cls_

    return decorator if cls is None else decorator(cls)


def get_handler(name: str) -> Type[Handler]:
    """
    Get the handler registered under ``name``.
    """
    if name not in _HANDLER_REGISTRY:
        raise KeyError(f"Handler '{name}' is not registered")
    return _HANDLER_REGISTRY[name]
