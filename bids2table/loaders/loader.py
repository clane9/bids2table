from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING
from typing_extensions import Protocol, runtime_checkable

from bids2table import RecordDict, StrOrPath

from .registry import get_loader

__all__ = ["LoaderConfig", "Loader", "loader_from_config"]


@dataclass
class LoaderConfig:
    name: str = MISSING
    kwargs: Optional[Dict[str, Any]] = None


@runtime_checkable
class Loader(Protocol):
    @staticmethod
    def __call__(path: StrOrPath) -> Optional[RecordDict]:
        ...


def loader_from_config(cfg: LoaderConfig) -> Loader:
    """
    Create a ``Loader`` from a config.
    """
    loader = get_loader(cfg.name)

    if not cfg.kwargs:
        return loader

    def partial_loader(path: StrOrPath) -> Optional[RecordDict]:
        return loader(path, **cfg.kwargs)

    return partial_loader
