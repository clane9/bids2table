from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING

from bids2table import RecordDict, StrOrPath

from . import text  # noqa
from .loader import Loader
from .registry import get_loader, register_loader  # noqa


@dataclass
class LoaderConfig:
    name: str = MISSING
    kwargs: Optional[Dict[str, Any]] = None


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
