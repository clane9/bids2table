import json
from typing import Any, Dict, Optional

from bids2table.types import RecordDict, StrOrPath

from .loader import Loader, LoaderConfig
from .registry import get_loader

__all__ = ["PartialLoader"]


class PartialLoader:
    """
    A wrapper for a loader function with partially applied kwargs.
    """

    def __init__(self, loader: Loader, kwargs: Optional[Dict[str, Any]] = None):
        self.loader = loader
        self.kwargs = kwargs

    def __call__(self, path: StrOrPath) -> Optional[RecordDict]:
        return self.loader(path, **(self.kwargs or {}))

    @classmethod
    def from_config(cls, cfg: LoaderConfig) -> "PartialLoader":
        """
        Create a ``PartialLoader`` from a config
        """
        kwargs = None if cfg.kwargs is None else dict(cfg.kwargs)
        return cls(loader=get_loader(cfg.name), kwargs=kwargs)

    def __str__(self) -> str:
        loader_fmt = getattr(self.loader, "__name__", str(self.loader))
        return json.dumps({"name": loader_fmt, "kwargs": self.kwargs})
