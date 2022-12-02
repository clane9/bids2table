from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING
from typing_extensions import Protocol, runtime_checkable

from bids2table import RecordDict, StrOrPath

__all__ = ["Loader", "LoaderConfig"]


@dataclass
class LoaderConfig:
    name: str = MISSING
    kwargs: Optional[Dict[str, Any]] = None


@runtime_checkable
class Loader(Protocol):
    def __call__(self, path: StrOrPath) -> Optional[RecordDict]:
        ...
