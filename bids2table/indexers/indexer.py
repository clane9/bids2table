from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from omegaconf import MISSING

from bids2table import RecordDict, StrOrPath
from bids2table.handlers import Handler
from bids2table.schema import Fields

__all__ = ["IndexerConfig", "Indexer"]


@dataclass
class IndexerConfig:
    name: str = MISSING
    fields: Optional[Dict[str, str]] = MISSING
    metadata: Optional[Dict[str, str]] = None


class Indexer(Handler):
    """
    An ``Indexer`` determines the table row where the record(s) for a path should be
    inserted. It does this by generating a "key" record for each file path.

    It is implemented as a special kind of ``Handler`` with additional methods enabling
    it to access the broader directory context.

    Sub-classes must implement:

    - ``_load()``: generate the row "key" record for given path.

    Sub-classes may also implement:

    - ``set_root()``: update the root directory context.
    """

    _CAST_WITH_NULL = True

    def __init__(
        self,
        fields: Fields,
        metadata: Optional[Dict[str, str]] = None,
    ):
        super().__init__(fields, metadata)
        self.root: Optional[Path] = None

    @abstractmethod
    def _load(self, path: StrOrPath) -> Optional[RecordDict]:
        """
        Generate a row "key" record for a given path, or ``None`` if a valid key can't
        be extracted. Note the record need not match the ``schema``.
        """
        raise NotImplementedError

    def __call__(self, path: StrOrPath) -> Optional[RecordDict]:
        """
        Generate a row "key" record matching the index schema for a given path, or
        ``None`` if a valid key can't be generated.
        """
        return super().__call__(path)

    def set_root(self, dirpath: StrOrPath) -> None:
        """
        (Re-)Initialize the root directory.
        """
        self.root = Path(dirpath)

    @classmethod
    def from_config(cls, cfg: IndexerConfig) -> "Indexer":  # type: ignore[override]
        """
        Initialze an Indexer from a config.
        """
        if cfg.fields is None:
            raise ValueError("cfg.fields is required")
        return cls(fields=cfg.fields, metadata=cfg.metadata)
