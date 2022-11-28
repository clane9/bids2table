import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from omegaconf import MISSING

from bids2table import RecordDict, StrOrPath

from .indexer import Indexer, IndexerConfig
from .registry import register_indexer

__all__ = [
    "BIDSEntityConfig",
    "BIDSIndexerConfig",
    "BIDSEntity",
    "BIDSIndexer",
]

BIDSValue = Union[str, int, float]

BIDS_DTYPES: Dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
}


@dataclass
class BIDSEntityConfig:
    name: str = MISSING
    key: Optional[str] = None
    pattern: Optional[str] = None
    dtype: str = "str"
    required: bool = False


@dataclass
class BIDSIndexerConfig(IndexerConfig):
    name: str = "bids_indexer"
    columns: List[BIDSEntityConfig] = MISSING
    fields: None = None
    metadata: None = None


class BIDSEntity:
    """
    A `BIDS entity`_.

    Args:
        name: Entity name.
        key: Path key (default: ``name``).
        pattern: Regex pattern for extracting the value. There should be a single
            (capturing group to capture the value. Should use a posix path separator (/)
            regardless of platform. (default: ``"[_/]{key}-(.+?)[._/]"``).
        dtype: Type name or type.
        required: Whether the entity is required.

    .. note::
        The implementation closely follows ``pybids.layout.Entity`` from `PyBIDS`_. For
        now, I decided not to directly reuse the API to spare the dependency and to not
        be strictly bound by the BIDS spec. But it might be good to revisit this.

    .. _BIDS entity: https://bids-specification.readthedocs.io/en/stable/02-common-principles.html#entities
    .. _PyBIDS: https://bids-standard.github.io/pybids/index.html
    """

    def __init__(
        self,
        name: str,
        key: Optional[str] = None,
        pattern: Optional[str] = None,
        dtype: Union[str, type] = "str",
        required: bool = False,
    ):
        if key is None:
            key = name
        if pattern is None:
            pattern = f"[_/]{key}-(.+?)[._/]"
        if not (dtype in BIDS_DTYPES or dtype in BIDS_DTYPES.values()):
            raise ValueError(
                f"Unexpected dtype {dtype}; expected one of "
                ", ".join(BIDS_DTYPES.keys())
            )

        self.name = name
        self.key = key
        self.pattern = pattern
        self.dtype = BIDS_DTYPES[dtype] if isinstance(dtype, str) else dtype
        self.required = required

        self._p = re.compile(pattern)
        if self._p.groups != 1:
            raise ValueError(
                f"pattern '{pattern}' expected to have exactly one (capturing) group."
            )

    def search(self, path: StrOrPath) -> Optional[BIDSValue]:
        """
        Search a path ``path`` for the BIDS entity and return its value, or ``None`` if
        the key is not found.

        Note that the path is first converted to posix with forward (/) slashes.
        """
        path = Path(path).absolute().as_posix()
        match = self._p.search(path)
        if match is None:
            return None
        value = self.dtype(match.group(1))
        return value

    @classmethod
    def from_config(cls, cfg: BIDSEntityConfig) -> "BIDSEntity":
        """
        Create a BIDS entity from a config.
        """
        return cls(
            name=cfg.name,
            key=cfg.key,
            pattern=cfg.pattern,
            dtype=cfg.dtype,
            required=cfg.required,
        )


@register_indexer(name="bids_indexer")
class BIDSIndexer(Indexer):
    """
    Indexer for a `BIDS`_ analysis directory.

    .. _BIDS: https://bids-specification.readthedocs.io

    Args:
        columns: list of BIDS entities making up the index.
    """

    def __init__(self, columns: List[BIDSEntity]):
        if len(columns) == 0:
            raise ValueError("At least one column required")

        super().__init__(fields={col.name: col.dtype for col in columns})
        self.columns = columns

    def _load(self, path: StrOrPath) -> Optional[RecordDict]:
        path = Path(path)
        record = {col.name: col.search(path) for col in self.columns}
        return record

    @classmethod
    def from_config(cls, cfg: BIDSIndexerConfig) -> "BIDSIndexer":  # type: ignore[override]
        """
        Create a BIDS indexer from a config.
        """
        columns = [BIDSEntity.from_config(entcfg) for entcfg in cfg.columns]
        return cls(columns=columns)
