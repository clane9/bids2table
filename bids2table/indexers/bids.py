import logging
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


# TODO: generalize this to defaults per known name
class BIDS_PATTERNS:
    """
    Regex patterns for a few BIDS entitites

        - suffix: ``"abc_bold.nii.gz"`` -> ``"bold"``
        - extension: ``"abc_bold.nii.gz"`` -> ``".nii.gz"``
    """

    suffix = r"_([a-zA-Z0-9]*?)\.[^/]+$"
    extension = r".*?(\.[^/]+)$"


@dataclass
class BIDSEntityConfig:
    name: str = MISSING
    key: Optional[str] = None
    pattern: Optional[str] = None
    dtype: str = "str"


@dataclass
class BIDSIndexerConfig(IndexerConfig):
    name: str = "bids_indexer"
    columns: List[BIDSEntityConfig] = MISSING
    fields: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, str]] = None


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
    ):
        if key is None:
            key = name
        if pattern is None:
            pattern = f"(?:[_/]|^){key}-(.+?)(?:[._/]|$)"
        if not (dtype in BIDS_DTYPES or dtype in BIDS_DTYPES.values()):
            raise ValueError(
                f"Unexpected dtype {dtype}; expected one of "
                ", ".join(BIDS_DTYPES.keys())
            )

        self.name = name
        self.key = key
        self.pattern = pattern
        self.dtype = dtype

        self._dtype = BIDS_DTYPES[dtype] if isinstance(dtype, str) else dtype
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
        path = Path(path).as_posix()
        match = self._p.search(path)
        if match is None:
            return None
        value = self._dtype(match.group(1))
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
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', key='{self.key}', "
            f"pattern={repr(self.pattern)}, dtype={repr(self.dtype)})"
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
        key = {}
        for col in self.columns:
            val = col.search(path)
            if val is None:
                logging.info(f"Missing index field '{col.name}' in {path}; discarding")
                return None
            key[col.name] = val
        return key

    @classmethod
    def from_config(cls, cfg: BIDSIndexerConfig) -> "BIDSIndexer":  # type: ignore[override]
        """
        Create a BIDS indexer from a config.
        """
        if cfg.fields or cfg.metadata:
            logging.warning("fields and metadata are not used in BIDSIndexerConfig")
        columns = [BIDSEntity.from_config(entcfg) for entcfg in cfg.columns]
        return cls(columns=columns)
