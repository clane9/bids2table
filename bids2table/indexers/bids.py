import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bids2table import RecordDict, StrOrPath

from .indexer import Indexer
from .registry import register_indexer

__all__ = [
    "BIDSEntity",
    "BIDSIndexer",
]

BIDSValue = Union[str, int, float]

BIDS_DTYPES: Dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
}


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


@register_indexer(name="bids_indexer")
class BIDSIndexer(Indexer):
    """
    Indexer for a `BIDS`_ analysis directory.

    .. _BIDS: https://bids-specification.readthedocs.io

    Args:
        columns: list of BIDS entities making up the index.
    """

    def __init__(
        self,
        columns: List[Union[str, Dict[str, Any], BIDSEntity]],
    ):
        columns_: List[BIDSEntity] = []
        for col in columns:
            if isinstance(col, str):
                col = BIDSEntity(name=col)
            elif isinstance(col, dict):
                col = BIDSEntity(**col)
            elif not isinstance(col, BIDSEntity):
                raise TypeError(f"Invalid BIDS index column {col}")
            columns_.append(col)

        super().__init__(fields={col.name: col.dtype for col in columns_})
        self.columns = columns_

    def _load(self, path: StrOrPath) -> Optional[RecordDict]:
        path = Path(path)
        record = {col.name: col.search(path) for col in self.columns}
        return record
