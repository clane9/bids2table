import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bids2table import Key, StrOrPath
from bids2table.index import Indexer

BIDSValue = Union[str, int, float]

VALID_DTYPES = {
    "str": str,
    "int": int,
    "float": float,
}

NULL_VALUES: Dict[str, BIDSValue] = {
    "str": "",
    "int": -1,
    "float": float("nan"),
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
        dtype: Dtype name, one of ``"str"``, ``"int"``, ``"float"`` (default: ``"str"``).
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
        dtype: str = "str",
        required: bool = False,
    ):
        if key is None:
            key = name
        if pattern is None:
            pattern = f"[_/]{key}-(.+?)[._/]"
        if dtype not in VALID_DTYPES:
            valid_dtypes_str = ", ".join(map(repr, VALID_DTYPES.keys()))
            raise ValueError(
                f"Unexpected dtype {dtype}; expected one of {valid_dtypes_str}"
            )

        self.name = name
        self.key = key
        self.pattern = pattern
        self.dtype = dtype
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
        value = match.group(1)
        value = self._cast(value)
        return value

    def _cast(self, value: str) -> BIDSValue:
        try:
            value = VALID_DTYPES[self.dtype](value)
        except ValueError:
            raise ValueError(f"Unable to cast value {value} to dtype '{self.dtype}'")
        return value


class BIDSIndexer(Indexer):
    """
    Indexer for a `BIDS`_ analysis directory.

    .. _BIDS: https://bids-specification.readthedocs.io

    Args:
        columns: list of BIDS entities making up the index.
    """

    def __init__(self, columns: List[BIDSEntity]):
        for col in columns:
            if col.dtype not in {"str", "int"}:
                raise ValueError(
                    "Only 'str' and 'int' BIDS entities supported in an index."
                )
        self.columns = columns

    @classmethod
    def from_config(cls, cfg: List[Union[str, Dict[str, Any]]]) -> "BIDSIndexer":
        """
        Initialize indexer from a list of column entity configs.
        """
        columns = []
        for entry in cfg:
            if isinstance(entry, str):
                col = BIDSEntity(entry)
            else:
                col = BIDSEntity(**entry)
            columns.append(col)
        return cls(columns)

    def get_key(self, path: StrOrPath) -> Optional[Key]:
        key = []
        for col in self.columns:
            val = col.search(path)
            if val is None:
                if col.required:
                    return None
                val = NULL_VALUES[col.dtype]
            assert isinstance(val, (str, int)), f"Invalid BIDS index column value {val}"
            key.append(val)
        return tuple(key)

    def index_names(self) -> List[str]:
        return [col.name for col in self.columns]
