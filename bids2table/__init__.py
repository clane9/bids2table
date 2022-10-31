__version__ = "0.1.0"

from pathlib import Path
from typing import Any, Dict, Tuple, Union

StrOrPath = Union[str, Path]
RecordDict = Dict[str, Any]
Index = Union[str, int]
Key = Union[Index, Tuple[Index, ...]]
