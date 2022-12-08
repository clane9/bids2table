from pathlib import Path
from typing import Any, Dict, Union

__all__ = ["StrOrPath", "RecordDict"]


StrOrPath = Union[str, Path]
RecordDict = Dict[str, Any]
