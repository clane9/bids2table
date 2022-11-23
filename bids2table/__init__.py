__version__ = "0.1.0"

from pathlib import Path
from typing import Any, Dict, Optional, Union

StrOrPath = Union[str, Path]
RecordDict = Dict[str, Any]

PATH = ["."]


def find_file(path: StrOrPath) -> Optional[Path]:
    """
    Find a file looking through the internal ``PATH``.
    """
    path = Path(path)
    if path.is_absolute():
        return path if path.exists() else None
    for root in PATH:
        if (root / path).exists():
            return root / path
    return None
