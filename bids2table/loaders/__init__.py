from pathlib import Path
from typing import Any, Dict, Optional, Union

from typing_extensions import Protocol

# TODO: is this the right idiom? What about `os.PathLike`?
StrOrPath = Union[str, Path]
RecordDict = Optional[Dict[str, Any]]


class Loader(Protocol):
    def __call__(self, path: StrOrPath, **kwargs) -> RecordDict:
        ...
