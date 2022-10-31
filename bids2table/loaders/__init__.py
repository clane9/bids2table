from typing import Optional

from typing_extensions import Protocol

from bids2table import RecordDict, StrOrPath


class Loader(Protocol):
    def __call__(self, path: StrOrPath, **kwargs) -> Optional[RecordDict]:
        ...
