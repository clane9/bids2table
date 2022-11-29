from typing import Optional

from typing_extensions import Protocol, runtime_checkable

from bids2table import RecordDict, StrOrPath

__all__ = ["Loader"]


@runtime_checkable
class Loader(Protocol):
    @staticmethod
    def __call__(path: StrOrPath) -> Optional[RecordDict]:
        ...
