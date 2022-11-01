from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from bids2table import Key, StrOrPath


class Context(ABC):
    def __init__(self) -> None:
        self.root: Optional[Path] = None

    @abstractmethod
    def set_root(self, dirpath: StrOrPath) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_key(self, path: StrOrPath) -> Optional[Key]:
        raise NotImplementedError

    @abstractmethod
    def index_names(self) -> List[str]:
        raise NotImplementedError


class BIDSContext(Context):
    """
    TODO: BIDSContext will handle the inference of metadata like subject, session,
    modality, etc. It can read global info from the directory, as well as local info
    for each path.
    """

    def set_root(self, dirpath: StrOrPath):
        # TODO
        self.root = Path(dirpath)

    def get_key(self, path: StrOrPath) -> Optional[Key]:
        # TODO
        return None

    def index_names(self) -> List[str]:
        # TODO
        return []
