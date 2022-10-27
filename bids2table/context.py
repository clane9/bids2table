from pathlib import Path
from typing import Any, Dict, Union


class BIDSContext:
    """
    TODO: BIDSContext will handle the inference of metadata like subject, session,
    modality, etc. It can read global info from the directory, as well as local info
    for each path.
    """

    def __init__(self, dirpath: Path):
        self.dirpath = dirpath

    def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        return {}
