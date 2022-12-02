__version__ = "0.1.0"

from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Union

StrOrPath = Union[str, Path]
RecordDict = Dict[str, Any]

PATH = ["."]


@contextmanager
def locate_file(path: StrOrPath, pkg: str = __package__):
    """
    A context manager to locate a file. Searches under:

        - ``pkg`` resources
        - all directories in the ``PATH``
    """
    path = Path(path)

    if path.is_absolute():
        yield path if path.exists() else None
    elif resources.is_resource(pkg, path.name):
        with resources.path(pkg, path.name) as p:
            yield p
    else:
        for root in PATH:
            if (root / path).exists():
                yield root / path
                break
        else:
            yield None
