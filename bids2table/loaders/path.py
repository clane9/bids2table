from pathlib import Path
from typing import Optional

from bids2table.types import RecordDict, StrOrPath

from .registry import register_loader


@register_loader
def load_path(
    path: StrOrPath,
    *,
    parent: Optional[str] = None,
    as_posix: bool = False,
) -> Optional[RecordDict]:
    """
    Return a ``dict`` with a single key ``"path"`` whose value is just the string
    ``path``, optionally relative to ``parent`` and as a posix path if
    ``as_posix`` is ``True``.
    """
    path = Path(path)
    if parent:
        path = path.relative_to(parent)
    path = path.as_posix() if as_posix else str(path)
    return {"path": path}
