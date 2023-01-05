from typing import Optional

from bids2table.types import RecordDict, StrOrPath
from bids2table.utils import fix_path

from .registry import register_loader


@register_loader
def load_path(
    path: StrOrPath,
    *,
    resolve: bool = False,
    parent: Optional[str] = None,
    posix: bool = False,
) -> Optional[RecordDict]:
    """
    Return a ``dict`` with a single key ``"path"`` whose value is just the string
    ``path``, optionally resolved and relative to ``parent`` and as a posix path if
    ``posix`` is ``True``.
    """
    path = fix_path(path, resolve=resolve, parent=parent, posix=posix)
    return {"path": path}
