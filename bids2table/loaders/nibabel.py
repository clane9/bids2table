from typing import Optional

from bids2table.types import RecordDict, StrOrPath
from bids2table.utils import FilePointer, fix_path

from .registry import register_loader


@register_loader
def load_nibabel_img(
    path: StrOrPath,
    *,
    resolve: bool = False,
    parent: Optional[str] = None,
    posix: bool = False,
) -> Optional[RecordDict]:
    """
    Return a ``dict`` with a single key ``"image"`` whose value is a pointer to a
    nibabel image. The path is optionally resolved and relative to ``parent`` and as a
    posix path if ``posix`` is ``True``.
    """
    import nibabel as nib

    path = fix_path(path, resolve=resolve, parent=parent, posix=posix)
    pointer = FilePointer(path, nib.load)
    return {"image": pointer}
