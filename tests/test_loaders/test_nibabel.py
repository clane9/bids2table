import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from bids2table.loaders.nibabel import load_nibabel_img
from bids2table.types import RecordDict
from bids2table.utils import FilePointer


@pytest.fixture
def nifti_path(tmp_path: Path) -> Path:
    fname = tmp_path / "image.nii.gz"
    data = np.ones((10, 8, 6), dtype=np.int16)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, fname)
    return fname


def test_load_nibabel_img(nifti_path: Path):
    record: RecordDict = load_nibabel_img(nifti_path)
    pointer: FilePointer = record["image"]
    logging.info("image: %s", pointer)
    assert isinstance(pointer, FilePointer)

    img = pointer.get()
    assert isinstance(img, nib.Nifti1Image)


if __name__ == "__main__":
    pytest.main([__file__])
