import ast
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bids2table.types import RecordDict, StrOrPath

from .registry import register_loader

# Design considerations
#   - short simple generic file loaders matching the `Loader` interface.
#   - kwargs to enable partialing (but not too many! don't want to have to grok a long
#     list of args).
#   - option to return None for empty or invalid inputs
#       - TODO: returning None effectively silences the warning. Should the loader just
#         assume the input is valid and handle everything outside?


@register_loader
def load_single_row_tsv(
    path: StrOrPath,
    *,
    sep: str = "\t",
    deserialize: bool = True,
    **kwargs,
) -> Optional[RecordDict]:
    """
    Load a data record represented as a single row tsv file. If ``deserialize`` is
    ``True``, will attempt to deserialize each string entry using ``ast.literal_eval``.
    ``**kwargs`` are pased through to ``pandas.read_csv``.

    .. warning::
        numeric-like strings, strings with quotes or back slashes, and other corner
        cases are likely not handled correctly.
    """
    df: pd.DataFrame = pd.read_csv(path, sep=sep, **kwargs)
    record = df.to_dict(orient="records")

    def loads(v):
        if not isinstance(v, str):
            return v
        elif v.isnumeric():
            return v
        else:
            try:
                v = ast.literal_eval(v)
            except Exception:
                pass
        return v

    if len(record) > 0:
        record = record[0]
        if deserialize:
            record = {k: loads(v) for k, v in record.items()}
    else:
        record = None
    return record


# TODO: How to handle the case where there are many variations of the same type of file.
# E.g. many tsv matrices corresponding to different atlases.
#
#   sub-NDARAA306NT2_ses-HBNsiteRU_task-rest_run-1_atlas-Juelichspace-MNI152NLin6res-2x2x2_desc-PearsonNilearn_correlations.tsv
#
# We don't necessarily want each one in a different row. But we don't want to have to
# write a separate handler for each either.
#
# I think the approach should be to generate handlers in a loop. And then possibly be
# able select handlers using a glob pattern.


@register_loader
def load_array_tsv(
    path: StrOrPath,
    *,
    sep: str = "\t",
    dtype: str = "float",
    triangular: bool = False,
    tri_k: int = 0,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load an array (vector or matrix) represented as a tsv file.

    If ``triangular``, the array is assumed to be a matrix. The upper triangular
    portion of the matrix is extracted and flattened, starting from the
    ``tri_k`` diagonal.

    Returns a ``dict`` with a single key ``"array"`` mapping to a nested
    ``dict`` with the following fields:

        shape: original array shape
        data: flattened numpy array
        triangular: ``bool`` indicating whether just the upper triangular
            portion is stored
        tri_k: triangular diagonal offset
    """
    arr: np.ndarray = np.loadtxt(path, delimiter=sep, dtype=dtype)
    shape = arr.shape
    if triangular:
        assert arr.ndim == 2, "a matrix is expected when ``triangular`` is ``True``"
        n, m = shape
        arr = arr[np.triu_indices(n=n, k=tri_k, m=m)]
    else:
        arr = arr.flatten()
    record = {
        "array": {"shape": shape, "data": arr, "triangular": triangular, "tri_k": tri_k}
    }
    return record


@register_loader
def load_dataframe_tsv(
    path: StrOrPath,
    *,
    sep: str = "\t",
    **kwargs,
) -> Optional[RecordDict]:
    """
    Load a tsv dataframe as a single record where each key maps to a
    one-dimensional numpy array.
    """
    df: pd.DataFrame = pd.read_csv(path, sep=sep, **kwargs)
    record = df.to_dict(orient="series")
    record = {k: v.values for k, v in record.items()}
    return record


@register_loader
def load_json_dict(path: StrOrPath, *, nested: bool = True) -> Optional[RecordDict]:
    """
    Load a json dictionary. If ``nested`` is ``False`` containers (i.e. lists, dicts)
    are discarded.
    """
    with open(path) as f:
        record = json.load(f)
    if not isinstance(record, dict):
        return None
    if not nested:
        del_keys = [k for k, v in record.items() if isinstance(v, (list, tuple, dict))]
        for k in del_keys:
            del record[k]
    return record
