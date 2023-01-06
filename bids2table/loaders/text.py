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


@register_loader
def load_array_tsv(
    path: StrOrPath,
    *,
    sep: str = "\t",
    dtype: str = "float",
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load an array (vector or matrix) represented as a tsv file. Returns a ``dict`` with
    a single key ``"array"`` mapping to a numpy ndarray.
    """
    array: np.ndarray = np.loadtxt(path, delimiter=sep, dtype=dtype)
    record = {"array": array}
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
