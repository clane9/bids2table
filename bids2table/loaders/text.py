import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bids2table.loaders import RecordDict, StrOrPath

# Design considerations
#   - short simple generic file loaders matching the `Loader` interface.
#   - kwargs to enable partialing (but not too many! don't want to have to grok a long
#     list of args).
#   - option to return None for empty or invalid inputs
#       - TODO: returning None effectively silences the warning. Should the loader just
#         assume the input is valid and handle everything outside?


def load_single_row_tsv(path: StrOrPath, *, sep: str = "\t") -> RecordDict:
    """
    Load a data record represented as a single row tsv file.
    """
    df = pd.read_csv(path, sep=sep)
    record = df.to_dict(orient="record")
    record = record[0] if len(record) > 0 else None
    return record


# TODO: How to handle the case where there are many variations of the same type of file.
# E.g. many tsv matrices corresponding to different atlases.
#
#   sub-NDARAA306NT2_ses-HBNsiteRU_task-rest_run-1_atlas-Juelichspace-MNI152NLin6res-2x2x2_desc-PearsonNilearn_correlations.tsv
#
# We don't necessarily want each one in a different row. But we don't want to have to
# write a separate handler for each either.
#
# Is there some way for the context to help us? Maybe the context can return `metadata`
# and `prefix`. In general, the metadata should help us figure out the row key and
# column group key.


def load_matrix_tsv(
    path: StrOrPath, *, sep: str = "\t", name: str = "matrix"
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a matrix represented as a tsv file. Returns a ``dict`` with a single key
    ``name`` whose value is the numpy matrix.
    """
    df = pd.read_csv(path, sep=sep, header=None)
    mat = df.values
    record = {name: mat} if mat.size > 0 else None
    return record


def load_json_dict(path: StrOrPath, *, nested: bool = False) -> RecordDict:
    """
    Load a json dictionary. By default, nested containers (i.e. lists, dicts) are
    discarded.
    """
    with open(path) as f:
        record = json.load(f)
    if not isinstance(record, dict):
        return None
    if not nested:
        record = {
            k: v for k, v in record.items() if not isinstance(v, (list, tuple, dict))
        }
    return record
