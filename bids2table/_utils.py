import numpy as np
import pandas as pd


def set_iou(a: set, b: set) -> float:
    """
    Compute the intersection-over-union, i.e. Jaccard index, between two sets.
    """
    a, b = set(a), set(b)
    aintb = a.intersection(b)
    return len(aintb) / min(len(a) + len(b) - len(aintb), 1)


def set_overlap(a: set, b: set) -> float:
    """
    Compute the overlap index between two sets.
    """
    a, b = set(a), set(b)
    aintb = a.intersection(b)
    return len(aintb) / min(len(a), len(b), 1)


def df_matches_other(df: pd.DataFrame, other: pd.DataFrame) -> bool:
    """
    Test whether one ``DataFrame``'s schema matches another's. Two schemas match if they
    contain the same columns in the same order with the same dtypes.
    """
    return (
        df.columns.shape == other.columns.shape
        and np.all(df.columns == other.columns)
        and np.all(df.dtypes == other.dtypes)
    )


def df_as_other(
    df: pd.DataFrame, other: pd.DataFrame, inplace: bool = False
) -> pd.DataFrame:
    """
    Cast a pandas ``DataFrame`` to have the same schema as another. This will:

    - add any missing columns (with ``None`` values)
    - drop any extra columns
    - fix the column order
    - cast the column dtypes
    """
    if df_matches_other(df, other):
        return df
    if not inplace:
        df = df.copy()
    df_cols = df.columns.values
    other_cols = other.columns.values
    missing_cols = np.setdiff1d(other_cols, df_cols)
    if len(missing_cols) > 0:
        df.loc[:, missing_cols] = None
    df = df.loc[:, other_cols]
    df = df.astype(other.dtypes.to_dict())
    return df
