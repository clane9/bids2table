import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pyarrow as pa
from pandas.api.extensions import ExtensionDtype, register_extension_dtype
from pandas.core.arrays import PandasArray

__all__ = [
    "StrOrPath",
    "RecordDict",
    "PaPickleType",
    "PaPickleScalar",
    "PaPickleArray",
    "PdPickleDtype",
    "PdPickleArray",
]


StrOrPath = Union[str, Path]
RecordDict = Dict[str, Any]


class PaPickleType(pa.PyExtensionType):
    """
    PyArrow binary extension type for holding arbitrary pickled objects.

    See `here <https://arrow.apache.org/docs/python/extending_types.html>`_ for more
    details and examples.
    """

    def __init__(self):
        super().__init__(pa.binary())

    def __reduce__(self):
        return PaPickleType, ()

    def __arrow_ext_scalar_class__(self):
        return PaPickleScalar

    def __arrow_ext_class__(self):
        return PaPickleArray

    def to_pandas_dtype(self):
        return PdPickleDtype()


class PaPickleScalar(pa.ExtensionScalar):
    """
    PyArrow object scalar that deserializes with pickle in python conversion
    (``scalar.as_py()``).
    """

    def as_py(self) -> Any:
        val = self.value
        if val is not None:
            val = pickle.loads(val.as_py())
        return val


class PaPickleArray(pa.ExtensionArray):
    """
    PyArrow pickle array that deserializes with pickle in python conversion
    (``array.to_pylist()``) and numpy conversion (``array.to_numpy()``).
    """

    def to_numpy(self, **kwargs):
        return np.array(self.to_pylist(), dtype=object)


@register_extension_dtype
class PdPickleDtype(ExtensionDtype):
    """
    Pandas extension dtype for arbitrary objects supporting conversion to PyArrow
    (via pickle serialization).

    See `here
    <https://pandas.pydata.org/docs/development/extending.html#extension-types>`_ for
    more details.
    """

    name = "pickle"
    type = object
    kind = "O"
    base = np.dtype("O")
    na_value = None

    @classmethod
    def construct_array_type(cls):
        """Return the array type associated with this dtype."""
        return PdPickleArray

    def __from_arrow__(
        self, array: Union[pa.Array, pa.ChunkedArray]
    ) -> "PdPickleArray":
        data = np.array(array.to_pylist(), dtype=object)
        return PdPickleArray(data)

    def __repr__(self) -> str:
        return "PdPickleDtype()"


class PdPickleArray(PandasArray):
    """
    Pandas extension array for arbitrary objects supporting conversion to PyArrow (via
    pickle serialization).
    """

    # In pandas.core.internals.api.make_block, PandasArray instances are automatically
    # unboxed to plain NumPy arrays. But the result is one-dimensional. When
    # `len(placement) == 1` (as is the case inside `Table.to_pandas()`), a 2d row
    # vector seems to be expected. As a result, `check_ndim` raises errors like:
    #
    #   ValueError: Wrong number of items passed 2, placement implies 1
    #
    # To bypass this shape mismatch issue, we make sure this PandasArray sub-class is
    # *not* automatically unboxed, by changing this _typ attribute. Searching for
    # ABCPandasArray in the pandas codebase gives a bit more background why this works.
    #
    # TLDR: we need to reset _typ for `Table.to_pandas()` to work properly.
    _typ = "extension"
    _dtype = PdPickleDtype()
    _internal_fill_value = None
    _str_na_value = None

    def __init__(self, values: np.ndarray, copy: bool = False):
        values = np.asarray(values, dtype=object)
        if values.ndim != 1:
            raise ValueError("Only one-dimensional arrays supported")
        if copy:
            values = values.copy()
        super(PandasArray, self).__init__(values, PdPickleDtype())

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Any = None, copy: bool = False
    ) -> "PdPickleArray":
        return PdPickleArray(scalars)

    def __arrow_array__(self, type: Optional[pa.DataType] = None) -> PaPickleArray:
        """
        Convert myself into a PyArrow binary Array, using pickle to serialize.
        """
        data = [pickle.dumps(val) for val in self._ndarray]
        data = pa.array(data, type=pa.binary())
        array = PaPickleArray.from_storage(PaPickleType(), data)
        return array
