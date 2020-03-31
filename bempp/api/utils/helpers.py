"""Various helper routines."""

import numpy as _np
import collections as _collections


def create_unique_id():
    """Create a unique id."""
    from uuid import uuid4

    return str(uuid4())


def align_array(arr, dtype, order):
    """
    Make sure that an array is contiguous and aligned with the right type.

    If order='F' use Fortran order. If order='C' use
    C order.

    """

    if order == "F":
        requirements = ["A", "F", "O", "E"]
    elif order == "C":
        requirements = ["A", "C", "O", "E"]
    else:
        raise ValueError("order must be one of 'C' or 'F'.")

    return _np.require(arr, dtype, requirements=requirements)


def assign_parameters(parameters):
    """
    Assigns a parameter object based on input.

    If parameters is None return the global_parameters object.
    Otherewise, return the parameters object again.

    """
    import bempp.api

    if parameters is None:
        new_parameters = bempp.api.GLOBAL_PARAMETERS
    else:
        new_parameters = parameters

    return new_parameters


def promote_to_double_precision(array):
    """Convert an array to real or complex double precision."""

    if array.dtype == "float32":
        return array.astype("float64", copy=False)
    if array.dtype == "complex64":
        return array.astype("complex128", copy=False)
    return array


def serialise_list_of_lists(array):
    """
    Serialises a list of lists (or other iterable).

    Returns a tuple (new_array, index_ptr), such
    that array[j] = new_array[index_ptr[j] : index_ptr[j + 1]]

    """

    new_list = []
    index_ptr = [0]

    count = 0
    for sublist in array:
        new_list.extend(sublist)
        count += len(sublist)
        index_ptr.append(count)
    return new_list, index_ptr


TypeContainer = _collections.namedtuple("TypeContainer", "real complex opencl")


def get_type(precision):
    """Return a TypeContainer depending on the given precision."""
    if precision == "single":
        return TypeContainer("float32", "complex64", "float")
    if precision == "double":
        return TypeContainer("float64", "complex128", "double")
    raise ValueError("precision must be one of 'single' or 'double'")


class MemProfiler:
    """Context manager to measure mem usage in bytes."""

    def __init__(self):
        """Constructor."""
        import psutil
        import os

        self._process = psutil.Process(os.getpid())
        self.start = 0
        self.end = 0
        self.interval = 0

    def __enter__(self):
        import gc

        self.start = self._process.memory_info()[0]
        return self

    def __exit__(self, *args):
        import gc

        self.end = self._process.memory_info()[0]
        self.interval = self.end - self.start


def numba_decorate(fun):
    """Numba decorator for functions."""
    import bempp.api
    import numba

    if not bempp.api.USE_JIT:
        return fun
    else:
        return numba.jit(
            nopython=True, parallel=True, error_model="numpy", fastmath=True
        )(fun)
