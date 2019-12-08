"""
Basic helpers that should be available independent of the api.

This is for module level helpers that should be available without importing
bempp.api first.

"""
import time as _time
import functools as _functools
import collections as _collections


def timeit(fun):
    """Decorator to time a method in Bempp"""

    @_functools.wraps(fun)
    def timed_fun(*args, **kwargs):
        """The actual timer function."""
        from bempp.api import log

        start_time = _time.time()
        res = fun(*args, **kwargs)
        end_time = _time.time()
        log(fun.__qualname__ + " : {0:.3e}s".format(end_time - start_time), "timing")
        return res

    return timed_fun


IndexList = _collections.namedtuple("IndexList", ["indices", "indexptr"])
