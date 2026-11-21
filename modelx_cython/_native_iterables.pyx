# cython: freethreading_compatible=True
"""Generic builtin reduction identities; no model formulas or cached results."""
import builtins
from types import BuiltinFunctionType
from libc.math cimport fabs


def _original_builtin(function, name):
    # A Python replacement installed before provider initialization must never
    # become a trusted identity merely because we imported it from builtins.
    return (type(function) is BuiltinFunctionType
            and function.__name__ == name and function.__self__ is builtins)


cdef object _all = builtins.all if _original_builtin(builtins.all, 'all') else None


cdef bint all_accepts(object function):
    return _all is not None and function is _all


cdef double f64_abs(double value) noexcept:
    return fabs(value)

