# cython: freethreading_compatible=True
"""Scalar rounding with a memo on worker-exclusive collect receivers."""

import builtins
from types import BuiltinFunctionType

from modelx_cython._rounding cimport native_round_digits


_canonical_python_round = builtins.round
if (type(_canonical_python_round) is not BuiltinFunctionType
        or _canonical_python_round.__self__ is not builtins
        or _canonical_python_round.__name__ != 'round'):
    _canonical_python_round = None


cdef class LookupContext:
    def __reduce__(self):
        # The memo is disposable. A restored receiver starts with an empty,
        # shared context until the collect worker takes exclusive ownership.
        return LookupContext, ()

    def _mx_enable_private_lookup(self):
        self._private_owner = True


cdef double checked_round_float(double value, long long digits) except? -1:
    cdef double result
    if native_round_digits(value, digits, &result):
        return result
    return _canonical_python_round(value, digits)
