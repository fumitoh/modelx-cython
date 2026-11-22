# cython: freethreading_compatible=True
"""Direct Cython helper ABI. No source, AST, expression tree or formula parser."""
import builtins
from types import BuiltinFunctionType
from cpython.version cimport PY_VERSION_HEX

try:
    from modelx_helpers.expression_streams import _SUM_VALUES
except ModuleNotFoundError as error:
    if error.name not in ('modelx_helpers', 'modelx_helpers.expression_streams'):
        raise
    # Builtin reductions use this provider even in an upstream-only model.
    _SUM_VALUES = object()
cdef object _sum_values = _SUM_VALUES
cdef object _builtin_sum = builtins.sum
if (type(_builtin_sum) is not BuiltinFunctionType or _builtin_sum.__name__ != 'sum'
        or _builtin_sum.__self__ is not builtins):
    _builtin_sum = None

cdef bint f64_accepts(object function):
    return (function is _sum_values or function is _builtin_sum and _builtin_sum is not None
            ) and PY_VERSION_HEX >= 0x030c0000


def provider_version():
    return 2
