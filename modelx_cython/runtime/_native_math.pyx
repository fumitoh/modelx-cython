# cython: freethreading_compatible=True
"""Use the standard C math operation for native floating-point arguments."""
from libc.math cimport erf as c_erf, exp as c_exp, isinf as c_isinf, log as c_log, sqrt as c_sqrt, isfinite
from math import erf as builtin_erf, exp as builtin_exp, isinf as builtin_isinf, log as builtin_log, sqrt as builtin_sqrt

try:
    from modelx_helpers.numeric_math import erf as python_erf, exp as python_exp, isinf as python_isinf, log as python_log, sqrt as python_sqrt
except ModuleNotFoundError as error:
    if error.name not in ('modelx_helpers', 'modelx_helpers.numeric_math'):
        raise
    python_erf, python_exp, python_isinf, python_log, python_sqrt = (
        builtin_erf, builtin_exp, builtin_isinf, builtin_log, builtin_sqrt)


cdef double erf(object function, MathArgument value) except? -1.0:
    if MathArgument is double:
        if (function is builtin_erf or function is python_erf) and isfinite(value):
            return c_erf(value)
    return function(value)


cdef double exp(object function, MathArgument value) except? -1.0:
    cdef double result
    if MathArgument is double:
        if function is builtin_exp or function is python_exp:
            result = c_exp(value)
            if isfinite(result):
                return result
    return function(value)


cdef bint isinf(object function, MathArgument value) except -1:
    if MathArgument is double:
        if function is builtin_isinf or function is python_isinf:
            return c_isinf(value) != 0
    return function(value)


cdef double log(object function, MathArgument value) except? -1.0:
    if MathArgument is double:
        if (function is builtin_log or function is python_log) and value > 0.0:
            return c_log(value)
    return function(value)


cdef double sqrt(object function, MathArgument value) except? -1.0:
    if MathArgument is double:
        if (function is builtin_sqrt or function is python_sqrt) and value >= 0.0:
            return c_sqrt(value)
    return function(value)


# ---------------------------------------------------------------- build-time resolution (W1.4)
# ``math.<fn>(x)`` on a module Reference is bound at build time: the call passes the
# Reference (a C field of the Space), never ``self.math.<fn>``, so no Python attribute
# is read per step. The C function runs while the Reference is the module the build
# resolved; a replaced Reference keeps its own member, looked up then. Results and
# errors are those of the module: the C path is taken only where libm and CPython agree
# (finite results of float or C-integer arguments; the module raises otherwise).
import math as _math_module
from libc.math cimport floor as c_floor, ceil as c_ceil, fabs
from cpython.float cimport PyFloat_CheckExact, PyFloat_AS_DOUBLE

try:
    import modelx_helpers.numeric_math as _numeric_math_module
except ModuleNotFoundError as error:
    if error.name not in ('modelx_helpers', 'modelx_helpers.numeric_math'):
        raise
    _numeric_math_module = _math_module

cdef object math_module = _math_module
cdef object numeric_math_module = _numeric_math_module
cdef object _math_floor = _math_module.floor
cdef object _math_ceil = _math_module.ceil
cdef tuple _constant_names = ('inf', 'nan', 'pi', 'e', 'tau')
cdef tuple _constant_values = (_math_module.inf, _math_module.nan, _math_module.pi,
                               _math_module.e, _math_module.tau)
cdef double _I64_BOUND = 9223372036854775808.0     # 2**63


cdef inline bint _real(MathValue value, double* out) noexcept:
    """The double the module would compute with, when its C path applies."""
    if MathValue is object:
        if PyFloat_CheckExact(value):
            out[0] = PyFloat_AS_DOUBLE(value)
            return True
        return False
    else:
        out[0] = <double>value        # PyLong_AsDouble rounds a C integer the same way
        return True


cdef inline bint _scalar_math(void* namespace) noexcept:
    return namespace == <void*>math_module or namespace == <void*>numeric_math_module


cdef double m_erf(void* namespace, MathValue value) except? -1.0:
    cdef double x
    if _scalar_math(namespace):
        if _real(value, &x) and isfinite(x):
            return c_erf(x)
        return builtin_erf(value)
    return (<object>namespace).erf(value)


cdef double m_exp(void* namespace, MathValue value) except? -1.0:
    cdef double x, result
    if _scalar_math(namespace):
        if _real(value, &x):
            result = c_exp(x)
            if isfinite(result):
                return result
        return builtin_exp(value)
    return (<object>namespace).exp(value)


cdef bint m_isinf(void* namespace, MathValue value) except -1:
    cdef double x
    if _scalar_math(namespace):
        if _real(value, &x):
            return c_isinf(x) != 0
        return builtin_isinf(value)
    return (<object>namespace).isinf(value)


cdef double m_log(void* namespace, MathValue value) except? -1.0:
    cdef double x
    if _scalar_math(namespace):
        if _real(value, &x) and x > 0.0:
            return c_log(x)
        return builtin_log(value)
    return (<object>namespace).log(value)


cdef double m_sqrt(void* namespace, MathValue value) except? -1.0:
    cdef double x
    if _scalar_math(namespace):
        if _real(value, &x) and x >= 0.0:
            return c_sqrt(x)
        return builtin_sqrt(value)
    return (<object>namespace).sqrt(value)


cdef long long m_floor_ll(void* namespace, MathNumber value) except? -1:
    """``math.floor`` of a typed number as a C integer (the int the module returns).

    Outside [-2**63, 2**63) the module's own result or error is taken; an int beyond
    the C range then raises OverflowError (the typed integer domain of the model).
    """
    cdef double rounded
    if namespace == <void*>math_module:
        if MathNumber is double:
            if isfinite(value):
                rounded = c_floor(value)
                if -_I64_BOUND <= rounded < _I64_BOUND:
                    return <long long>rounded
            return _math_floor(value)
        else:
            return value
    return (<object>namespace).floor(value)


cdef long long m_ceil_ll(void* namespace, MathNumber value) except? -1:
    """``math.ceil`` of a typed number as a C integer; see ``m_floor_ll``."""
    cdef double rounded
    if namespace == <void*>math_module:
        if MathNumber is double:
            if isfinite(value):
                rounded = c_ceil(value)
                if -_I64_BOUND <= rounded < _I64_BOUND:
                    return <long long>rounded
            return _math_ceil(value)
        else:
            return value
    return (<object>namespace).ceil(value)


cdef object m_floor(void* namespace, object value):
    """``math.floor`` of any object: an exact float takes the C path, the rest the module."""
    cdef double rounded
    if namespace == <void*>math_module:
        if PyFloat_CheckExact(value) and isfinite(PyFloat_AS_DOUBLE(value)):
            rounded = c_floor(PyFloat_AS_DOUBLE(value))
            if -_I64_BOUND <= rounded < _I64_BOUND:
                return <long long>rounded
        return _math_floor(value)
    return (<object>namespace).floor(value)


cdef object m_ceil(void* namespace, object value):
    """``math.ceil`` of any object; see ``m_floor``."""
    cdef double rounded
    if namespace == <void*>math_module:
        if PyFloat_CheckExact(value) and isfinite(PyFloat_AS_DOUBLE(value)):
            rounded = c_ceil(PyFloat_AS_DOUBLE(value))
            if -_I64_BOUND <= rounded < _I64_BOUND:
                return <long long>rounded
        return _math_ceil(value)
    return (<object>namespace).ceil(value)


cdef bint m_isclose(void* namespace, MathValue a, MathValue2 b, double rel_tol, double abs_tol,
                    int given) except -1:
    """``math.isclose(a, b[, rel_tol=][, abs_tol=])`` (W1.8), CPython's ``math_isclose_impl``:
    the arguments become doubles as the module converts them (``PyFloat_AsDouble``, a then
    b), negative tolerances raise its ValueError, equal values (also equal infinities) are
    close, an infinity is close to nothing else, otherwise ``|b - a|`` is compared with
    ``rel_tol * |b|``, ``rel_tol * |a|`` and ``abs_tol``; nan is close to nothing.
    """
    cdef double x, y, diff
    if namespace == <void*>math_module:
        x = a
        y = b
        if rel_tol < 0.0 or abs_tol < 0.0:
            raise ValueError("tolerances must be non-negative")
        if x == y:
            return True
        if c_isinf(x) or c_isinf(y):
            return False
        diff = fabs(y - x)
        return diff <= fabs(rel_tol * y) or diff <= fabs(rel_tol * x) or diff <= abs_tol
    if given == 3:
        return (<object>namespace).isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    if given == 1:
        return (<object>namespace).isclose(a, b, rel_tol=rel_tol)
    if given == 2:
        return (<object>namespace).isclose(a, b, abs_tol=abs_tol)
    return (<object>namespace).isclose(a, b)


cdef object m_constant(void* namespace, Py_ssize_t index):
    """``math.inf`` / ``nan`` / ``pi`` / ``e`` / ``tau`` (index into the names above)."""
    if namespace == <void*>math_module:
        return _constant_values[index]
    return getattr(<object>namespace, _constant_names[index])
