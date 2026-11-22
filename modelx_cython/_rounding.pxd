"""Guarded binary64 rounding from the retained scalar-math experiment."""

from libc.math cimport copysign, fabs, floor, fma, isfinite
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_CheckExact
from cpython.long cimport PyLong_AsLongLongAndOverflow, PyLong_CheckExact


cdef inline double _nearest_even(double scaled, double residual) noexcept nogil:
    cdef double lower = floor(scaled)
    cdef double fraction = scaled - lower
    if (fraction > 0.5 or fraction == 0.5
            and (residual > 0.0 or residual == 0.0 and (<long long>lower & 1))):
        return lower + 1.0
    return lower


cdef inline bint native_round_digits(double value, long long digits,
                                     double* result) noexcept nogil:
    cdef double scale = 1.0, magnitude, scaled, residual = 0.0, rounded
    cdef long long i
    if not -22 <= digits <= 22:
        return False
    if not isfinite(value) or value == 0.0:
        result[0] = value
        return True
    # Powers of ten through 10**22 are exact in binary64.
    for i in range(digits if digits >= 0 else -digits):
        scale *= 10.0
    magnitude = fabs(value)
    scaled = magnitude * scale if digits >= 0 else magnitude / scale
    if not scaled < 4503599627370496.0:
        return False
    if scaled - floor(scaled) == 0.5:
        # Recover the side of a tie when scaling rounded to the midpoint.
        residual = (fma(magnitude, scale, -scaled) if digits >= 0
                    else fma(-scaled, scale, magnitude))
    rounded = _nearest_even(scaled, residual)
    result[0] = copysign(rounded / scale if digits >= 0 else rounded * scale, value)
    return True


cdef inline object _mx_round_one(object function, object value, object canonical):
    cdef double number, magnitude
    if canonical is not None and function is canonical and PyFloat_CheckExact(value):
        number = PyFloat_AS_DOUBLE(value)
        magnitude = fabs(number)
        if magnitude < 4503599627370496.0:
            return <long long>copysign(_nearest_even(magnitude, 0.0), number)
    return function(value)


cdef inline object _mx_round_digits(object function, object value, object digits, object canonical):
    cdef int overflow = 0
    cdef long long places
    cdef double result
    if (canonical is not None and function is canonical and PyFloat_CheckExact(value)
            and PyLong_CheckExact(digits)):
        places = PyLong_AsLongLongAndOverflow(digits, &overflow)
        if not overflow and native_round_digits(PyFloat_AS_DOUBLE(value), places, &result):
            return result
    return function(value, digits)
