from libc.math cimport isfinite
from libc.string cimport memcmp


cdef class LookupContext:
    cdef bint _private_owner, round_ready
    cdef double round_input, round_result
    cdef long long round_digits


cdef object _canonical_python_round
cdef double checked_round_float(double value, long long digits) except? -1


cdef inline double repeated_round_float(LookupContext context, object function, double value,
                                         long long digits) except? -1:
    cdef double result
    if function is not _canonical_python_round or _canonical_python_round is None:
        return function(value, digits)
    if (context._private_owner and context.round_ready and context.round_digits == digits
            and memcmp(&context.round_input, &value, sizeof(double)) == 0):
        return context.round_result
    result = checked_round_float(value, digits)
    if context._private_owner and isfinite(value):
        context.round_input = value
        context.round_digits = digits
        context.round_result = result
        context.round_ready = True
    return result
