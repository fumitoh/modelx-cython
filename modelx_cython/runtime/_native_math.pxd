ctypedef fused MathArgument:
    object
    double

cdef double erf(object function, MathArgument value) except? -1.0
cdef double exp(object function, MathArgument value) except? -1.0
cdef bint isinf(object function, MathArgument value) except -1
cdef double log(object function, MathArgument value) except? -1.0
cdef double sqrt(object function, MathArgument value) except? -1.0

# W1.4: build-time resolved members of a module Reference (namespace identity, no
# per-call attribute read); see _native_math.pyx.
ctypedef fused MathNumber:
    double
    long long

ctypedef fused MathValue:
    double
    long long
    object

ctypedef fused MathValue2:
    double
    long long
    object

cdef object math_module
cdef object numeric_math_module
# diag-mt2: ``namespace`` is the module Reference as a BORROWED pointer: the generated call
# passes ``cast(p_void, self.<ref>)`` (the Space keeps the Reference alive), so a per-step
# call takes no reference on the module every worker shares (an atomic increment and decrement
# on one object per call at four workers). Identity checks compare pointers; the Python
# fallback for another namespace turns it into an object there.
cdef double m_erf(void* namespace, MathValue value) except? -1.0
cdef double m_exp(void* namespace, MathValue value) except? -1.0
cdef bint m_isinf(void* namespace, MathValue value) except -1
cdef double m_log(void* namespace, MathValue value) except? -1.0
cdef double m_sqrt(void* namespace, MathValue value) except? -1.0
cdef long long m_floor_ll(void* namespace, MathNumber value) except? -1
# W1.8: math.isclose; ``given`` bit 1: rel_tol written, bit 2: abs_tol written
cdef bint m_isclose(void* namespace, MathValue a, MathValue2 b, double rel_tol, double abs_tol,
                    int given) except -1
cdef long long m_ceil_ll(void* namespace, MathNumber value) except? -1
cdef object m_floor(void* namespace, object value)
cdef object m_ceil(void* namespace, object value)
cdef object m_constant(void* namespace, Py_ssize_t index)

# W1.4: min/max of typed operands as Python evaluates them: the first operand is kept
# unless a later one compares strictly smaller (min) or larger (max); nan and -0.0
# follow from the same comparison. Nested left to right for n arguments.
cdef inline double min_f64(double a, double b) noexcept:
    return b if b < a else a

cdef inline double max_f64(double a, double b) noexcept:
    return b if b > a else a

cdef inline long long min_i64(long long a, long long b) noexcept:
    return b if b < a else a

cdef inline long long max_i64(long long a, long long b) noexcept:
    return b if b > a else a

# Mixed operands (one double, one C integer): Python compares them exactly and returns
# the selected operand with its own class, so only the result is an object.
from libc.math cimport floor as _floor

cdef inline int _order_f64_i64(double d, long long i) noexcept:
    """-1, 0 or 1 as d is below, equal to or above i (exactly); 2 when d is nan."""
    cdef double whole
    cdef long long w
    if d != d:
        return 2
    if d < -9223372036854775808.0:
        return -1
    if d >= 9223372036854775808.0:
        return 1
    whole = _floor(d)
    w = <long long>whole
    if w < i:
        return -1
    if w > i:
        return 1
    return 0 if d == whole else 1

cdef inline object min_f64_i64(double a, long long b):
    return b if _order_f64_i64(a, b) == 1 else a          # b < a

cdef inline object min_i64_f64(long long a, double b):
    return b if _order_f64_i64(b, a) == -1 else a         # b < a

cdef inline object max_f64_i64(double a, long long b):
    return b if _order_f64_i64(a, b) == -1 else a         # b > a

cdef inline object max_i64_f64(long long a, double b):
    return b if _order_f64_i64(b, a) == 1 else a          # b > a
