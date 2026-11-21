"""Header-only float-start sum state; no input-provider runtime dependency.

Contributions must already be exact double values. Python 3.12 introduced
compensated float summation; older interpreters use ordered simple addition.
This is a numeric reducer, not an arbitrary Python sum state machine.
"""
from cpython.version cimport PY_VERSION_HEX
from libc.math cimport fabs, isfinite

cdef struct F64Sum:
    double total
    double correction


cdef inline void f64_init(F64Sum* state) noexcept:
    state.total = 0.0
    state.correction = 0.0


cdef inline void f64_add(F64Sum* state, double value) noexcept:
    cdef double following = state.total + value
    if PY_VERSION_HEX >= 0x030C0000:
        if fabs(state.total) >= fabs(value):
            state.correction += (state.total - following) + value
        else:
            state.correction += (value - following) + state.total
    state.total = following


cdef inline double f64_result(F64Sum* state) noexcept:
    if state.correction != 0.0 and isfinite(state.correction):
        return state.total + state.correction
    return state.total


# ---------------------------------------------------------------- exact built-in sum
# ``PySum`` reproduces CPython 3.14 ``builtin_sum_impl`` (Python/bltinmodule.c) for
# one ``sum(iterable[, start])`` whose items are produced by native code: the C
# integer fast loop (Py_ssize_t accumulator, overflow leaves it), the compensated
# float loop (Neumaier, ints added through PyLong_AsDouble), the compensated complex
# loop and the generic PyNumber_Add loop, with the same transitions between them.
# Typed items (double, long long, bint) are the exact float/int/bool objects the
# interpreter would see, so they take the same branch without being boxed.
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_XDECREF
from cpython.long cimport PyLong_Check, PyLong_CheckExact, PyLong_AsLongLongAndOverflow, PyLong_AsDouble
from cpython.float cimport PyFloat_Check, PyFloat_CheckExact, PyFloat_AS_DOUBLE, PyFloat_AsDouble, PyFloat_FromDouble
from cpython.complex cimport PyComplex_CheckExact, PyComplex_AsCComplex, PyComplex_FromDoubles, Py_complex
from cpython.bool cimport PyBool_Check
from cpython.number cimport PyNumber_Add
from libc.limits cimport LLONG_MAX, LLONG_MIN

cdef enum:
    SUM_INT = 0
    SUM_FLOAT = 1
    SUM_COMPLEX = 2
    SUM_OBJECT = 3

cdef struct PySum:
    int mode
    long long whole         # SUM_INT accumulator
    double hi, lo           # SUM_FLOAT, and the real part in SUM_COMPLEX
    double ihi, ilo         # imaginary part in SUM_COMPLEX
    PyObject* value         # owned running result in SUM_OBJECT, else NULL


cdef inline void _cs_add(double* hi, double* lo, double x) noexcept:
    cdef double t = hi[0] + x
    if PY_VERSION_HEX >= 0x030C0000:
        if fabs(hi[0]) >= fabs(x):
            lo[0] += (hi[0] - t) + x
        else:
            lo[0] += (x - t) + hi[0]
    hi[0] = t


cdef inline double _cs_value(double hi, double lo) noexcept:
    if lo != 0.0 and isfinite(lo):
        return hi + lo
    return hi


cdef inline void sum_init(PySum* state) noexcept:
    """``sum(iterable)``: start is the int 0."""
    state.mode = SUM_INT
    state.whole = 0
    state.hi = state.lo = state.ihi = state.ilo = 0.0
    state.value = NULL


cdef inline void _sum_hold(PySum* state, object value) noexcept:
    Py_INCREF(value)
    Py_XDECREF(state.value)
    state.value = <PyObject*>value
    state.mode = SUM_OBJECT


cdef inline object _sum_current(PySum* state):
    if state.mode == SUM_INT:
        return state.whole
    if state.mode == SUM_FLOAT:
        return PyFloat_FromDouble(_cs_value(state.hi, state.lo))
    if state.mode == SUM_COMPLEX:
        return PyComplex_FromDoubles(_cs_value(state.hi, state.lo), _cs_value(state.ihi, state.ilo))
    return <object>state.value


cdef inline int _sum_generic(PySum* state, object item) except -1:
    """PyNumber_Add(result, item), then the fast loop that follows the current one."""
    cdef int source = state.mode
    cdef object result = PyNumber_Add(_sum_current(state), item)
    cdef Py_complex z
    if source == SUM_INT and PyFloat_CheckExact(result):
        state.mode = SUM_FLOAT
        state.hi, state.lo = PyFloat_AS_DOUBLE(result), 0.0
    elif source <= SUM_FLOAT and PyComplex_CheckExact(result):
        z = PyComplex_AsCComplex(result)
        state.mode = SUM_COMPLEX
        state.hi, state.lo, state.ihi, state.ilo = z.real, 0.0, z.imag, 0.0
    else:
        _sum_hold(state, result)
    return 0


cdef inline bint _sum_whole(PySum* state, long long b) noexcept:
    if (b <= LLONG_MAX - state.whole) if state.whole >= 0 else (b >= LLONG_MIN - state.whole):
        state.whole += b
        return True
    return False


cdef inline int sum_add_f64(PySum* state, double x) except -1:
    """Add an exact float item."""
    if state.mode == SUM_FLOAT:
        _cs_add(&state.hi, &state.lo, x)
    elif state.mode == SUM_INT:
        state.mode = SUM_FLOAT              # int + float, then the float loop
        state.hi, state.lo = <double>state.whole + x, 0.0
    elif state.mode == SUM_COMPLEX:
        _cs_add(&state.hi, &state.lo, x)
    else:
        _sum_generic(state, PyFloat_FromDouble(x))
    return 0


cdef inline int sum_add_ll(PySum* state, long long v) except -1:
    """Add an exact int item."""
    if state.mode == SUM_INT:
        if not _sum_whole(state, v):
            _sum_generic(state, v)
    elif state.mode == SUM_FLOAT:
        _cs_add(&state.hi, &state.lo, <double>v)
    elif state.mode == SUM_COMPLEX:
        _cs_add(&state.hi, &state.lo, <double>v)
    else:
        _sum_generic(state, v)
    return 0


cdef inline int sum_add_bint(PySum* state, bint v) except -1:
    """Add a bool item (an int everywhere but in the generic loop)."""
    if state.mode == SUM_OBJECT:
        _sum_generic(state, True if v else False)
    else:
        sum_add_ll(state, 1 if v else 0)
    return 0


cdef inline int sum_add_object(PySum* state, object item) except -1:
    """Add any item object, branch for branch as builtin_sum_impl."""
    cdef int overflow = 0
    cdef long long b
    cdef Py_complex z
    if state.mode == SUM_INT:
        if PyLong_CheckExact(item) or PyBool_Check(item):
            b = PyLong_AsLongLongAndOverflow(item, &overflow)
            if overflow == 0 and _sum_whole(state, b):
                return 0
    elif state.mode == SUM_FLOAT:
        if PyFloat_CheckExact(item):
            _cs_add(&state.hi, &state.lo, PyFloat_AS_DOUBLE(item))
            return 0
        if PyLong_Check(item):
            _cs_add(&state.hi, &state.lo, PyLong_AsDouble(item))
            return 0
    elif state.mode == SUM_COMPLEX:
        if PyComplex_CheckExact(item):
            z = PyComplex_AsCComplex(item)
            _cs_add(&state.hi, &state.lo, z.real)
            _cs_add(&state.ihi, &state.ilo, z.imag)
            return 0
        if PyLong_Check(item):
            _cs_add(&state.hi, &state.lo, PyLong_AsDouble(item))
            return 0
        if PyFloat_Check(item):
            _cs_add(&state.hi, &state.lo, PyFloat_AS_DOUBLE(item))
            return 0
    _sum_generic(state, item)
    return 0


cdef inline object sum_result(PySum* state):
    """The value builtin sum returns (a new reference); the state is unchanged."""
    return _sum_current(state)


cdef inline double sum_result_f64(PySum* state) except? -1.0:
    """The result converted at a double boundary, as Cython converts an object."""
    if state.mode == SUM_FLOAT:
        return _cs_value(state.hi, state.lo)
    if state.mode == SUM_INT:
        return <double>state.whole
    return PyFloat_AsDouble(_sum_current(state))


cdef inline void sum_clear(PySum* state) noexcept:
    """Release the generic running result; call on every exit."""
    Py_XDECREF(state.value)
    state.value = NULL
