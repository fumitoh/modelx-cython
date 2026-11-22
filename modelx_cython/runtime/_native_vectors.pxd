"""Typed vectors: the cached value of a cell whose every traced value is a list of
floats (F64Vec) or of integers (I64Vec), and the locals of formulas that build them.

A vector owns a C buffer and its length. The formula that allocates one writes it
(vec_set, vec_append, direct stores in lowered loops) before it returns; after that
nothing writes it, so readers on any thread need no lock. Python sees a read-only
sequence (len, index, slice, iteration, element-wise ==, buffer, pickle).

Sums of float vectors use a compensated state equal to CPython's built-in float
sum bit for bit. It is a private copy of ``_float_sum_state`` (owned by another
package); the two are to be unified.
"""
from cpython.version cimport PY_VERSION_HEX
from cpython.ref cimport PyObject
from libc.math cimport fabs, isfinite


cdef extern from *:
    """
    /* One mechanism on every C compiler: CPython's own atomics. From 3.13 Python.h includes
       cpython/pyatomic.h, whose functions are underscore-prefixed interpreter API (not the
       stable ABI): _Py_atomic_load_ptr_acquire and _Py_atomic_compare_exchange_ptr here,
       _Py_atomic_add_int64 and _Py_atomic_load_int64_relaxed in _native_inputs. Before 3.13
       there is no free-threaded build, and every caller (VecCache.put/get, len, items) runs
       Python object code, so it holds the GIL: plain loads and stores are ordered by it. */
    #if PY_VERSION_HEX >= 0x030D0000
    #define __pyx_vec_load(p) ((PyObject *)_Py_atomic_load_ptr_acquire((p)))
    static CYTHON_INLINE int __pyx_vec_publish(PyObject **slot, PyObject *value) {
        PyObject *expected = NULL;
        return _Py_atomic_compare_exchange_ptr(slot, &expected, value);
    }
    #else
    #define __pyx_vec_load(p) (*(p))
    static CYTHON_INLINE int __pyx_vec_publish(PyObject **slot, PyObject *value) {
        if (*slot != NULL)
            return 0;
        *slot = value;
        return 1;
    }
    #endif
    """
    PyObject* __pyx_vec_load(PyObject** p) noexcept nogil
    bint __pyx_vec_publish(PyObject** slot, PyObject* value) noexcept nogil


cdef class F64Vec:
    cdef double* data
    cdef Py_ssize_t n
    cdef Py_ssize_t cap


cdef class I64Vec:
    cdef long long* data
    cdef Py_ssize_t n
    cdef Py_ssize_t cap


cdef class VecCache:
    """The values of a vector cell with one integer parameter, by index (dense cache).

    A slot is published once (compare-and-swap from NULL) and never replaced while the cache
    lives, so readers on any thread take it with one acquire load and no lock."""
    cdef PyObject** slots
    cdef Py_ssize_t n
    cdef long long lo
    cdef object put(self, long long key, object value)


cdef inline object cache_get(VecCache cache, long long key):
    """The cached value at key, or None (a vector value is never None)."""
    cdef unsigned long long i = <unsigned long long>(key - cache.lo)
    cdef PyObject* p
    if i < <unsigned long long>cache.n:
        p = __pyx_vec_load(&cache.slots[i])
        if p != NULL:
            return <object>p
    return None


# ---------------------------------------------------------------- compensated sum (private copy)
cdef struct VecSum:
    double total
    double correction


cdef inline void sum_init(VecSum* state) noexcept nogil:
    state.total = 0.0
    state.correction = 0.0


cdef inline void sum_add(VecSum* state, double value) noexcept nogil:
    cdef double following = state.total + value
    if PY_VERSION_HEX >= 0x030C0000:
        if fabs(state.total) >= fabs(value):
            state.correction += (state.total - following) + value
        else:
            state.correction += (value - following) + state.total
    state.total = following


cdef inline double sum_result(VecSum* state) noexcept nogil:
    if state.correction != 0.0 and isfinite(state.correction):
        return state.total + state.correction
    return state.total


# ---------------------------------------------------------------- construction
cdef F64Vec vec_new(Py_ssize_t n)
cdef F64Vec vec_full(Py_ssize_t n, double value)
cdef F64Vec vec_of(object value)
cdef F64Vec vec_slice(F64Vec v, long long a, long long b)
cdef int _vec_grow(F64Vec v) except -1

cdef I64Vec ivec_new(Py_ssize_t n)
cdef I64Vec ivec_full(Py_ssize_t n, long long value)
cdef I64Vec ivec_of(object value)
cdef I64Vec ivec_slice(I64Vec v, long long a, long long b)
cdef int _ivec_grow(I64Vec v) except -1
cdef long long _ivec_sum_slow(I64Vec v, Py_ssize_t a, Py_ssize_t b) except? -1

cdef double _vec_get_slow(F64Vec v, long long i) except? -1.0
cdef int _vec_set_slow(F64Vec v, long long i, double value) except -1
cdef long long _ivec_get_slow(I64Vec v, long long i) except? -1
cdef int _ivec_set_slow(I64Vec v, long long i, long long value) except -1


# ---------------------------------------------------------------- element access
cdef inline double vec_get(F64Vec v, long long i) except? -1.0:
    """v[i] with Python index semantics; the in-range test is one unsigned compare."""
    if <unsigned long long>i < <unsigned long long>v.n:
        return v.data[i]
    return _vec_get_slow(v, i)


cdef inline int vec_set(F64Vec v, long long i, double value) except -1:
    if <unsigned long long>i < <unsigned long long>v.n:
        v.data[i] = value
        return 0
    return _vec_set_slow(v, i, value)


cdef inline long long ivec_get(I64Vec v, long long i) except? -1:
    if <unsigned long long>i < <unsigned long long>v.n:
        return v.data[i]
    return _ivec_get_slow(v, i)


cdef inline int ivec_set(I64Vec v, long long i, long long value) except -1:
    if <unsigned long long>i < <unsigned long long>v.n:
        v.data[i] = value
        return 0
    return _ivec_set_slow(v, i, value)


cdef inline int vec_append(F64Vec v, double value) except -1:
    """list.append on a vector the formula allocated (it is not published yet)."""
    if v.n == v.cap:
        _vec_grow(v)
    v.data[v.n] = value
    v.n += 1
    return 0


cdef inline int ivec_append(I64Vec v, long long value) except -1:
    if v.n == v.cap:
        _ivec_grow(v)
    v.data[v.n] = value
    v.n += 1
    return 0


# ---------------------------------------------------------------- bounds of a lowered loop
cdef inline bint in_range(long long lo, long long hi, Py_ssize_t n) noexcept nogil:
    """Whether every index of [lo, hi) lies in [0, n) (an empty range trivially does)."""
    return hi <= lo or (lo >= 0 and hi <= n)


# ---------------------------------------------------------------- sums
cdef inline void _clip(long long* a, long long* b, long long n) noexcept nogil:
    """Python slice normalisation of [a:b] on a length n (step 1)."""
    if a[0] < 0:
        a[0] += n
        if a[0] < 0:
            a[0] = 0
    elif a[0] > n:
        a[0] = n
    if b[0] < 0:
        b[0] += n
        if b[0] < 0:
            b[0] = 0
    elif b[0] > n:
        b[0] = n


cdef inline double vec_sum_range(F64Vec v, long long a, long long b) noexcept:
    """sum(v[a:b]): the float sum CPython computes, bit for bit (start 0)."""
    cdef VecSum s
    cdef Py_ssize_t i
    _clip(&a, &b, v.n)
    sum_init(&s)
    for i in range(<Py_ssize_t>a, <Py_ssize_t>b):
        sum_add(&s, v.data[i])
    return sum_result(&s)


cdef inline double vec_sum(F64Vec v) noexcept:
    cdef VecSum s
    cdef Py_ssize_t i
    sum_init(&s)
    for i in range(v.n):
        sum_add(&s, v.data[i])
    return sum_result(&s)


cdef inline long long ivec_sum_range(I64Vec v, long long a, long long b) except? -1:
    """sum(v[a:b]) of an integer vector; OverflowError where int64 would overflow."""
    cdef long long total = 0
    cdef Py_ssize_t i
    _clip(&a, &b, v.n)
    for i in range(<Py_ssize_t>a, <Py_ssize_t>b):
        if mx_vec_add_overflow(total, v.data[i], &total):
            return _ivec_sum_slow(v, <Py_ssize_t>a, <Py_ssize_t>b)
    return total


cdef inline long long ivec_sum(I64Vec v) except? -1:
    return ivec_sum_range(v, 0, v.n)


cdef extern from *:
    """
    #include <limits.h>
    /* exact signed add in portable C (no compiler builtin): the test runs before the add */
    static CYTHON_INLINE int mx_vec_add_overflow(long long a, long long b, long long *out) {
        if ((b > 0 && a > LLONG_MAX - b) || (b < 0 && a < LLONG_MIN - b))
            return 1;
        *out = a + b;
        return 0;
    }
    """
    bint mx_vec_add_overflow(long long a, long long b, long long* out) nogil
