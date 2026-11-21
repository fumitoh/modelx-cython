# cython: freethreading_compatible=True, boundscheck=False, wraparound=False
"""Typed vectors of vector cells; see _native_vectors.pxd."""
from cpython.float cimport PyFloat_AS_DOUBLE
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free, PyMem_Calloc
from cpython.ref cimport Py_INCREF, Py_XDECREF
import operator

cdef extern from "Python.h":
    Py_ssize_t PyList_GET_SIZE(object)
    PyObject* PyList_GET_ITEM(object, Py_ssize_t)
    Py_ssize_t PyTuple_GET_SIZE(object)
    PyObject* PyTuple_GET_ITEM(object, Py_ssize_t)

cdef double _TWO53 = 9007199254740992.0


# ---------------------------------------------------------------- F64Vec
cdef class F64Vec:
    """Floats in a C double buffer: the cached value of a float vector cell.

    Compiled code reads and writes the buffer through the functions below; it is immutable once
    the formula that built it has returned, so threads read it without a lock. Python never sees
    it (D154): a cell's public method and every use in Python code get ``tolist()``, a new list.
    """

    __hash__ = None

    def __dealloc__(self):
        PyMem_Free(self.data)

    def tolist(self):
        cdef Py_ssize_t i
        return [self.data[i] for i in range(self.n)]

    def __repr__(self):
        return 'F64Vec(%r)' % (self.tolist(),)


cdef F64Vec vec_new(Py_ssize_t n):
    cdef F64Vec v = F64Vec.__new__(F64Vec)
    if n < 0:
        n = 0
    v.data = <double*>PyMem_Malloc((n if n > 0 else 1) * sizeof(double))
    if v.data == NULL:
        raise MemoryError()
    v.n = n
    v.cap = n if n > 0 else 1
    return v


cdef F64Vec vec_full(Py_ssize_t n, double value):
    cdef F64Vec v = vec_new(n)
    cdef Py_ssize_t i
    for i in range(v.n):
        v.data[i] = value
    return v


cdef int _vec_grow(F64Vec v) except -1:
    cdef Py_ssize_t cap = v.cap * 2 if v.cap > 0 else 8
    cdef double* data = <double*>PyMem_Realloc(v.data, cap * sizeof(double))
    if data == NULL:
        raise MemoryError()
    v.data = data
    v.cap = cap
    return 0


cdef inline double _exact_float(object item) except? -1.0:
    """A float item, or an int that a double holds exactly (a list built from ints
    on a path the trace did not see); anything else is a TypeError."""
    cdef double x
    if type(item) is float:
        return PyFloat_AS_DOUBLE(item)
    if isinstance(item, float):
        return <double>item
    if isinstance(item, int) and not isinstance(item, bool):
        x = <double>item
        if -_TWO53 <= x <= _TWO53 and int(x) == item:
            return x
        raise TypeError(f'a float vector cannot hold the integer {item!r} exactly')
    raise TypeError(f'a float vector holds floats, not {type(item).__name__}')


cdef F64Vec vec_of(object value):
    """The value itself if it is an F64Vec, else a copy of a sequence of floats."""
    cdef F64Vec v
    cdef Py_ssize_t i, n
    if type(value) is F64Vec:
        return <F64Vec>value
    if value is None:
        raise TypeError("a float vector cell or local holds None, not a list of floats")
    if type(value) is I64Vec:
        value = list(value)
    if type(value) is list:
        n = len(value)
        v = vec_new(n)
        for i in range(n):
            v.data[i] = _exact_float(<object>PyList_GET_ITEM(value, i))
        return v
    if type(value) is not tuple:
        value = tuple(value)
    n = len(value)
    v = vec_new(n)
    for i in range(n):
        v.data[i] = _exact_float(<object>PyTuple_GET_ITEM(value, i))
    return v


cdef F64Vec vec_slice(F64Vec v, long long a, long long b):
    """v[a:b] as a new vector (Python slice normalisation, step 1)."""
    cdef F64Vec r
    cdef Py_ssize_t i
    _clip(&a, &b, v.n)
    r = vec_new(b - a if b > a else 0)
    for i in range(r.n):
        r.data[i] = v.data[a + i]
    return r


cdef double _vec_get_slow(F64Vec v, long long i) except? -1.0:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        raise IndexError("list index out of range")
    return v.data[i]


cdef int _vec_set_slow(F64Vec v, long long i, double value) except -1:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        raise IndexError("list assignment index out of range")
    v.data[i] = value
    return 0


# ---------------------------------------------------------------- I64Vec
cdef class I64Vec:
    """Integers (int64) in a C buffer: the cached value of an integer vector cell (see F64Vec)."""

    __hash__ = None

    def __dealloc__(self):
        PyMem_Free(self.data)

    def tolist(self):
        cdef Py_ssize_t i
        return [self.data[i] for i in range(self.n)]

    def __repr__(self):
        return 'I64Vec(%r)' % (self.tolist(),)


cdef I64Vec ivec_new(Py_ssize_t n):
    cdef I64Vec v = I64Vec.__new__(I64Vec)
    if n < 0:
        n = 0
    v.data = <long long*>PyMem_Malloc((n if n > 0 else 1) * sizeof(long long))
    if v.data == NULL:
        raise MemoryError()
    v.n = n
    v.cap = n if n > 0 else 1
    return v


cdef I64Vec ivec_full(Py_ssize_t n, long long value):
    cdef I64Vec v = ivec_new(n)
    cdef Py_ssize_t i
    for i in range(v.n):
        v.data[i] = value
    return v


cdef int _ivec_grow(I64Vec v) except -1:
    cdef Py_ssize_t cap = v.cap * 2 if v.cap > 0 else 8
    cdef long long* data = <long long*>PyMem_Realloc(v.data, cap * sizeof(long long))
    if data == NULL:
        raise MemoryError()
    v.data = data
    v.cap = cap
    return 0


cdef inline long long _exact_int(object item) except? -1:
    if type(item) is int:
        return item
    if isinstance(item, int) and not isinstance(item, bool):
        return operator.index(item)
    raise TypeError(f'an integer vector holds integers, not {type(item).__name__}')


cdef I64Vec ivec_of(object value):
    """The value itself if it is an I64Vec, else a copy of a sequence of integers."""
    cdef I64Vec v
    cdef Py_ssize_t i, n
    if type(value) is I64Vec:
        return <I64Vec>value
    if value is None:
        raise TypeError("an integer vector cell or local holds None, not a list of integers")
    if type(value) is not list and type(value) is not tuple:
        value = list(value)
    n = len(value)
    v = ivec_new(n)
    for i in range(n):
        v.data[i] = _exact_int(value[i])
    return v


cdef I64Vec ivec_slice(I64Vec v, long long a, long long b):
    cdef I64Vec r
    cdef Py_ssize_t i
    _clip(&a, &b, v.n)
    r = ivec_new(b - a if b > a else 0)
    for i in range(r.n):
        r.data[i] = v.data[a + i]
    return r


cdef long long _ivec_sum_slow(I64Vec v, Py_ssize_t a, Py_ssize_t b) except? -1:
    total = sum(v.data[i] for i in range(a, b))
    return total     # OverflowError when it does not fit int64


cdef long long _ivec_get_slow(I64Vec v, long long i) except? -1:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        raise IndexError("list index out of range")
    return v.data[i]


cdef int _ivec_set_slow(I64Vec v, long long i, long long value) except -1:
    if i < 0:
        i += v.n
    if i < 0 or i >= v.n:
        raise IndexError("list assignment index out of range")
    v.data[i] = value
    return 0


# ---------------------------------------------------------------- dense cache of vector cells
cdef class VecCache:

    def __cinit__(self, Py_ssize_t n, long long lo):
        self.slots = <PyObject**>PyMem_Calloc(n if n > 0 else 1, sizeof(PyObject*))
        if self.slots == NULL:
            raise MemoryError()
        self.n = n
        self.lo = lo

    def __dealloc__(self):
        cdef Py_ssize_t i
        if self.slots != NULL:
            for i in range(self.n):
                Py_XDECREF(self.slots[i])
            PyMem_Free(self.slots)

    cdef object put(self, long long key, object value):
        """Publish value at key unless another thread did first; returns the published one."""
        cdef unsigned long long i = <unsigned long long>(key - self.lo)
        cdef PyObject* p
        if i >= <unsigned long long>self.n:
            return value        # outside the sampled domain: not cached
        Py_INCREF(value)
        if __pyx_vec_publish(&self.slots[i], <PyObject*>value):
            return value
        Py_XDECREF(<PyObject*>value)
        p = __pyx_vec_load(&self.slots[i])
        return <object>p

    def __len__(self):
        cdef Py_ssize_t i, k = 0
        for i in range(self.n):
            if __pyx_vec_load(&self.slots[i]) != NULL:
                k += 1
        return k

    def items(self):
        """(key, value) of the filled slots."""
        cdef Py_ssize_t i
        cdef PyObject* p
        out = []
        for i in range(self.n):
            p = __pyx_vec_load(&self.slots[i])
            if p != NULL:
                out.append((self.lo + i, <object>p))
        return out

    def __reduce__(self):
        return (_vec_cache_from_items, (self.n, self.lo, self.items()))


def _vec_cache_from_items(n, lo, items):
    """Unpickling: a VecCache holding the given (key, value) pairs."""
    cdef VecCache cache = VecCache(n, lo)
    for key, value in items:
        cache.put(key, value)
    return cache
