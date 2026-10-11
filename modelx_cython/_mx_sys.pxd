from libc.math cimport (
    pow as _mx_c_pow, floor as _mx_c_floor, isfinite as _mx_c_isfinite,
    exp as _mx_c_exp, log as _mx_c_log)


cdef inline double _mx_pow(double b, double e) except? -1.0:
    """``**`` between operands mx2cy has typed as C numbers.

    Cython would otherwise evaluate a fractional power on
    ``double complex``, which is slower and does not compile at all
    where the result is compared.  Only powers whose exponent is
    provably floating are rewritten to this, so integer arithmetic is
    unaffected; see :mod:`modelx_cython.powers`.

    Raises rather than returning the ``nan`` or ``inf`` C ``pow``
    gives for the two inputs a model should not have produced: a
    negative base under a fractional exponent, where Python returns a
    complex number the compiled model has no type for, and ``0.0`` to
    a negative power, where Python itself raises.  Both guards are
    limited to finite operands, because CPython disposes of an
    infinite base or exponent before either case applies and C ``pow``
    already agrees with it there.  Other divergences from Python's
    ``float.__pow__`` are left alone, notably overflow, on which
    Python raises and every C double in the generated model saturates.
    """
    if _mx_c_isfinite(b) and _mx_c_isfinite(e):
        if b < 0.0 and e != _mx_c_floor(e):
            raise ValueError(
                "a negative number cannot be raised to a fractional power")
        if b == 0.0 and e < 0.0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
    return _mx_c_pow(b, e)


# math.exp, math.log and math.pow on C doubles, through the C library.
# Used only where the spec sets "use_libm" (see modelx_cython.powers).
# Where the inputs and the C result are all finite, the C result is
# returned: bit for bit what CPython's math module returns, as measured
# with MSVC and as follows where CPython calls the same C functions.
# Every other case -- an infinite or nan input, and the overflow, pole
# and domain errors where C returns inf or nan and Python raises
# OverflowError or ValueError -- is handed to Python's math module, so
# that it returns or raises exactly what Python does.

cdef inline double _mx_exp(double x) except? -1.0:
    """``math.exp(x)``; see above."""
    cdef double r = _mx_c_exp(x)
    if _mx_c_isfinite(x) and _mx_c_isfinite(r):
        return r
    import math
    return math.exp(x)


cdef inline double _mx_log(double x) except? -1.0:
    """``math.log(x)`` with one argument; see above."""
    cdef double r = _mx_c_log(x)
    if _mx_c_isfinite(x) and _mx_c_isfinite(r):
        return r
    import math
    return math.log(x)


cdef inline double _mx_math_pow(double x, double y) except? -1.0:
    """``math.pow(x, y)``, which is not ``x ** y``; see above."""
    cdef double r = _mx_c_pow(x, y)
    if _mx_c_isfinite(x) and _mx_c_isfinite(y) and _mx_c_isfinite(r):
        return r
    import math
    return math.pow(x, y)


cdef extern from *:
    """
    /* Acquire/release accessors for the _has_ flags of locked Spaces.
       The writer stores the cached value, then the flag with release
       semantics, under the model lock; a reader that sees the flag with
       acquire semantics is therefore guaranteed to see the value, without
       taking the lock. pyatomic.h is included by Python.h from CPython 3.13,
       the first version with a free-threaded build; a build with the GIL
       needs no ordering, and neither does an older version of Python. */
    #if defined(Py_GIL_DISABLED)
    #define __MX_LOAD_FLAG(p)      _Py_atomic_load_int_acquire((const int *)(p))
    #define __MX_STORE_FLAG(p, v)  _Py_atomic_store_int_release((int *)(p), (v))
    #else
    #define __MX_LOAD_FLAG(p)      (*(p))
    #define __MX_STORE_FLAG(p, v)  ((void)(*(p) = (v)))
    #endif
    """
    bint _mx_load_flag "__MX_LOAD_FLAG" (bint *p) noexcept nogil
    void _mx_store_flag "__MX_STORE_FLAG" (bint *p, bint v) noexcept nogil

cdef class BaseMxObject:
    pass

cdef class BaseParent(BaseMxObject):

    cdef public dict  _mx_spaces
    cdef public BaseParent _parent
    cdef public BaseModel _model
    cdef public str _name
    # The threading.RLock shared by the locked Spaces of the model, or None.
    # Declared on the common base of the model and of the Spaces: a locked
    # Space reads it through its typed _model field and through itself.
    cdef public object _mx_lock

cdef class BaseModel(BaseParent):
    pass

cdef class BaseSpace(BaseParent):

    cdef bint _mx_is_cells_set
    cdef dict _mx_cells

    cdef BaseSpace _space
    cdef dict _mx_itemspaces
    cdef public list _mx_roots
