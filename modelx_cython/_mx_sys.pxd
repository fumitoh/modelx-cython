from libc.math cimport (
    pow as _mx_c_pow, floor as _mx_c_floor, isfinite as _mx_c_isfinite)
from libc.limits cimport LLONG_MAX, LLONG_MIN
from libc.string cimport memcmp
from cpython.long cimport PyLong_AsLongLong, PyLong_Check, PyLong_CheckExact
from cpython.number cimport PyNumber_Index
from cpython.object cimport Py_TYPE, PyObject_GenericGetAttr, PyObject_GenericSetAttr
from cpython.unicode cimport PyUnicode_Check, PyUnicode_Compare
from modelx_cython._rounding cimport _mx_round_one, _mx_round_digits
cimport cython


cdef inline object _mx_named(object error, str path):
    """``error`` marked as naming the Cell ``path``."""
    error._mx_cell = path
    return error


cdef inline object _mx_named_overflow(object error, str path):
    """The OverflowError to raise from Cell ``path``: ``error`` itself when a Cell already named it,
    else a new one naming ``path``, caused by ``error``."""
    if getattr(error, '_mx_cell', None) is not None:
        return error
    named = _mx_named(OverflowError(f"{path}: {error} (integer arithmetic beyond 64 bits)"), path)
    named.__cause__ = error
    return named


cdef inline long long _mx_exact_int64(object value, str path) except? -1:
    """Cross a signed 64-bit boundary through the Python index protocol.

    An int or any int subclass (IntEnum, bool) is read directly: PyNumber_Index returns such an
    object unchanged, so the value and the OverflowError are the same, without taking and
    releasing a reference on it (diag-scale P3: enum members are shared by every worker)."""
    try:
        if PyLong_Check(value):
            return PyLong_AsLongLong(value)
        return PyLong_AsLongLong(PyNumber_Index(value))
    except TypeError as error:
        raise _mx_named(TypeError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
        ), path) from error
    except OverflowError as error:
        raise _mx_named(OverflowError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
        ), path) from error


ctypedef unsigned long _mx_native_ulong
ctypedef unsigned long long _mx_native_ulonglong

ctypedef fused _mx_integer_source:
    object
    long long
    _mx_native_ulong
    _mx_native_ulonglong
    double


cdef inline long long _mx_integer_result(_mx_integer_source value, str path) except? -1:
    """Check Python objects and other C scalars before storing a signed result."""
    if _mx_integer_source is object:
        return _mx_exact_int64(value, path)
    elif _mx_integer_source is double:
        raise _mx_named(TypeError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
        ), path)
    elif _mx_integer_source is _mx_native_ulong or _mx_integer_source is _mx_native_ulonglong:
        if value > LLONG_MAX:
            raise _mx_named(OverflowError(
                f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."
            ), path)
        return <long long>value
    else:
        return value


cdef extern from *:
    """
    /* Python's a // b on C integers with the branches Cython emits on LP64 platforms, on every
       platform: Cython 3.3.1 compiles its LLONG_MIN // -1 check only where long long is as wide as
       long, so the C division traps there (STATUS_INTEGER_OVERFLOW on Windows). The error value
       is LLONG_MAX (a quotient only of LLONG_MAX // 1), so a quotient of -1 needs no error check.
       The path is the Cell's, for the signature of the integer helpers; the checked mode's
       handler names the OverflowError after it. */
    static CYTHON_INLINE long long __mx_signed_floordiv(long long a, long long b, PyObject *path) {
        long long q, r;
        (void)path;
        if (b == 0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "integer division or modulo by zero");
            return LLONG_MAX;
        }
        if (b == -1 && a == LLONG_MIN) {
            PyErr_SetString(PyExc_OverflowError, "value too large to perform division");
            return LLONG_MAX;
        }
        q = a / b;
        r = a - q * b;
        return q - ((r != 0) & ((r ^ b) < 0));
    }
    """
    long long _mx_signed_floordiv "__mx_signed_floordiv"(long long left, long long right, str path) \
        except? 9223372036854775807


cdef extern from *:
    """
    /* Python's a % b on C integers: Cython's remainder (zero check, sign adaptation), and 0 for
       b == -1, where C's a % b traps (LLONG_MIN % -1) and Cython does not check it. A Python
       remainder is smaller than b in magnitude, so LLONG_MAX is never one: the error value. */
    static CYTHON_INLINE long long __mx_signed_remainder(long long a, long long b, PyObject *path) {
        long long r;
        (void)path;
        if (b == 0) {
            PyErr_SetString(PyExc_ZeroDivisionError, "integer division or modulo by zero");
            return LLONG_MAX;
        }
        if (b == -1)
            return 0;
        r = a % b;
        return r + ((r != 0) & ((r ^ b) < 0)) * b;
    }
    """
    long long _mx_signed_remainder "__mx_signed_remainder"(long long left, long long right, str path) \
        except 9223372036854775807


@cython.overflowcheck(True)
cdef inline long long _mx_signed_left_shift(long long value, long long count, str path) except? -1:
    """Python's ``value << count`` on C integers: a negative count raises ValueError, a count of
    64 or more is exact for 0 (and -1 << 63), and a result beyond 64 bits raises OverflowError
    naming the Cell. Cython's own checked shift raises for negative values (-3 << 2)."""
    if count < 0:
        raise ValueError("negative shift count")
    if count >= 63:
        if value == 0:
            return 0
        if count == 63 and value == -1:
            return LLONG_MIN
        raise _mx_named(OverflowError(f"{path}: value too large (integer arithmetic beyond 64 bits)"), path)
    try:
        return value * ((<long long>1) << count)
    except OverflowError as error:
        raise _mx_named_overflow(error, path)


@cython.overflowcheck(True)
cdef inline long long _mx_signed_pow(long long base, long long exponent, str path) except? -1:
    """``base ** exponent`` of two C integers, exactly or OverflowError (Cython's overflowcheck on
    the products; Cython's own C integer power wraps). A negative exponent has a float value in
    Python, which a C integer result cannot hold: TypeError naming the Cell. By squaring: each
    square taken is at most the result in magnitude (|base| >= 2), so no check fails for a power
    that fits."""
    cdef long long result = 1
    if exponent < 0:
        raise _mx_named(TypeError(
            f"{path} requires an exact signed 64-bit integer. Rebuild with a wider sample or spec."), path)
    try:
        while True:
            if exponent & 1:
                result = result * base
            exponent >>= 1
            if exponent == 0:
                return result
            base = base * base
    except OverflowError as error:
        raise _mx_named_overflow(error, path)


cdef inline long long _mx_signed_right_shift(long long value, long long count, str path) except? -1:
    if count < 0:
        raise ValueError("negative shift count")
    if count >= 63:
        return -1 if value < 0 else 0
    # Only shift nonnegative C values. Python rounds negative shifts down.
    if value < 0:
        return -1 - ((-1 - value) >> count)
    return value >> count


cdef inline bint _mx_ref_name_is(object name, str expected) except? -1:
    return PyUnicode_Check(name) and PyUnicode_Compare(name, expected) == 0


cdef inline int _mx_generic_setattr(object receiver, object name, object value) except -1:
    return PyObject_GenericSetAttr(receiver, name, value)


cdef inline object _mx_generic_getattr(object receiver, object name):
    return PyObject_GenericGetAttr(receiver, name)


cdef inline int _mx_check_enum_schema(object receiver, object owner, tuple bindings) except -1:
    """Declared model-code bindings are fixed until the model is rebuilt."""
    cdef object actual = <object>Py_TYPE(receiver)
    cdef object base, name
    if actual is owner:
        return 0
    for base in actual.__mro__:
        if base is owner:
            break
        for name in bindings:
            if name in base.__dict__:
                raise TypeError(f"enum definition {name!r} is fixed in this compiled model; rebuild the model to change it")
    return 0


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


cdef struct PowerMemo:
    bint ready
    double base
    double exponent
    double value


cdef inline double _mx_memo_pow(PowerMemo* memo, double base, double exponent) except? -1.0:
    cdef double value
    if (memo.ready and memcmp(&memo.base, &base, sizeof(double)) == 0
            and memcmp(&memo.exponent, &exponent, sizeof(double)) == 0):
        return memo.value
    value = _mx_pow(base, exponent)
    if _mx_c_isfinite(base) and _mx_c_isfinite(exponent):
        memo.base = base
        memo.exponent = exponent
        memo.value = value
        memo.ready = True
    return value


@cython.cdivision(True)
cdef inline bint _mx_divisible(long long dividend, long long divisor) except? -1:
    if divisor == 0:
        raise ZeroDivisionError("integer division or modulo by zero")
    return divisor == -1 or dividend % divisor == 0


cdef inline bint _mx_divisible_after_decrement(long long dividend, long long divisor) except? -1:
    if dividend == LLONG_MIN:
        return ((<object>dividend) - 1) % divisor == 0
    return _mx_divisible(dividend - 1, divisor)


cdef inline bint _mx_divisible_after_increment(long long dividend, long long divisor) except? -1:
    if dividend == LLONG_MAX:
        return ((<object>dividend) + 1) % divisor == 0
    return _mx_divisible(dividend + 1, divisor)


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


cdef extern from *:
    """
    /* The short readiness flags of a dense cache of a shared Space. Workers fill the
       cache concurrently: the value is stored, then the flag after a release fence;
       a reader loads the flag, then an acquire fence orders its read of the value. A
       relaxed 16-bit access and the fences are CPython's (pyatomic.h, 3.13 and later);
       a build with the GIL keeps plain accesses. On x86 the fences emit no instruction. */
    static inline short __mx_load_dense_flag(short *p) {
    #if defined(Py_GIL_DISABLED)
        short flag = (short)_Py_atomic_load_int16_relaxed((int16_t *)p);
        _Py_atomic_fence_acquire();
        return flag;
    #else
        return *p;
    #endif
    }
    static inline void __mx_store_dense_flag(short *p, short v) {
    #if defined(Py_GIL_DISABLED)
        _Py_atomic_fence_release();
        _Py_atomic_store_int16_relaxed((int16_t *)p, (int16_t)v);
    #else
        *p = v;
    #endif
    }
    """
    short _mx_load_dense_flag "__mx_load_dense_flag" (short *p) noexcept nogil
    void _mx_store_dense_flag "__mx_store_dense_flag" (short *p, short v) noexcept nogil

cdef class BaseMxObject:
    pass

cdef class BaseParent(BaseMxObject):

    cdef public dict  _mx_spaces
    cdef public BaseParent _parent
    cdef public BaseModel _model
    cdef public str _name
    # The reentrant lock shared by the locked Spaces of the model, or None.
    # Declared on the common base of the model and of the Spaces: a locked
    # Space reads it through its typed _model field and through itself.
    cdef public object _mx_lock

cdef class BaseModel(BaseParent):
    cdef public object _mx_collect_workers_override
    cdef public object _mx_collect_dispatch_state

cdef class BaseSpace(BaseParent):

    cdef bint _mx_is_cells_set
    cdef dict _mx_cells

    cdef BaseSpace _space
    cdef dict _mx_itemspaces
    cdef public list _mx_roots
