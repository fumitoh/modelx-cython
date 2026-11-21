from modelx_cython._float_sum_state cimport F64Sum, f64_init, f64_add, f64_result
# W1.4: the exact built-in sum state used by lowered sum(...) (see reductions.GeneralReductions)
from modelx_cython._float_sum_state cimport (
    PySum, sum_init, sum_add_f64, sum_add_ll,
    sum_add_bint, sum_add_object, sum_result, sum_result_f64, sum_clear)
cdef bint f64_accepts(object function)


# diag-scale P3: an integer item of an exact tuple read borrowed (no reference count change on the
# item: a tuple of IntEnum members holds objects every worker shares). True and *out set when the
# item is an int (any int subclass, as PyNumber_Index returns it unchanged) that fits in a signed
# 64-bit integer; False otherwise, with no error set, so the caller takes the object path, which
# raises exactly what it raises today. `t` must be an exact tuple and `i` in range.
cdef extern from *:
    """
    static CYTHON_INLINE int __pyx_mx_tuple_int64(PyObject *t, Py_ssize_t i, long long *out) {
        PyObject *item = PyTuple_GET_ITEM(t, i);
        int overflow = 0;
        long long value;
        if (!PyLong_Check(item)) return 0;
        value = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow || (value == -1 && PyErr_Occurred())) { PyErr_Clear(); return 0; }
        *out = value;
        return 1;
    }
    """
    bint tuple_int64 "__pyx_mx_tuple_int64" (object t, Py_ssize_t i, long long *out) noexcept
    Py_ssize_t tuple_size "PyTuple_GET_SIZE" (object t) noexcept
    bint exact_tuple "PyTuple_CheckExact" (object o) noexcept
