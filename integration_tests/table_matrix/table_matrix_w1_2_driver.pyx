# cython: language_level=3, freethreading_compatible=True
# W1.2 matrix driver: typed handle reads in the shape W1.3's lowering emits
# (`h.f64(h.row_n(k...), h.col("c"))` with the column resolved first, as pandas raises a
# missing column before a missing key). One call per read; `cint` passes every key part as a C
# long long (int and enum keys in compiled code), else as objects.
from {PROVIDER} cimport TableHandle, RecordHandle, table_handle, record_handle
from {PROVIDER} import handle_fallbacks


cdef Py_ssize_t _row(TableHandle h, tuple p, bint cint) except -2:
    cdef Py_ssize_t n = len(p)
    cdef long long a, b, c, d, e, f
    if cint:
        a = p[0]
        if n == 1:
            return h.row_1(a)
        b = p[1]
        if n == 2:
            return h.row_2(a, b)
        c = p[2]
        if n == 3:
            return h.row_3(a, b, c)
        d = p[3]
        if n == 4:
            return h.row_4(a, b, c, d)
        e = p[4]
        if n == 5:
            return h.row_5(a, b, c, d, e)
        f = p[5]
        return h.row_6(a, b, c, d, e, f)
    if n == 1:
        return h.row_1(<object>p[0])
    if n == 2:
        return h.row_2(<object>p[0], <object>p[1])
    if n == 3:
        return h.row_3(<object>p[0], <object>p[1], <object>p[2])
    if n == 4:
        return h.row_4(<object>p[0], <object>p[1], <object>p[2], <object>p[3])
    if n == 5:
        return h.row_5(<object>p[0], <object>p[1], <object>p[2], <object>p[3], <object>p[4])
    return h.row_6(<object>p[0], <object>p[1], <object>p[2], <object>p[3], <object>p[4], <object>p[5])


cdef bint _has(TableHandle h, tuple p, bint cint) except -1:
    cdef Py_ssize_t n = len(p)
    cdef long long a, b, c, d, e, f
    if cint:
        a = p[0]
        if n == 1:
            return h.has_1(a)
        b = p[1]
        if n == 2:
            return h.has_2(a, b)
        c = p[2]
        if n == 3:
            return h.has_3(a, b, c)
        d = p[3]
        if n == 4:
            return h.has_4(a, b, c, d)
        e = p[4]
        if n == 5:
            return h.has_5(a, b, c, d, e)
        f = p[5]
        return h.has_6(a, b, c, d, e, f)
    if n == 1:
        return h.has_1(<object>p[0])
    if n == 2:
        return h.has_2(<object>p[0], <object>p[1])
    if n == 3:
        return h.has_3(<object>p[0], <object>p[1], <object>p[2])
    if n == 4:
        return h.has_4(<object>p[0], <object>p[1], <object>p[2], <object>p[3])
    if n == 5:
        return h.has_5(<object>p[0], <object>p[1], <object>p[2], <object>p[3], <object>p[4])
    return h.has_6(<object>p[0], <object>p[1], <object>p[2], <object>p[3], <object>p[4], <object>p[5])


cdef TableHandle _sub(TableHandle h, tuple p, bint cint):
    cdef Py_ssize_t n = len(p)
    cdef long long a, b, c, d, e
    if cint:
        a = p[0]
        if n == 1:
            return h.sub_1(a)
        b = p[1]
        if n == 2:
            return h.sub_2(a, b)
        c = p[2]
        if n == 3:
            return h.sub_3(a, b, c)
        d = p[3]
        if n == 4:
            return h.sub_4(a, b, c, d)
        e = p[4]
        return h.sub_5(a, b, c, d, e)
    if n == 1:
        return h.sub_1(<object>p[0])
    if n == 2:
        return h.sub_2(<object>p[0], <object>p[1])
    if n == 3:
        return h.sub_3(<object>p[0], <object>p[1], <object>p[2])
    if n == 4:
        return h.sub_4(<object>p[0], <object>p[1], <object>p[2], <object>p[3])
    return h.sub_5(<object>p[0], <object>p[1], <object>p[2], <object>p[3], <object>p[4])


def handle(t):
    return table_handle(t)


def loc(t, tuple parts, str column, bint cint):          # float(T.loc[k, "c"]) / T.at[k, "c"]
    cdef TableHandle h = table_handle(t)
    cdef Py_ssize_t c = h.col(column)
    return h.f64(_row(h, parts, cint), c)


def loc_int(t, tuple parts, str column, bint cint):      # int(T.loc[k, "c"])
    cdef TableHandle h = table_handle(t)
    cdef Py_ssize_t c = h.col(column)
    return h.i64(_row(h, parts, cint), c)


def loc_obj(t, tuple parts, str column, bint cint):      # T.loc[k, "c"] kept as an object
    cdef TableHandle h = table_handle(t)
    cdef Py_ssize_t c = h.col(column)
    return h.obj(_row(h, parts, cint), c)


def contains(t, tuple parts, bint cint):                 # k in T.index
    return _has(table_handle(t), parts, cint)


def sub(t, tuple parts, bint cint):                      # sub = T.loc[prefix]
    return _sub(table_handle(t), parts, cint)


def first_last(t):                                       # T.index.min(), T.index.max()
    cdef TableHandle h = table_handle(t)
    return h.first_int(), h.last_int()


def first_last_key(t):
    cdef TableHandle h = table_handle(t)
    return h.first_key(), h.last_key()


def clamp(t, long long x):                               # min(max(x, T.index.min()), T.index.max())
    cdef TableHandle h = table_handle(t)
    return min(max(x, h.first_int()), h.last_int())


def clamp_read(t, long long x, str column):              # T.loc[min(max(x, lo), hi), "c"]
    cdef TableHandle h = table_handle(t)
    cdef Py_ssize_t c = h.col(column)
    cdef long long k = min(max(x, h.first_int()), h.last_int())
    return h.f64(h.row_1(k), c)


def asof(t, long long x):                                # T.index.asof(x)
    return table_handle(t).asof_int(x)


def asof_read(t, long long x, str column):               # T.loc[T.index.asof(x), "c"]
    cdef TableHandle h = table_handle(t)
    cdef Py_ssize_t c = h.col(column)
    return h.f64(h.asof_row(x), c)


def asof_arith(t, long long x):                          # T.index.asof(x) * 2 + 1
    cdef TableHandle h = table_handle(t)
    return h.asof_int(x) * 2 + 1


def searchsorted(t, long long x):                        # T.index.searchsorted(x, side="right")
    return table_handle(t).searchsorted_right(x)


def key_at(t, Py_ssize_t i):                             # T.index[i]
    return table_handle(t).key_int(i)


def successor(t, long long x):                           # T.index[T.index.searchsorted(x, side="right")]
    cdef TableHandle h = table_handle(t)
    return h.key_int(h.searchsorted_right(x))


def row_reads(t, tuple first, tuple second, str column, bint cint):
    # row = T.loc[k1]; v = row["c"]; row = T.loc[k2]; return v + row["c"]   (row locals: rows)
    cdef TableHandle h = table_handle(t)
    cdef Py_ssize_t row = _row(h, first, cint)
    cdef double value = h.f64(row, h.col(column))
    row = _row(h, second, cint)
    return value + h.f64(row, h.col(column))


def row_record(t, tuple parts, bint cint):               # a row local that escapes
    cdef TableHandle h = table_handle(t)
    return h.row_record(_row(h, parts, cint))


def column(t, str name, tuple parts, bint cint):         # T["c"][k]
    cdef TableHandle h = table_handle(t)
    cdef TableHandle view = h.column_view(h.col(name))
    return view.f64(_row(view, parts, cint), 0)


def field(record, str name):                             # model_point()["x"] as a float
    cdef RecordHandle r = record_handle(record)
    return r.f64(r.field(name))


def field_int(record, str name):
    cdef RecordHandle r = record_handle(record)
    return r.i64(r.field(name))


def field_obj(record, str name):
    cdef RecordHandle r = record_handle(record)
    return r.obj(r.field(name))


def field_sum(record, str a, str b):                     # model_point()["a"] * 2 + model_point()["b"]
    cdef RecordHandle r = record_handle(record)
    return r.f64(r.field(a)) * 2 + r.f64(r.field(b))


def spin(t, long long n, long long lo, long long hi, str column):
    """n reads of the handle (thread check): sum of T.loc[lo + i % (hi - lo + 1), "c"]."""
    cdef TableHandle h = table_handle(t)
    cdef Py_ssize_t c = h.col(column)
    cdef long long i, span = hi - lo + 1
    cdef double total = 0
    for i in range(n):
        total += h.f64(h.row_1(lo + i % span), c)
    return total


def spin2(t, long long n, long long a, long long lo, long long hi, str column):
    """n two-level reads through a sub-table: sum of T.loc[a].loc[lo + i % span, "c"]."""
    cdef TableHandle h = table_handle(t)
    cdef TableHandle s
    cdef Py_ssize_t c = h.col(column)
    cdef long long i, span = hi - lo + 1
    cdef double total = 0
    for i in range(n):
        s = h.sub_1(a)
        total += s.f64(s.row_1(lo + i % span), c) + h.f64(h.row_2(a, lo + i % span), c)
    return total


def sub_object(t, tuple parts, bint cint):               # return T.loc[prefix] (a table-valued cell, D45)
    return _sub(table_handle(t), parts, cint).table_object()
