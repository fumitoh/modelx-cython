cimport cython
from libc.math cimport fabs
from modelx_cython._float_sum_state cimport F64Sum, f64_init, f64_add, f64_result

cdef Py_ssize_t indexed_position(object table, object key) except? -2

cdef inline double f64_abs(double value) noexcept nogil:
    return fabs(value)

cdef bint f64_accepts(object function)


# Packed keys borrow object components from the key owner during lookup.
from cpython.ref cimport PyObject
cdef struct KeyPart:
    int kind
    long long number
    PyObject* text
cdef void key_int(KeyPart*, long long, bint) noexcept
cdef void key_text(KeyPart*, object) noexcept
ctypedef long long KeyInteger
ctypedef fused KeyArg0:
    KeyInteger
    bint
    object

# ---- M1 typed table handles (interface fixed by W1.1; bodies W1.2) --------------------
# K = TableKey: a key part is a C long long (int, enum member, bool) or an object (str, other).
# n = 1..6 key parts. Errors are the original pandas errors (KeyError(original key) etc.).
ctypedef fused TableKey:
    long long
    object

cdef enum TableScheme:
    TABLE_DENSE = 1     # int range, row = key - base
    TABLE_SORTED = 2    # sorted int keys
    TABLE_HASHED = 3    # packed KeyPart hash index (KeyedView scheme)
    TABLE_PRODUCT = 4   # rectangular: one index per axis

# Private handle storage (W1.2). A key part normalised for comparison: kinds
# 0 never matches, 1 int (bools too, flagged), 2 non-integral float, 3 str, 4 missing (NaN/None
# under a pandas index spec), 5 any other object (compared with ==, counted as a fallback).
cdef struct HandlePart:
    unsigned char kind
    unsigned char boolean
    long long i
    double f
    PyObject* o             # borrowed: the handle keeps the label alive
    Py_hash_t h

# A key index over `count` keys of `nparts` parts each: DENSE (ints first..last in order),
# SORTED (ascending ints, row of each) or HASHED (open addressing over stored parts).
cdef struct HandleKeys:
    int scheme
    Py_ssize_t count, nparts, pstride, poff
    long long first, last
    const long long* skeys
    const Py_ssize_t* srows
    const HandlePart* parts     # key r, part j: parts[r * pstride + poff + j]
    const Py_ssize_t* slots     # row + 1, 0 empty
    size_t mask
    const unsigned char* modes  # pandas boolean label mode per part: 0 any, 1 bool only, 2 no bool
    # D86 (diag-vec): a dense product index over a HASHED multi-part key whose parts are all plain ints
    # in a small box: grid[row-major position] = row + 1 (0 absent); NULL when not built (stays hashed)
    const Py_ssize_t* grid
    const long long* gfirst     # per part: smallest key
    const unsigned long long* gspan     # per part: largest - smallest + 1

@cython.final
cdef class TableHandle:
    # Immutable after __init__; built once per bound table, children eagerly (D25 Q2).
    cdef readonly object owner      # helper Table (Python fallbacks, pickling)
    cdef int scheme
    cdef Py_ssize_t rows, width, levels, base   # base/rows: a sub-table views its parent's arrays
    # private (W1.2): everything below is set by the constructor and never written afterwards
    cdef object _hold               # buffers, part arrays, labels: kept alive for the C pointers
    cdef tuple _prefix              # prefix from the root table (() for a root)
    cdef tuple _columns
    cdef tuple _children            # depth-1 sub-tables, by axis position (PRODUCT) or child slot
    cdef tuple _labels              # row labels (row forms), for row_record and errors
    cdef object _first_obj, _last_obj   # index.min()/max() as objects; _no_extreme when none
    cdef object _array              # contiguous ndarray when cells need Python indexing
    cdef HandleKeys _keys           # full-key index (DENSE, SORTED, HASHED)
    cdef HandleKeys _ckeys          # HASHED: first part of each child -> child slot
    cdef HandleKeys* _axes          # PRODUCT: one index per remaining level
    cdef const Py_ssize_t* _spans   # PRODUCT: rows per position of each remaining level
    cdef const unsigned char* _ckind    # per root column: 1 float, 2 int, 3 bool, 4 object
    cdef const double* _dbl         # root arrays, row-major: (base + row) * _stride + _colbase + col
    cdef const long long* _int
    cdef PyObject** _obj
    cdef Py_ssize_t _stride, _colbase
    cdef bint _spec_missing, _int_keys, _increasing, _preserve
    cdef const long long* _okeys    # single-level int keys in row order
    cdef object _table_obj          # table_object(): built with the handle, None for column views
    cdef object _key_index          # D86: 'grid', or why a multi-level key stays hashed; None otherwise
    cdef tuple _rowrefs             # D132: the helper's own row tuples in handle row order (root rows form), else None
    cdef object _frame_owner        # D132: the owner when it is a frame-built table (its rows remember the frame row)
    cdef Py_ssize_t col(self, object name) except -2               # identity, then ==; KeyError(name)
    cdef Py_ssize_t row_1(self, TableKey p0) except -2              # KeyError(original key)
    cdef Py_ssize_t row_2(self, TableKey p0, TableKey p1) except -2
    cdef Py_ssize_t row_3(self, TableKey p0, TableKey p1, TableKey p2) except -2
    cdef Py_ssize_t row_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except -2
    cdef Py_ssize_t row_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3,
                          TableKey p4) except -2
    cdef Py_ssize_t row_6(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3,
                          TableKey p4, TableKey p5) except -2
    cdef bint has_1(self, TableKey p0) except -1
    cdef bint has_2(self, TableKey p0, TableKey p1) except -1
    cdef bint has_3(self, TableKey p0, TableKey p1, TableKey p2) except -1
    cdef bint has_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except -1
    cdef bint has_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4) except -1
    cdef bint has_6(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4,
                    TableKey p5) except -1
    cdef double f64(self, Py_ssize_t row, Py_ssize_t col) except? -1     # float column
    cdef long long i64(self, Py_ssize_t row, Py_ssize_t col) except? -1  # int / bool / enum column
    cdef object obj(self, Py_ssize_t row, Py_ssize_t col)                # str and other columns
    cdef long long first_int(self) except? -1                    # index.min()
    cdef long long last_int(self) except? -1                     # index.max()
    cdef object first_key(self)                                  # non-integer keys
    cdef object last_key(self)
    cdef long long first_part_int(self, Py_ssize_t k) except? -1   # index.min()[k], integer level k
    cdef long long last_part_int(self, Py_ssize_t k) except? -1    # index.max()[k]
    cdef long long asof_int(self, long long k) except? -1        # KeyError(k) below the first key (D25a)
    cdef Py_ssize_t asof_row(self, long long k) except -2
    cdef Py_ssize_t searchsorted_right(self, long long k) noexcept  # D25b
    cdef long long key_int(self, Py_ssize_t i) except? -1        # sub.index[i]; IndexError, negative i
    cdef TableHandle sub_1(self, TableKey p0)                     # n < levels; shared child, KeyError
    cdef TableHandle sub_2(self, TableKey p0, TableKey p1)
    cdef TableHandle sub_3(self, TableKey p0, TableKey p1, TableKey p2)
    cdef TableHandle sub_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3)
    cdef TableHandle sub_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4)
    cdef object row_record(self, Py_ssize_t row)                 # row local that escapes
    cdef TableHandle column_view(self, Py_ssize_t col)           # D13 column selection T["c"]
    cdef object table_object(self)                               # D45: the bound NativeTable of this handle

@cython.final
cdef class RecordHandle:
    # A Record (model_point()), fields resolved once; f64/i64/obj by slot.
    cdef readonly object owner
    cdef tuple _items               # private (W1.2): the Record's own (name, value) pairs
    cdef Py_ssize_t field(self, object name) except -2
    cdef double f64(self, Py_ssize_t slot) except? -1
    cdef long long i64(self, Py_ssize_t slot) except? -1
    cdef object obj(self, Py_ssize_t slot)

cdef TableHandle table_handle(object value)      # NativeTable.handle or a new handle
cdef RecordHandle record_handle(object value)

# W1.9: borrowed sub-tables and index labels (module functions; nothing is written).
# A TableRef is a borrowed pointer to a shared child handle: the parent handle (held by the
# caller for the whole formula) keeps it alive, so reads through it take no reference;
# use it as <TableHandle>ref. sub_ref_n raises KeyError as sub_n.
ctypedef PyObject* TableRef


# diag-jp (D75): the column index of a literal column name, resolved on first use and held in a
# per-instance slot (index + 1; 0 unresolved) next to the held handle, so a per-step read does not
# look the name up. `ref` is the held handle, borrowed. KeyError(name) as col() when absent (the slot
# stays 0, so the next read raises again, as before).
cdef inline Py_ssize_t held_col(Py_ssize_t* slot, TableRef ref, object name) except -2:
    if slot[0] == 0:
        slot[0] = (<TableHandle>ref).col(name) + 1
    return slot[0] - 1


# D79 (diag-vec): the row of a complete key whose parts are build-time integer constants, resolved
# on first use and held in a per-instance slot (row + 1; 0 unresolved) next to the held handle, as
# held_col. KeyError as row_n when absent (the slot stays 0, so the next read raises again).
cdef inline Py_ssize_t held_row_1(Py_ssize_t* slot, TableRef ref, long long p0) except -2:
    if slot[0] == 0:
        slot[0] = (<TableHandle>ref).row_1(p0) + 1
    return slot[0] - 1
cdef inline Py_ssize_t held_row_2(Py_ssize_t* slot, TableRef ref, long long p0, long long p1) except -2:
    if slot[0] == 0:
        slot[0] = (<TableHandle>ref).row_2(p0, p1) + 1
    return slot[0] - 1
cdef inline Py_ssize_t held_row_3(Py_ssize_t* slot, TableRef ref, long long p0, long long p1,
                                  long long p2) except -2:
    if slot[0] == 0:
        slot[0] = (<TableHandle>ref).row_3(p0, p1, p2) + 1
    return slot[0] - 1
# diag-run8 F3: the same for a key whose parts are fixed per item but may be objects (a str label):
# the generated read tests the slot before evaluating the key, so the key is read once per item.
cdef inline Py_ssize_t held_row_obj_1(Py_ssize_t* slot, TableRef ref, object p0) except -2:
    if slot[0] == 0:
        slot[0] = (<TableHandle>ref).row_1(p0) + 1
    return slot[0] - 1
cdef inline Py_ssize_t held_row_obj_2(Py_ssize_t* slot, TableRef ref, object p0, object p1) except -2:
    if slot[0] == 0:
        slot[0] = (<TableHandle>ref).row_2(p0, p1) + 1
    return slot[0] - 1
cdef inline Py_ssize_t held_row_obj_3(Py_ssize_t* slot, TableRef ref, object p0, object p1,
                                      object p2) except -2:
    if slot[0] == 0:
        slot[0] = (<TableHandle>ref).row_3(p0, p1, p2) + 1
    return slot[0] - 1
# D78: the slot of a literal field of a held record handle, resolved on first use and held in a
# per-instance slot (index + 1; 0 unresolved), reset with the handle. KeyError(name) as field().
cdef inline Py_ssize_t held_field(Py_ssize_t* slot, RecordHandle r, object name) except -2:
    if slot[0] == 0:
        slot[0] = r.field(name) + 1
    return slot[0] - 1
cdef TableRef sub_ref_1(TableHandle h, TableKey p0) except NULL
cdef TableRef sub_ref_2(TableHandle h, TableKey p0, TableKey p1) except NULL
cdef TableRef sub_ref_3(TableHandle h, TableKey p0, TableKey p1, TableKey p2) except NULL
cdef TableRef sub_ref_4(TableHandle h, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except NULL
cdef TableRef sub_ref_5(TableHandle h, TableKey p0, TableKey p1, TableKey p2, TableKey p3,
                        TableKey p4) except NULL
cdef object key_label(TableHandle h, Py_ssize_t i)          # h.index[i] in row order: label or tuple
cdef double key_f64(TableHandle h, Py_ssize_t i) except? -1  # float(h.index[i])
cdef object level_extreme(TableHandle h, object level, bint last)  # index.get_level_values(level).min/max()
# ---- end of the M1 handle interface ------------------------------------------------------

cdef class NativeTable:
    cdef readonly TableHandle handle    # D25 Q8: embedded; None until W1.2 builds it
    cdef object _owner, _numeric_view, _indexed_view, _single_label, _input_keys
    cdef object _contiguous_view
    cdef public object _mx_python_frame   # api-tables (D101): the read-only DataFrame given to Python callers
    cdef const double* _input_values
    cdef const unsigned char* _input_numeric
    cdef const long long* _input_positions
    cdef long long _input_low, _input_high
    cdef Py_ssize_t _input_width
    cpdef object row(self, object key)
    cpdef object slice(self, object key)

cdef NativeTable bind_native_table(object value, str cell)
cdef NativeTable _bind_owned(object value, str cell)
cpdef NativeTable table_from_value(object value, str cell, object schema=*)


cdef object omitted_argument

cdef object row(object function, tuple keywords, object table, object key)
