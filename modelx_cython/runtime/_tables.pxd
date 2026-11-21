# Bind-time preparation of a pandas table for the reads the compiler makes fast (design: toolkit p2/tables/DESIGN.md).
# A PreparedTable is built once from the user's DataFrame or Series and never changes afterwards. The key lookups
# and value reads below are inline, so the C compiler inlines them into the compiled model; they take no
# reference on a shared object (an object-column read returns a new reference to the value it reads).

cimport cython
from cpython.ref cimport PyObject

cdef enum:
    LEVEL_DENSE = 0      # consecutive int labels: code = key - first
    LEVEL_SORTED = 1     # other int labels: binary search over the sorted labels
    LEVEL_OBJECT = 2     # any other labels (text, enum members, floats): dict label -> code

cdef enum:
    COLUMN_F64 = 0
    COLUMN_I64 = 1
    COLUMN_OBJECT = 2

cdef class PreparedTable:
    cdef object _source                    # weak reference to the user's frame or Series (``source``): a prepared
                                           # copy never keeps a replaced frame alive
    cdef readonly bint is_series
    cdef readonly Py_ssize_t nrows, nlevels
    cdef readonly tuple level_labels       # per level: its labels in code order (a list)
    cdef tuple level_codes                 # per level: dict label -> code for LEVEL_OBJECT, else None
    cdef int[8] level_kind
    cdef long long[8] level_first          # LEVEL_DENSE
    cdef Py_ssize_t[8] level_size          # distinct labels
    cdef Py_ssize_t[8] level_offset        # LEVEL_SORTED: start of the level's labels in sorted_labels
    cdef const long long* sorted_labels    # raw pointers into the arrays held by `_arrays` (no bounds checks)
    cdef long long[8] radix                # combined code = sum(code[i] * radix[i]); radix[nlevels - 1] = 1
    cdef const long long* combined         # combined code of every stored row, ascending (rows are stored sorted)
    cdef const Py_ssize_t* grid            # combined code -> stored row, or -1 (when the product is small)
    cdef const Py_ssize_t* pos_row         # T-5: pandas position -> stored row
    cdef readonly bint ordered             # T-5: the stored order is pandas order (the index is lexsorted)
    cdef readonly tuple ordered_within     # T-5: per prefix depth, the stored rows of each prefix in pandas order
    cdef readonly tuple level_names        # T-5: index.names
    cdef bint has_grid
    cdef readonly dict column_positions    # column label -> position
    cdef const int[::1] column_kind        # COLUMN_*
    cdef const Py_ssize_t[::1] column_slot # position within its block
    cdef const double* f64                 # stored row x float64 column, row-major
    cdef Py_ssize_t f64_width
    cdef const long long* i64              # stored row x int64 column, row-major
    cdef Py_ssize_t i64_width
    cdef tuple _arrays                     # the numpy arrays the pointers point into (kept alive, never written)
    cdef tuple objects                     # per object column: tuple of the values pandas returns, by stored row
    cdef readonly object max_label, min_label   # index.max(), index.min() as pandas returns them
    cdef long long[8] max_parts, min_parts      # T-5c: their int parts as C integers (0 for another kind)
    cdef const double[::1] asof_keys       # sorted labels of a flat numeric index (for asof)
    cdef readonly object asof_labels       # the labels themselves, by position, as pandas returns them
    cdef object _name_arrays               # T-5: column name map: open addressing, `name_mask + 1` entries of
    cdef const Py_ssize_t* name_obj        # (borrowed pointer to the label, its hash, its position or -1); the
    cdef const Py_hash_t* name_hash        # labels stay alive as the keys of column_positions
    cdef const Py_ssize_t* name_pos
    cdef Py_ssize_t name_mask
    cdef object _slot_array                # T-4.2: per read signature of the model, one plus the slot of its column
    cdef const Py_ssize_t* slots           # (1 for a read of the index), or -1 when this table is not of the kinds
    cdef Py_ssize_t n_slots                # the read was compiled for (decided once, when the table is prepared)

    cdef inline Py_ssize_t code_ll(self, int level, long long key) noexcept nogil:
        """The code of an int key part at ``level``, or -1 (absent, or an object level)."""
        cdef Py_ssize_t lo, hi, mid, end
        cdef int kind = self.level_kind[level]
        if kind == LEVEL_DENSE:
            if key < self.level_first[level] or key - self.level_first[level] >= self.level_size[level]:
                return -1
            return <Py_ssize_t>(key - self.level_first[level])
        if kind == LEVEL_SORTED:
            lo = self.level_offset[level]
            end = hi = lo + self.level_size[level]
            while lo < hi:
                mid = (lo + hi) >> 1
                if self.sorted_labels[mid] < key:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < end and self.sorted_labels[lo] == key:
                return lo - self.level_offset[level]
        return -1

    cdef inline Py_ssize_t row_codes(self, const Py_ssize_t* codes, int count) noexcept nogil:
        """The stored row of a full key given as level codes, or -1."""
        cdef long long combined = 0
        cdef int i
        if count != self.nlevels:
            return -1
        for i in range(count):
            if codes[i] < 0 or codes[i] >= self.level_size[i]:
                return -1
            combined += codes[i] * self.radix[i]
        if count == 1:
            return <Py_ssize_t>combined          # one level: rows are stored in code order, one per code
        if self.has_grid:
            return self.grid[combined]
        return self.row_search(combined)

    cdef inline double f64_at(self, Py_ssize_t row, Py_ssize_t slot) noexcept nogil:
        return self.f64[row * self.f64_width + slot]

    cdef inline long long i64_at(self, Py_ssize_t row, Py_ssize_t slot) noexcept nogil:
        return self.i64[row * self.i64_width + slot]

    cdef Py_ssize_t row_search(self, long long combined) noexcept nogil
    cdef Py_ssize_t name_position(self, object label) except -2
    cdef Py_ssize_t code_obj(self, int level, object key) except -2
    cdef Py_ssize_t row_key(self, object key) except -2
    cdef bint prefix_range(self, const Py_ssize_t* codes, int count, Py_ssize_t* start, Py_ssize_t* stop) noexcept nogil
    cdef Py_ssize_t asof_position(self, double x) noexcept nogil
    cdef object object_at(self, Py_ssize_t row, Py_ssize_t slot)
    cpdef Py_ssize_t column(self, object label) except -2


# ---- reads of compiled code (T-3) ---------------------------------------------------------------------
# A PreparedRef is a borrowed pointer to a PreparedTable held by the reading item (or a local of the formula):
# the holder keeps it alive, so a read through it takes no reference. ``slot`` is one plus the column's slot
# (``slot(ref, signature)``).
ctypedef PyObject* PreparedRef
cdef object prepared(object frame)
cdef object bind_ref(object frame)
cdef object bind_own(object frame)
cdef object bind_ref_checked(object frame)
cdef Py_ssize_t hold_code(object table, int level, object value) except -2
cdef Py_ssize_t asof_row(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                         Py_ssize_t c3, Py_ssize_t c4, double x) noexcept
cdef Py_ssize_t hold_row(object table, object key, bint assigned) except -2
cdef object object_field(PreparedRef ref, Py_ssize_t slot, Py_ssize_t row)


# ---- reads by level codes (T-4; T-5: the one read path for every key) and held rows ----------------------------
# A key of up to six parts is passed as one level code per part: ``held - 1`` for a part fixed per item (resolved
# once by ``hold_code``), ``lcode(ref, level, k)`` for a C integer part, ``ocode(ref, level, k)`` for any other
# part. A missing key raises KeyError, as pandas does.

cdef inline Py_ssize_t lcode(PreparedRef ref, int level, long long key) noexcept nogil:
    return (<PreparedTable>ref).code_ll(level, key)


cdef inline Py_ssize_t ocode(PreparedRef ref, int level, object key) except -2:
    return (<PreparedTable>ref).code_obj(level, key)


# The code of a key part on an int level, by the C type Cython gives the key (decisions from typing, not from the
# form of the key expression): a C integer is looked up directly, a C double as pandas finds an integral float, a C
# bool is no label, any other value as ``ocode`` (looked up as pandas does).
# The list names every C type a generated key expression can have, so that each selects exactly one version: a type
# missing from it would be ambiguous between the integer versions and stop the build.
ctypedef fused _key_t:
    bint
    int
    long
    long long
    Py_ssize_t
    Py_UCS4
    double
    object


cdef inline Py_ssize_t kcode(PreparedRef ref, int level, _key_t key) except -2:
    cdef long long whole
    if _key_t is object:
        return (<PreparedTable>ref).code_obj(level, key)
    elif _key_t is bint:
        return -1
    elif _key_t is Py_UCS4:             # one character of a text value: the text, looked up as pandas does
        return (<PreparedTable>ref).code_obj(level, <object>key)
    elif _key_t is double:
        if key >= -9.0e18 and key <= 9.0e18:
            whole = <long long>key
            if <double>whole == key:
                return (<PreparedTable>ref).code_ll(level, whole)
        return -1
    else:
        return (<PreparedTable>ref).code_ll(level, key)


# Whether a key part names one label. A C number does, decided when the model is compiled; any other value is looked
# at when the read runs: a list, a mask, a slice or a table selects several rows, and the read then runs on pandas.
cdef bint selects_rows(object key)


cdef inline bint is_label(_key_t key) noexcept:
    if _key_t is object:
        return not selects_rows(key)
    else:
        return True


cdef int missing_codes(PreparedRef ref, const Py_ssize_t* codes, int count) except -1


cdef inline Py_ssize_t _row_c(PreparedRef ref, const Py_ssize_t* codes, int count) except -2:
    cdef Py_ssize_t row = (<PreparedTable>ref).row_codes(codes, count)
    if row < 0:
        missing_codes(ref, codes, count)
    return row


cdef inline double f64_c(PreparedRef ref, Py_ssize_t slot, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                         Py_ssize_t c3, Py_ssize_t c4, Py_ssize_t c5) except? -1.0:
    """``frame.loc[key, column]`` of a float64 column, the key given as level codes (unused codes are 0)."""
    cdef Py_ssize_t[6] codes
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4; codes[5] = c5
    return (<PreparedTable>ref).f64_at(_row_c(ref, codes, count), slot - 1)


cdef inline long long i64_c(PreparedRef ref, Py_ssize_t slot, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                            Py_ssize_t c3, Py_ssize_t c4, Py_ssize_t c5) except? -1:
    cdef Py_ssize_t[6] codes
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4; codes[5] = c5
    return (<PreparedTable>ref).i64_at(_row_c(ref, codes, count), slot - 1)


cdef inline bint contains_c(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                            Py_ssize_t c3, Py_ssize_t c4, Py_ssize_t c5) noexcept nogil:
    cdef Py_ssize_t[6] codes
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4; codes[5] = c5
    return (<PreparedTable>ref).row_codes(codes, count) >= 0


cdef inline Py_ssize_t _row_1(PreparedRef ref, Py_ssize_t c0) noexcept nogil:
    """The stored row of a one-part key given as its level code, or -1: on a one-level index the row is the code."""
    if (<PreparedTable>ref).nlevels == 1 and 0 <= c0 < (<PreparedTable>ref).level_size[0]:
        return c0
    return -1


# the same reads of a one-part key (no code array)
cdef inline double f64_1(PreparedRef ref, Py_ssize_t slot, Py_ssize_t c0) except? -1.0:
    cdef Py_ssize_t row = _row_1(ref, c0)
    if row < 0:
        missing_codes(ref, &c0, 1)
    return (<PreparedTable>ref).f64_at(row, slot - 1)


cdef inline long long i64_1(PreparedRef ref, Py_ssize_t slot, Py_ssize_t c0) except? -1:
    cdef Py_ssize_t row = _row_1(ref, c0)
    if row < 0:
        missing_codes(ref, &c0, 1)
    return (<PreparedTable>ref).i64_at(row, slot - 1)


cdef inline bint contains_1(PreparedRef ref, Py_ssize_t c0) noexcept nogil:
    return _row_1(ref, c0) >= 0


# T-5: the stored row of a complete key (a row local ``row = T.loc[key]``); KeyError as pandas
cdef inline Py_ssize_t row_c(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                             Py_ssize_t c3, Py_ssize_t c4, Py_ssize_t c5) except -2:
    cdef Py_ssize_t[6] codes
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4; codes[5] = c5
    return _row_c(ref, codes, count)


cdef inline Py_ssize_t row_1(PreparedRef ref, Py_ssize_t c0) except -2:
    cdef Py_ssize_t row = _row_1(ref, c0)
    if row < 0:
        missing_codes(ref, &c0, 1)
    return row


# a field of a row held per item (``row - 1`` from ``hold_row``)
cdef inline double f64_row(PreparedRef ref, Py_ssize_t slot, Py_ssize_t row) noexcept nogil:
    return (<PreparedTable>ref).f64_at(row - 1, slot - 1)


cdef inline long long i64_row(PreparedRef ref, Py_ssize_t slot, Py_ssize_t row) noexcept nogil:
    return (<PreparedTable>ref).i64_at(row - 1, slot - 1)


# T-4: extremes of a one-level int index (``T.index.max()`` / ``.min()``), as C integers (no reference taken)
cdef inline long long max_ll(PreparedRef ref) noexcept:
    if (<PreparedTable>ref).level_kind[0] == LEVEL_DENSE:
        return (<PreparedTable>ref).level_first[0] + (<PreparedTable>ref).level_size[0] - 1
    return (<PreparedTable>ref).sorted_labels[(<PreparedTable>ref).level_offset[0] + (<PreparedTable>ref).level_size[0] - 1]


cdef inline long long min_ll(PreparedRef ref) noexcept:
    if (<PreparedTable>ref).level_kind[0] == LEVEL_DENSE:
        return (<PreparedTable>ref).level_first[0]
    return (<PreparedTable>ref).sorted_labels[(<PreparedTable>ref).level_offset[0]]


# T-5c: part ``level`` (an int level) of ``T.index.max()`` / ``.min()`` of a non-empty index of several levels, as a C
# integer: the label tuple pandas returns is not touched (no reference taken on an object shared by every thread)
cdef inline long long max_part_ll(PreparedRef ref, int level) noexcept nogil:
    return (<PreparedTable>ref).max_parts[level]


cdef inline long long min_part_ll(PreparedRef ref, int level) noexcept nogil:
    return (<PreparedTable>ref).min_parts[level]


# ---- T-5: index walks: labels by stored row, positions, views of a prefix --------------------------------------
# A view is the stored rows [start, stop) of a prefix (``view_start``/``view_stop``), or the whole table, whose
# pandas positions map to stored rows through ``pos_row``. Walks of a view are native only on an ordered table
# (its read signature), where pandas order is the stored order.

@cython.cdivision(True)
cdef inline Py_ssize_t level_code_at(PreparedRef ref, Py_ssize_t row, int level) noexcept nogil:
    # combined codes are not negative: C division (Python's floor rules would only add branches)
    if level == (<PreparedTable>ref).nlevels - 1:
        return <Py_ssize_t>((<PreparedTable>ref).combined[row] % (<PreparedTable>ref).level_size[level])
    return <Py_ssize_t>(((<PreparedTable>ref).combined[row] // (<PreparedTable>ref).radix[level])
                        % (<PreparedTable>ref).level_size[level])


cdef inline long long label_ll(PreparedRef ref, Py_ssize_t row, int level) noexcept nogil:
    """The label of an int level at a stored row."""
    cdef Py_ssize_t code = level_code_at(ref, row, level)
    if (<PreparedTable>ref).level_kind[level] == LEVEL_DENSE:
        return (<PreparedTable>ref).level_first[level] + code
    return (<PreparedTable>ref).sorted_labels[(<PreparedTable>ref).level_offset[level] + code]


cdef inline Py_ssize_t whole_row(PreparedRef ref, Py_ssize_t position) noexcept nogil:
    return (<PreparedTable>ref).pos_row[position]


cdef inline Py_ssize_t nrows(PreparedRef ref) noexcept nogil:
    return (<PreparedTable>ref).nrows


cdef object label_obj(PreparedRef ref, Py_ssize_t row, int level)
cdef long long view_range(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3,
                          Py_ssize_t c4) noexcept


cdef inline Py_ssize_t range_start(long long packed) noexcept nogil:
    return <Py_ssize_t>(packed >> 32) if packed >= 0 else -1


cdef inline Py_ssize_t range_stop(long long packed) noexcept nogil:
    return <Py_ssize_t>(packed - ((packed >> 32) << 32)) if packed >= 0 else -1


cdef Py_ssize_t view_start(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3,
                           Py_ssize_t c4) noexcept
cdef Py_ssize_t view_stop(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3,
                          Py_ssize_t c4) noexcept
cdef Py_ssize_t position(Py_ssize_t start, Py_ssize_t stop, Py_ssize_t i) except -2
cdef Py_ssize_t search(PreparedRef ref, Py_ssize_t start, Py_ssize_t stop, int level, double x, bint right) noexcept
cdef Py_ssize_t asof_at(PreparedRef ref, Py_ssize_t start, Py_ssize_t stop, int level, double x, int error) except -2
cdef Py_ssize_t named_slot(PreparedRef ref, object label) except -2
cdef double field_f64(PreparedRef ref, Py_ssize_t row, object label) except? -1.0
cdef object max_obj(PreparedRef ref)
cdef long long level_max_ll(PreparedRef ref, int level) noexcept
cdef long long level_min_ll(PreparedRef ref, int level) noexcept
cdef object min_obj(PreparedRef ref)


# T-4.2: the slot of a read, through the table its item holds: one C load (a read stays pandas when it is <= 0)
cdef inline Py_ssize_t slot(PreparedRef ref, int signature) noexcept:
    if signature < (<PreparedTable>ref).n_slots:
        return (<PreparedTable>ref).slots[signature]
    return -1
