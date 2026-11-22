# cython: freethreading_compatible=True, boundscheck=False, wraparound=False, initializedcheck=False
"""Bind-time preparation of a pandas table (T-2; design toolkit p2/tables/DESIGN.md).

``prepare(value)`` turns the user's DataFrame or Series into a ``PreparedTable`` the compiled reads use: the rows
stored sorted by their level codes, one ascending array of combined codes (and a dense grid when small) for keyed
lookups, typed column blocks copied from the frame, and the index extremes and asof keys. The user's object is
never changed or replaced: it stays what every formula sees (``source``). ``prepare`` returns None, and
``unprepared_reason`` says why, for a table whose reads must stay pandas (a non-unique index, NaN labels, labels
that compare equal although pandas keeps them apart, duplicate column labels, too many key combinations).

A lookup equals ``value.loc[key, column]`` on the tables ``prepare`` accepts: the tests compare them on random
keys, misses included (``tests/test_prepared_tables.py``). The key lookups and value reads compiled code uses are
inline methods of the .pxd.
"""
from libc.math cimport isnan

import threading
from types import MethodType
import weakref

import numpy as np
import pandas as pd

cdef Py_ssize_t _GRID_LIMIT = 65536
cdef long long _COMBINED_LIMIT = 1LL << 62
cdef int _MAX_LEVELS = 8
cdef object _zip_attrs = pd.DataFrame.attrs
cdef object _zip_loc = pd.DataFrame.loc
cdef object _zip_accessor_type = type(pd.DataFrame().loc)
cdef object _zip_accessor_getitem = _zip_accessor_type.__getitem__
cdef object _zip_frame_getitem = pd.DataFrame.__getitem__
cdef object _zip_xs = pd.DataFrame.xs
cdef object _zip_levels = pd.MultiIndex.levels
cdef object _zip_level_get_loc = pd.Index.get_loc


cdef extern from "Python.h":
    Py_hash_t _object_hash "PyObject_Hash"(PyObject* o) except -1
    int _object_equal "PyObject_RichCompareBool"(PyObject* a, PyObject* b, int op) except -1
    int _Py_EQ "Py_EQ"


cdef inline bint _plain_int(object label):
    # an int label (including IntEnum members and numpy integers), not a bool
    return isinstance(label, (int, np.integer)) and not isinstance(label, (bool, np.bool_))


cdef class PreparedTable:
    """A pandas table prepared for keyed reads. Immutable once built."""

    @property
    def source(self):
        """The user's frame or Series this table was prepared from (None once it is gone)."""
        return self._source()

    def __reduce__(self):
        return (prepare, (self.source,))

    cdef Py_ssize_t row_search(self, long long combined) noexcept nogil:
        """The stored row of a combined code by binary search (a table without a grid), or -1. Out of line, so the
        inline reads stay small."""
        cdef Py_ssize_t lo = 0, hi = self.nrows, mid
        while lo < hi:
            mid = (lo + hi) >> 1
            if self.combined[mid] < combined:
                lo = mid + 1
            else:
                hi = mid
        if lo < self.nrows and self.combined[lo] == combined:
            return lo
        return -1

    cdef Py_ssize_t name_position(self, object label) except -2:
        """T-5: the position of column ``label`` (a name chosen at run time), or -1: the hash and equality of a dict
        lookup, through borrowed pointers only (no reference, no lock per read)."""
        cdef Py_hash_t h = _object_hash(<PyObject*>label)
        cdef Py_ssize_t i = <Py_ssize_t>(<size_t>h & <size_t>self.name_mask), position
        while True:
            position = self.name_pos[i]
            if position < 0:
                return -1
            if <PyObject*>self.name_obj[i] == <PyObject*>label or (
                    self.name_hash[i] == h and _object_equal(<PyObject*>self.name_obj[i], <PyObject*>label, _Py_EQ)):
                return position
            i = (i + 1) & self.name_mask

    cdef Py_ssize_t code_obj(self, int level, object key) except -2:
        """The code of any key part at ``level``, or -1 when absent."""
        cdef object code
        codes = self.level_codes[level]
        if codes is None:
            if type(key) is int:            # the common case first (no numpy type lookups)
                return self.code_ll(level, <long long>key)
            if isinstance(key, (float, np.floating)) and float(key).is_integer():
                key = int(key)          # pandas finds an integral float key on an int index
            if not _plain_int(key):
                return -1
            return self.code_ll(level, <long long>key)
        code = (<dict>codes).get(key)
        return -1 if code is None else <Py_ssize_t>code

    cdef Py_ssize_t row_key(self, object key) except -2:
        """The stored row of a full key given as Python objects (a scalar, or a tuple), or -1."""
        cdef Py_ssize_t[8] buffer
        cdef Py_ssize_t code
        cdef int level
        if self.nlevels == 1:
            if isinstance(key, tuple):
                return -1
            code = self.code_obj(0, key)
            return code
        if not isinstance(key, tuple) or len(<tuple>key) != self.nlevels:
            return -1
        for level in range(self.nlevels):
            code = self.code_obj(level, (<tuple>key)[level])
            if code < 0:
                return -1
            buffer[level] = code
        return self.row_codes(buffer, self.nlevels)

    cdef bint prefix_range(self, const Py_ssize_t* codes, int count, Py_ssize_t* start, Py_ssize_t* stop) noexcept nogil:
        """The stored rows [start, stop) whose first ``count`` level codes are ``codes``; False when none."""
        cdef long long low = 0, high
        cdef int i
        cdef Py_ssize_t lo, hi, mid
        if count < 1 or count > self.nlevels:
            return False
        for i in range(count):
            if codes[i] < 0 or codes[i] >= self.level_size[i]:
                return False
            low += codes[i] * self.radix[i]
        high = low + self.radix[count - 1]
        lo, hi = 0, self.nrows
        while lo < hi:
            mid = (lo + hi) >> 1
            if self.combined[mid] < low:
                lo = mid + 1
            else:
                hi = mid
        start[0] = lo
        hi = self.nrows
        while lo < hi:
            mid = (lo + hi) >> 1
            if self.combined[mid] < high:
                lo = mid + 1
            else:
                hi = mid
        stop[0] = lo
        return stop[0] > start[0]

    cdef Py_ssize_t asof_position(self, double x) noexcept nogil:
        """Position in asof_labels of the last label <= x, or -1 (x below the first label, or NaN)."""
        cdef Py_ssize_t lo = 0, hi = self.asof_keys.shape[0], mid
        if isnan(x):
            return -1
        while lo < hi:
            mid = (lo + hi) >> 1
            if self.asof_keys[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    cdef object object_at(self, Py_ssize_t row, Py_ssize_t slot):
        return (<tuple>self.objects[slot])[row]

    cpdef Py_ssize_t column(self, object label) except -2:
        """The position of column ``label`` (None for a Series); KeyError as pandas raises it."""
        position = self.column_positions.get(label)
        if position is None:
            raise KeyError(label)
        return position

    # ------------------------------------------------------------ Python access (tests, fallbacks)
    def key_codes(self, key):
        """Level codes of a full key (a scalar for one level, a tuple otherwise); None when a part is absent."""
        parts = key if self.nlevels > 1 else (key,)
        if not isinstance(parts, tuple) or len(parts) != self.nlevels:
            return None
        codes = []
        for level, part in enumerate(parts):
            code = self.code_obj(level, part)
            if code < 0:
                return None
            codes.append(code)
        return codes

    def row(self, key):
        """The stored row of ``key``, or -1."""
        cdef Py_ssize_t[8] buffer
        codes = self.key_codes(key)
        if codes is None:
            return -1
        for i, code in enumerate(codes):
            buffer[i] = code
        return self.row_codes(buffer, len(codes))

    def value(self, key, column=None):
        """``source.loc[key, column]`` (``source.loc[key]`` for a Series), as pandas returns it."""
        cdef Py_ssize_t row = self.row(key)
        cdef Py_ssize_t position
        if row < 0:
            raise KeyError(key)
        position = self.column(None if self.is_series else column)
        kind = self.column_kind[position]
        slot = self.column_slot[position]
        if kind == COLUMN_F64:
            return np.float64(self.f64_at(row, slot))
        if kind == COLUMN_I64:
            return np.int64(self.i64_at(row, slot))
        return self.object_at(row, slot)

    def contains(self, key):
        """``key in source.index``."""
        return self.row(key) >= 0

    def asof(self, x):
        """``source.index.asof(x)`` for a flat numeric index and a number x (pandas' asof of NaN differs by version:
        the compiled reads read NaN on pandas, and this returns NaN for it)."""
        position = self.asof_position(float(x))
        return self.asof_labels[position] if position >= 0 else np.nan


def unprepared_reason(value):
    """Why ``prepare(value)`` returns None (None when it does not)."""
    try:
        _build(value)
    except ValueError as error:
        return str(error)
    return None


def prepare(value):
    """The PreparedTable of a DataFrame or Series, or None when its reads must stay pandas."""
    try:
        return _build(value)
    except ValueError:
        return None


def _level(values):
    """(kind, labels in code order, first, dict or None) for the labels of one index level."""
    unique = list(dict.fromkeys(values))
    if values.hasnans:
        raise ValueError('NaN labels')
    if len(unique) != len(pd.unique(values)):
        raise ValueError('labels that compare equal but pandas keeps apart')
    if unique and all(_plain_int(label) for label in unique):
        ordered = sorted(unique)
        if ordered[len(ordered) - 1] - ordered[0] + 1 == len(ordered):
            return LEVEL_DENSE, ordered, ordered[0], None
        return LEVEL_SORTED, ordered, 0, None
    try:
        ordered = sorted(unique)
    except TypeError:
        ordered = unique                     # labels without an order: codes in first appearance
    codes = {label: code for code, label in enumerate(ordered)}
    if len(codes) != len(ordered):
        raise ValueError('labels that compare equal but pandas keeps apart')
    return LEVEL_OBJECT, ordered, 0, codes


cdef void _point(PreparedTable table, sorted_array, combined, grid, f64, i64, pos_row):
    """Point the table's raw pointers into its arrays and keep the arrays alive with it."""
    cdef const long long[::1] s = sorted_array
    cdef const long long[::1] c = combined
    cdef const Py_ssize_t[::1] g = grid
    cdef const double[:, ::1] f = f64
    cdef const long long[:, ::1] i = i64
    cdef const Py_ssize_t[::1] p = pos_row
    table._arrays = (sorted_array, combined, grid, f64, i64, pos_row)
    for array in table._arrays:
        array.flags.writeable = False
    table.sorted_labels = &s[0]
    table.combined = &c[0]
    table.grid = &g[0]
    table.f64 = &f[0, 0]
    table.i64 = &i[0, 0]
    table.pos_row = &p[0]


cdef void _name_map(PreparedTable table):
    """T-5: the column name map of ``name_position`` (open addressing, at least twice as many entries as columns)."""
    cdef Py_ssize_t size = 8, i
    while size < 2 * len(table.column_positions):
        size *= 2
    objs = np.zeros(size, dtype=np.intp)
    hashes = np.zeros(size, dtype=np.int64)
    positions = np.full(size, -1, dtype=np.intp)
    for label, position in table.column_positions.items():     # the dict's own key objects (kept alive by it)
        h = hash(label)
        i = h & (size - 1)
        while positions[i] >= 0:
            i = (i + 1) & (size - 1)
        objs[i], hashes[i], positions[i] = id(label), h, position
    cdef const Py_ssize_t[::1] o = objs
    cdef const long long[::1] hh = hashes
    cdef const Py_ssize_t[::1] p = positions
    table._name_arrays = (objs, hashes, positions)
    table.name_obj, table.name_hash, table.name_pos = &o[0], <const Py_hash_t*>&hh[0], &p[0]
    table.name_mask = size - 1


cdef PreparedTable _build(object value):
    cdef PreparedTable table = PreparedTable.__new__(PreparedTable)
    cdef Py_ssize_t i, n
    cdef long long product = 1
    if isinstance(value, pd.Series):
        table.is_series = True
        frame = value.to_frame(name=0)
    elif isinstance(value, pd.DataFrame):
        table.is_series = False
        frame = value
    else:
        raise ValueError(f'not a pandas DataFrame or Series: {type(value).__name__}')
    index = frame.index
    if not index.is_unique:
        raise ValueError('the index is not unique')
    if not table.is_series and not frame.columns.is_unique:
        raise ValueError('duplicate column labels')
    if index.nlevels > _MAX_LEVELS:
        raise ValueError(f'more than {_MAX_LEVELS} index levels')
    table._source = weakref.ref(value)
    table.nrows = n = len(index)
    table.nlevels = index.nlevels
    levels = [_level(index.get_level_values(i)) for i in range(table.nlevels)]
    sorted_labels, offset = [], 0
    for i in range(table.nlevels):
        kind, labels, first, codes = levels[i]
        table.level_kind[i] = kind
        table.level_first[i] = first
        table.level_size[i] = len(labels)
        table.level_offset[i] = offset
        if kind == LEVEL_SORTED:
            sorted_labels.extend(labels)
            offset += len(labels)
    table.level_labels = tuple(level[1] for level in levels)
    table.level_codes = tuple(level[3] for level in levels)
    sorted_array = np.ascontiguousarray(sorted_labels if sorted_labels else [0], dtype=np.int64)
    cdef const long long[::1] sorted_view = sorted_array
    table._arrays = (sorted_array,)              # alive before code_ll reads it below
    table.sorted_labels = &sorted_view[0]
    for i in range(table.nlevels - 1, -1, -1):
        table.radix[i] = product
        product *= max(1, table.level_size[i])
        if product > _COMBINED_LIMIT:
            raise ValueError('too many key combinations')
    codes = np.zeros(n, dtype=np.int64)
    for i in range(table.nlevels):
        labels = index.get_level_values(i)
        level_codes = np.fromiter((table.code_obj(i, label) for label in labels), dtype=np.int64, count=n)
        codes += level_codes * table.radix[i]
    order = np.argsort(codes, kind='stable')
    pos_row = np.zeros(max(1, n), dtype=np.intp)       # T-5: pandas position -> stored row
    pos_row[order] = np.arange(n, dtype=np.intp)
    table.ordered = bool((order == np.arange(n)).all())
    # T-5: per prefix depth d, whether the stored rows of every prefix of d levels are in pandas order (a walk of a
    # view of that depth is then in pandas order); depth 0 is the whole table
    within = [table.ordered]
    for d in range(1, table.nlevels):
        prefix = codes[order] // table.radix[d - 1] if n else codes
        same = prefix[1:] == prefix[:-1]
        within.append(bool(n < 2 or (order[1:][same] > order[:-1][same]).all()))
    table.ordered_within = tuple(within)
    table.level_names = tuple(index.names)
    combined = np.ascontiguousarray(codes[order] if n else np.zeros(1, dtype=np.int64), dtype=np.int64)
    if product <= _GRID_LIMIT:
        grid = np.full(max(1, product), -1, dtype=np.intp)
        grid[combined[:n]] = np.arange(n, dtype=np.intp)
        table.has_grid = True
    else:
        grid = np.zeros(1, dtype=np.intp)
    # columns: copies, stored in the sorted row order
    labels = [None] if table.is_series else list(frame.columns)
    kinds, slots, f64_cols, i64_cols, objects = [], [], [], [], []
    for position, label in enumerate(labels):
        column = frame.iloc[:, position]
        if column.dtype == np.float64:
            kinds.append(COLUMN_F64)
            slots.append(len(f64_cols))
            f64_cols.append(column.to_numpy(dtype=np.float64, copy=True)[order])
        elif column.dtype == np.int64:
            kinds.append(COLUMN_I64)
            slots.append(len(i64_cols))
            i64_cols.append(column.to_numpy(dtype=np.int64, copy=True)[order])
        else:
            kinds.append(COLUMN_OBJECT)
            slots.append(len(objects))
            objects.append(tuple(column.iat[int(row)] for row in order))
    table.column_positions = {label: position for position, label in enumerate(labels)}
    _name_map(table)
    table.column_kind = np.asarray(kinds, dtype=np.intc) if kinds else np.zeros(1, dtype=np.intc)
    table.column_slot = np.asarray(slots, dtype=np.intp) if slots else np.zeros(1, dtype=np.intp)
    f64 = np.ascontiguousarray(np.column_stack(f64_cols) if f64_cols else np.zeros((max(1, n), 1)), dtype=np.float64)
    i64 = np.ascontiguousarray(np.column_stack(i64_cols) if i64_cols else np.zeros((max(1, n), 1), dtype=np.int64),
                               dtype=np.int64)
    table.f64_width, table.i64_width = f64.shape[1], i64.shape[1]
    _point(table, sorted_array, combined, grid, f64, i64, pos_row)
    table.objects = tuple(objects)
    # index extremes and asof keys, as pandas gives them
    table.max_label = index.max() if n else None
    table.min_label = index.min() if n else None
    if n and table.nlevels > 1:
        for level in range(min(table.nlevels, 8)):
            table.max_parts[level] = _int_part(table.max_label[level])
            table.min_parts[level] = _int_part(table.min_label[level])
    if table.nlevels == 1 and n and index.dtype.kind in 'iuf':
        sorted_index = index.sort_values()
        table.asof_keys = np.ascontiguousarray(sorted_index.to_numpy(dtype=np.float64))
        table.asof_labels = [sorted_index[i] for i in range(n)]
    else:
        table.asof_keys = np.zeros(0)
        table.asof_labels = []
    return table



cdef long long _int_part(object part):
    """An int part of an extreme label as a C integer (0 for a part of another kind: never read)."""
    if isinstance(part, (int, np.integer)) and not isinstance(part, (bool, np.bool_)):
        return int(part)
    return 0


# ---- binding (T-3) ----------------------------------------------------------------------------------
# One PreparedTable per frame object, prepared on first use and dropped when the frame dies: a new frame is
# prepared afresh, and no prepared copy of an old frame is ever used for it.
cdef dict _prepared = {}               # id(frame) -> (weak reference to the frame, PreparedTable or None)
_lock = threading.RLock()


def _forget(key):
    with _lock:
        _prepared.pop(key, None)


try:
    from modelx_cython.runtime.table_cells import register_prepared as _register
except ImportError:         # the runtime used on its own (tests)
    pass
else:
    _register(lambda value: bind_ref(value), __name__.split('.')[0])


cdef object prepared(object frame):
    """The PreparedTable of ``frame`` (None when its reads must stay pandas), prepared once per object."""
    key = id(frame)
    with _lock:
        entry = _prepared.get(key)
    if entry is not None and entry[0]() is frame:
        return entry[1]
    signatures = _thread_signatures()
    axes = _zip_origin(frame, signatures)
    table = prepare(frame)
    if table is not None:
        _decide(table, signatures, axes)
    ref = weakref.ref(frame, lambda _, key=key: _forget(key))
    with _lock:
        _prepared[key] = (ref, table)
    return table


cdef PreparedTable _NOT_PREPARED = PreparedTable.__new__(PreparedTable)     # every slot -1: reads stay pandas


cdef object bind_ref(object frame):
    """T-4.2: what an item holds for one frame receiver: its PreparedTable, or a table whose every slot is -1 when
    the frame is not prepared (its reads then run on pandas). Taken once per item and receiver."""
    table = prepared(frame)
    return _NOT_PREPARED if table is None else table


cdef object _signatures = None
_own = threading.local()        # T-5c: each worker thread's copy of the signatures (see _thread_signatures)


cdef object model_signatures():
    """The read signatures of the compiled model this runtime copy belongs to (``_mx_table_signatures.SIGNATURES``:
    (levels, columns) per id, written at build), loaded once."""
    global _signatures
    if _signatures is None:
        import importlib
        package = __name__.rpartition('.')[0]
        try:        # the runtime used on its own (no model package) has no compiled reads
            _signatures = (tuple(importlib.import_module(package + '._mx_table_signatures').SIGNATURES)
                           if package else ())
        except ImportError:
            _signatures = ()
    return _signatures


cdef object _fresh(object value):
    """A copy of a signature element that shares no object with the original (texts and tuples rebuilt; small ints
    and None are immortal)."""
    if type(value) is str:
        return ''.join([c for c in value])
    if type(value) is tuple:
        return tuple([_fresh(v) for v in value])
    return value


cdef object _thread_signatures():
    """T-5c: the model's read signatures as this thread's own objects: a per-item preparation (a frame each item
    computes itself) under several worker threads then touches no Python object another thread reads (no shared
    reference counts)."""
    found = getattr(_own, 'signatures', None)
    if found is None:
        found = _own.signatures = _fresh(tuple(model_signatures()))
    return found


cdef object bind_own(object frame):
    """T-5c: an item's binding of a frame it computes itself (a receiver no Data space publishes): its prepared
    table, prepared for this item only. The shared prepared-table map is only looked up (without the lock: a frame
    another item already prepared is reused); a new preparation is not registered (no lock, no weak reference),
    and its verdicts are decided on this thread's own signatures. The item holds it until its Space is cleared."""
    entry = _prepared.get(id(frame))
    if entry is not None and entry[0]() is frame:
        return _NOT_PREPARED if entry[1] is None else entry[1]
    table = prepare(frame)
    if table is None:
        return _NOT_PREPARED
    _decide(table, _thread_signatures(), None)
    return table


cdef object _zip_origin(object frame, object signatures):
    """Capture axes before pandas can copy metadata during preparation."""
    if (type(frame) is not pd.DataFrame or pd.DataFrame.attrs is not _zip_attrs
            or not any(kind == -7 for _, _, _, kind in signatures)):
        return None
    attrs = frame.attrs
    if type(attrs) is not dict or attrs:
        return None
    index, columns = frame.index, frame.columns
    if type(index) is pd.MultiIndex and type(columns) is pd.Index:
        return index, columns
    return None


cdef Py_ssize_t _zip_slot(PreparedTable table, object label, object axes):
    """Admit the narrow ordered ZIP shape at the original Data Cell's bind."""
    if axes is None:
        return -1
    frame = table._source()
    if (type(frame) is not pd.DataFrame or table.nlevels != 2 or not table.ordered_within[1]
            or pd.DataFrame.loc is not _zip_loc or pd.DataFrame.__getitem__ is not _zip_frame_getitem
            or pd.DataFrame.xs is not _zip_xs or pd.MultiIndex.levels is not _zip_levels):
        return -1
    index, columns = axes
    prefix = index.levels[0]
    if (frame.index is not index or frame.columns is not columns
            or type(prefix) is not pd.Index or type(index.levels[1]) is not pd.Index
            or not _zip_method(prefix.get_loc, prefix, _zip_level_get_loc)
            or index.levels[1].dtype != np.dtype('int64')
            or any(type(key) is not str for key in table.level_labels[0])
            or any(type(name) is not str for name in columns)):
        return -1
    cdef Py_ssize_t position = table.column_positions.get(label, -1)
    if position < 0 or table.column_kind[position] != COLUMN_F64:
        return -1
    table._zip_axes = axes
    return table.column_slot[position] + 1


cdef inline bint _zip_method(object method, object owner, object implementation):
    return type(method) is MethodType and method.__self__ is owner and method.__func__ is implementation


cdef void _decide(PreparedTable table, object signatures, object zip_axes):
    """T-4.2: decide, once per prepared table, the slot of every read signature of the model:
    (levels, columns, label, kind) ->
    - ``levels``: one letter per index level, ``i`` for a level a read takes integer keys on (it must still be an
      int level), ``o`` for any other; the number of levels must be the same;
    - ``columns``: for a row read by field, every (label, kind) of the traced frame (the row's dtype depends on all
      of them); else None;
    - ``label``, ``kind``: the column read and its block kind (-1: index membership; -2: extremes of a non-empty
      index; -3: a walk of the index in pandas order: the table ordered within every prefix of ``label`` levels
      (0 or None: the whole table); -4: a row read by field, ``columns`` given;
      -5: ``T.loc[a, b]`` read as a row prefix, no label of level ``label`` (b's) being a column label;
      -7: a ZIP over an ordered string/int64 prefix and a float64 column).
    The slot is one plus the column's slot in its block (1 for a read of the index), or -1."""
    cdef int level
    cdef Py_ssize_t position
    slots = []
    for levels, columns, label, kind in signatures:
        result = len(levels) == table.nlevels
        if result:
            for level in range(table.nlevels):
                if levels[level] == 'i' and table.level_kind[level] == LEVEL_OBJECT:
                    result = False
                    break
        if result and columns is not None:
            if len(columns) != len(table.column_positions):
                result = False
            else:
                for name, expected in columns:
                    position = table.column_positions.get(name, -1)
                    if position < 0 or table.column_kind[position] != expected:
                        result = False
                        break
        value = -1
        if result:
            if kind == -7:
                value = _zip_slot(table, label, zip_axes)
            elif kind == -6:      # T-5: level label[1] is named label[0] and is an int level (get_level_values)
                value = 1 if (label[1] < table.nlevels and (label[0] is None or table.level_names[label[1]] == label[0])
                              and table.level_kind[label[1]] != LEVEL_OBJECT and table.nrows > 0) else -1
            elif kind == -5:      # T-5: `T.loc[a, b]` is a row prefix: no label of level `label` is a column label
                value = 1 if not any(name in table.column_positions for name in table.level_labels[label]) else -1
            elif kind < 0:      # -1 the index; -2 its extremes (not empty); -3 a walk in pandas order (ordered)
                ordered = table.ordered_within[label] if kind == -3 and label is not None and label < table.nlevels \
                    else table.ordered
                value = 1 if (kind in (-1, -4) or (kind == -2 and table.nrows > 0) or (kind == -3 and ordered)) else -1
            else:
                position = table.column_positions.get(label, -1)
                if position >= 0 and table.column_kind[position] == kind:
                    value = table.column_slot[position] + 1
        slots.append(value)
    array = np.asarray(slots + [-1], dtype=np.intp)
    table._slot_array = array
    table.slots = <const Py_ssize_t*><size_t>array.ctypes.data
    table.n_slots = len(slots)


def native_slots(frame):
    """T-5c (tests): how many of the model's read signatures ``frame`` serves natively (0 when its reads run on
    pandas)."""
    cdef PreparedTable table = prepared(frame)
    if table is None:
        return 0
    return sum(1 for i in range(table.n_slots) if table.slots[i] > 0)


def check_unchanged(frame):
    """T-4 check mode (spec key ``verify_inputs``): raise if ``frame`` no longer equals the copy prepared when it was
    bound, i.e. it was edited in place after its first read."""
    cdef PreparedTable table = prepared(frame)
    cdef PreparedTable fresh
    if table is None:
        return
    fresh = prepare(frame)
    if fresh is None or not _same(table, fresh):
        raise RuntimeError(
            f'a table read by compiled code was edited in place after it was bound ({type(frame).__name__} of '
            f'shape {getattr(frame, "shape", None)}): its fast reads use the copy prepared at the first read; give '
            f'the edited frame to a new model instance instead')


cdef bint _same(PreparedTable a, PreparedTable b):
    if a.nrows != b.nrows or a.nlevels != b.nlevels or a.level_labels != b.level_labels:
        return False
    if a.column_positions != b.column_positions or a.objects != b.objects:
        return False
    if len(a._arrays) != len(b._arrays):
        return False
    for x, y in zip(a._arrays, b._arrays):
        if x.dtype != y.dtype or x.shape != y.shape:
            return False
        if not np.array_equal(x, y, equal_nan=x.dtype.kind == 'f'):
            return False
    return True


cdef object bind_ref_checked(object frame):
    """``bind_ref`` in check mode: the frame is first checked against its prepared copy."""
    check_unchanged(frame)
    return bind_ref(frame)


cdef Py_ssize_t hold_code(object table, int level, object value) except -2:
    """T-4: one plus the code of ``value`` at ``level`` of a held table (resolved once per item), or -1 when the
    table is not prepared or the value is absent (the read then runs on pandas, which raises as it does)."""
    cdef Py_ssize_t code
    if table is None or table is _NOT_PREPARED:
        return -1
    code = (<PreparedTable>table).code_obj(level, value)
    return code + 1 if code >= 0 else -1


cdef Py_ssize_t asof_row(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2,
                         Py_ssize_t c3, Py_ssize_t c4, double x) noexcept:
    """T-4: ``P.index.asof(x)`` on the sub-frame P = ``T.loc[prefix]`` (``count`` prefix codes) whose one remaining
    level is an int level, as the stored row of that label in T: the row of the largest label <= x among the
    prefix's rows; -1 when there is none or x is NaN (pandas returns NaN; the read then runs on pandas).
    Read through the borrowed pointer only (T-4.2: no reference on the shared table per step)."""
    cdef Py_ssize_t[6] codes
    cdef Py_ssize_t start, stop, lo, hi, mid
    cdef int last = (<PreparedTable>ref).nlevels - 1
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4
    if count != last or x != x or (<PreparedTable>ref).level_kind[last] == LEVEL_OBJECT:
        return -1
    if not (<PreparedTable>ref).prefix_range(codes, count, &start, &stop):
        return -1
    lo, hi = start, stop        # rows of the prefix, ascending by the last level's label
    while lo < hi:
        mid = (lo + hi) >> 1
        if _last_label(ref, mid) <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1 if lo > start else -1


cdef inline long long _last_label(PreparedRef ref, Py_ssize_t row) noexcept:
    cdef int last = (<PreparedTable>ref).nlevels - 1
    cdef long long combined = (<PreparedTable>ref).combined[row]
    cdef Py_ssize_t code = <Py_ssize_t>(combined % (<PreparedTable>ref).radix[last - 1]) if last > 0 else <Py_ssize_t>combined
    if (<PreparedTable>ref).level_kind[last] == LEVEL_DENSE:
        return (<PreparedTable>ref).level_first[last] + code
    return (<PreparedTable>ref).sorted_labels[(<PreparedTable>ref).level_offset[last] + code]


cdef Py_ssize_t hold_row(object table, object key, bint assigned) except -2:
    """T-4: one plus the stored row of ``key`` in a held table, for the fields of a row cell (``T.loc[key]``);
    -1 when the table is not prepared, the key is absent, or the row cell holds a value the user assigned
    (``assigned``: the item's own flag, T-4.2; the fields are then read from that value, on pandas)."""
    cdef Py_ssize_t row
    if table is None or table is _NOT_PREPARED or assigned:
        return -1
    row = (<PreparedTable>table).row_key(key)
    return row + 1 if row >= 0 else -1


cdef bint selects_rows(object key):
    """A key part pandas does not look up as one label: it selects rows (a list, an array, a mask, a slice, an
    index, a table) or is a key of several parts."""
    if type(key) is int or type(key) is float or type(key) is str:      # the common labels first
        return False
    return isinstance(key, (list, tuple, slice, set, dict, np.ndarray, pd.Series, pd.Index, pd.DataFrame))


cdef int missing_codes(PreparedRef ref, const Py_ssize_t* codes, int count) except -1:
    """KeyError for a key given as codes: the labels of the parts found, None for a part absent from its level
    (the type is pandas'; the key shown lists the parts found)."""
    cdef PreparedTable table = <PreparedTable>ref
    parts = tuple(table.level_labels[i][codes[i]] if i < table.nlevels and 0 <= codes[i] < table.level_size[i]
                  else None for i in range(count))
    raise KeyError(parts if count > 1 else parts[0])


cdef object object_field(PreparedRef ref, Py_ssize_t slot, Py_ssize_t row):
    return (<PreparedTable>ref).object_at(row - 1, slot - 1)


# ---- T-5: index walks ------------------------------------------------------------------------------------
cdef object label_obj(PreparedRef ref, Py_ssize_t row, int level):
    """The label of any level at a stored row, the object pandas holds."""
    return (<PreparedTable>ref).level_labels[level][level_code_at(ref, row, level)]


cdef long long view_range(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3,
                          Py_ssize_t c4) noexcept:
    """T-5: the stored rows [start, stop) of a prefix given as level codes, packed as ``start << 32 | stop``, or -1
    when the prefix has no row (one search for both ends)."""
    cdef Py_ssize_t[5] codes
    cdef Py_ssize_t start, stop
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4
    if not (<PreparedTable>ref).prefix_range(codes, count, &start, &stop):
        return -1
    return (<long long>start << 32) | <long long>stop


cdef object zip_prepared_prefix(object prepared, object frame, object accessor, object key, Py_ssize_t signature):
    """Read a freshly boxed list from the already published ordered snapshot, or decline."""
    cdef PreparedTable table
    cdef PreparedRef ref
    cdef Py_ssize_t column, row, start, stop
    cdef long long bounds
    if type(prepared) is not PreparedTable or type(frame) is not pd.DataFrame or type(key) is not str:
        return None
    table = <PreparedTable>prepared
    ref = <PreparedRef>table
    column = slot(ref, signature)
    if column <= 0 or table._source() is not frame:
        return None
    index, columns = table._zip_axes
    if (frame.index is not index or frame.columns is not columns or type(accessor) is not _zip_accessor_type
            or _zip_accessor_type.__getitem__ is not _zip_accessor_getitem or pd.DataFrame.loc is not _zip_loc
            or pd.DataFrame.__getitem__ is not _zip_frame_getitem or pd.DataFrame.xs is not _zip_xs
            or pd.MultiIndex.levels is not _zip_levels):
        return None
    prefix = index.levels[0]
    if (type(prefix) is not pd.Index or not _zip_method(prefix.get_loc, prefix, _zip_level_get_loc)
            or not _zip_method(frame.xs, frame, _zip_xs)):
        return None
    if pd.DataFrame.attrs is not _zip_attrs:
        return None
    attrs = frame.attrs
    if type(attrs) is not dict or attrs:
        return None
    bounds = view_range(ref, 1, ocode(ref, 0, key), 0, 0, 0, 0)
    if bounds < 0:
        return None
    start, stop = range_start(bounds), range_stop(bounds)
    cdef list pairs = []
    for row in range(start, stop):
        pairs.append((label_ll(ref, row, 1), f64_row(ref, column, row + 1)))
    return pairs


cdef Py_ssize_t view_start(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3,
                           Py_ssize_t c4) noexcept:
    """The first stored row of the prefix given as ``count`` level codes, or -1 when it has no row."""
    cdef Py_ssize_t[5] codes
    cdef Py_ssize_t start, stop
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4
    if not (<PreparedTable>ref).prefix_range(codes, count, &start, &stop):
        return -1
    return start


cdef Py_ssize_t view_stop(PreparedRef ref, int count, Py_ssize_t c0, Py_ssize_t c1, Py_ssize_t c2, Py_ssize_t c3,
                          Py_ssize_t c4) noexcept:
    cdef Py_ssize_t[5] codes
    cdef Py_ssize_t start, stop
    codes[0] = c0; codes[1] = c1; codes[2] = c2; codes[3] = c3; codes[4] = c4
    if not (<PreparedTable>ref).prefix_range(codes, count, &start, &stop):
        return -1
    return stop


cdef Py_ssize_t position(Py_ssize_t start, Py_ssize_t stop, Py_ssize_t i) except -2:
    """``index[i]`` of a view: the stored row (``i`` counted from the end when negative); IndexError as pandas."""
    cdef Py_ssize_t n = stop - start
    if i < 0:
        i += n
    if i < 0 or i >= n:
        raise IndexError(f'index {i if i >= 0 else i - n} is out of bounds for axis 0 with size {n}')
    return start + i


cdef Py_ssize_t search(PreparedRef ref, Py_ssize_t start, Py_ssize_t stop, int level, double x, bint right) noexcept:
    """``index.searchsorted(x, side)`` of an ordered view on an int level: the position (from ``start``) of the first
    label > x (right) or >= x (left)."""
    cdef Py_ssize_t lo = start, hi = stop, mid
    cdef double label
    while lo < hi:
        mid = (lo + hi) >> 1
        label = <double>label_ll(ref, mid, level)
        if label < x or (right and label == x):
            lo = mid + 1
        else:
            hi = mid
    return lo - start


cdef Py_ssize_t asof_at(PreparedRef ref, Py_ssize_t start, Py_ssize_t stop, int level, double x, int error) except -2:
    """``index.asof(x)`` of an ordered view on an int level as the stored row of the label; when there is none (x
    below the first label, or NaN) the error pandas then raises: ValueError for ``int(...)`` of it (``error`` 0),
    KeyError for a key read with it (1)."""
    cdef Py_ssize_t row
    if x == x:
        row = start + search(ref, start, stop, level, x, True) - 1
        if row >= start:
            return row
    if error == 0:
        raise ValueError('cannot convert float NaN to integer')
    raise KeyError(float('nan'))


cdef object max_obj(PreparedRef ref):
    return (<PreparedTable>ref).max_label


cdef object min_obj(PreparedRef ref):
    return (<PreparedTable>ref).min_label


cdef double field_f64(PreparedRef ref, Py_ssize_t row, object label) except? -1.0:
    """``float(row[label])`` of a stored row held per item (``row`` is one plus the row), the field chosen at run
    time; KeyError as pandas for a label that is not a column."""
    cdef Py_ssize_t position = (<PreparedTable>ref).name_position(label)
    if position < 0:
        raise KeyError(label)
    cdef int kind = (<PreparedTable>ref).column_kind[position]
    cdef Py_ssize_t slot = (<PreparedTable>ref).column_slot[position]
    if kind == COLUMN_F64:
        return (<PreparedTable>ref).f64_at(row - 1, slot)
    if kind == COLUMN_I64:
        return <double>(<PreparedTable>ref).i64_at(row - 1, slot)
    return float((<PreparedTable>ref).object_at(row - 1, slot))


cdef Py_ssize_t named_slot(PreparedRef ref, object label) except -2:
    """One plus the slot of the column named at run time on a table whose columns are all of one kind (its read
    signature checks them); KeyError as pandas for a label that is not a column."""
    cdef Py_ssize_t position = (<PreparedTable>ref).name_position(label)
    if position < 0:
        raise KeyError(label)
    return (<PreparedTable>ref).column_slot[position] + 1


cdef long long level_max_ll(PreparedRef ref, int level) noexcept:
    """``T.index.get_level_values(level).max()`` of an int level (non-empty): its largest label (borrowed pointer
    only)."""
    if (<PreparedTable>ref).level_kind[level] == LEVEL_DENSE:
        return (<PreparedTable>ref).level_first[level] + (<PreparedTable>ref).level_size[level] - 1
    return (<PreparedTable>ref).sorted_labels[(<PreparedTable>ref).level_offset[level]
                                              + (<PreparedTable>ref).level_size[level] - 1]


cdef long long level_min_ll(PreparedRef ref, int level) noexcept:
    if (<PreparedTable>ref).level_kind[level] == LEVEL_DENSE:
        return (<PreparedTable>ref).level_first[level]
    return (<PreparedTable>ref).sorted_labels[(<PreparedTable>ref).level_offset[level]]
