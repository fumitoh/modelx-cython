# cython: freethreading_compatible=True
"""Cython implementation of explicit immutable helpers, copied by mx2cy.

Only exact immutable Table/Record instances are specialized. Mutable inputs,
subclasses and replaced function References call the original supplied function.
Views belong to their Table and contain no formula or policy state.
"""
from array import array
from cpython.array cimport array as Array, clone
from enum import IntEnum
import math
from libc.math cimport fabs, isfinite, floor
from libc.limits cimport LONG_MIN, LONG_MAX
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_AsDouble, PyFloat_CheckExact
from cpython.long cimport PyLong_AsLong, PyLong_AsLongLongAndOverflow, PyLong_Check
from cpython.number cimport PyNumber_Index
from cpython.object cimport PyObject_IsTrue
from cpython.tuple cimport PyTuple_GET_ITEM
from cpython.version cimport PY_VERSION_HEX
from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport cython
from modelx_helpers.immutable_inputs import Table, Record, _index_key, _valid_index_key
from cpython.object cimport Py_TYPE as _mx_Py_TYPE
# diag-4w: the helper Table class as a C pointer, for exact-type tests on the per-step read path; reading the
# module global costs a dict lookup and a reference-count update on the shared class object from every worker
cdef PyObject* _TABLE_TYPE = <PyObject*>Table
from modelx_helpers import immutable_inputs as _immutable_inputs
from modelx_helpers.typed_inputs import _FLOOR, _native_operations
from modelx_cython._call_layout cimport forward_call

__modelx_cython_provider__ = (1, 'modelx_cython.runtime._native_inputs', 'owned-table-1')

cdef object _missing = object()
cdef object omitted_argument = _missing
cdef object _sum_f64 = _native_operations['sum_f64']
cdef object _rectangular_storage = getattr(_immutable_inputs, '_RECTANGULAR_STORAGE', object())
cdef object _contiguous_storage = getattr(_immutable_inputs, '_CONTIGUOUS_STORAGE', object())


cdef class TableView:
    cdef tuple values, columns
    cdef dict column_positions
    cdef long long[::1] integers
    cdef long long[::1] sorted_positions
    cdef long long first_integer, last_integer
    cdef bint integral
    cdef bint numeric
    cdef bint contiguous
    cdef Py_ssize_t width
    cdef double[::1] numbers

    def __init__(self, table):
        cdef object key
        cdef tuple keys = table._keys
        cdef tuple ordered
        if not keys:
            return
        for key in keys:
            if type(key) is not int:
                return
        ordered = table._ordered
        low, high = ordered[0], ordered[-1]
        if not -(1 << 63) <= low <= high < (1 << 63):
            return
        self.integral = True
        self.values = table._values
        self.columns = table._columns
        self.column_positions = dict(zip(self.columns, range(len(self.columns))))
        self.integers = array('q', ordered)
        self.sorted_positions = array('q', [table._positions[key] for key in ordered])
        self.first_integer = low
        self.last_integer = high
        # Table guarantees unique keys. Compute the span as Python ints.
        self.contiguous = high - low == len(ordered) - 1
        self.numeric = all(type(value) is float for row in self.values for value in row)
        if self.numeric:
            self.width = len(self.columns)
            self.numbers = array('d', [value for row in self.values for value in row])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline Py_ssize_t right(self, long long key) noexcept:
        cdef Py_ssize_t lo = 0, hi = self.integers.shape[0], mid
        if self.contiguous:
            if key < self.first_integer:
                return 0
            if key >= self.last_integer:
                return hi
            return <Py_ssize_t>(key - self.first_integer) + 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if key < self.integers[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    cdef inline Py_ssize_t position(self, long long key, bint preceding, bint clamp) noexcept:
        cdef Py_ssize_t row
        if clamp and key >= self.last_integer:
            return self.integers.shape[0] - 1
        row = self.right(key) - 1
        if row >= 0 and (preceding or self.integers[row] == key):
            return row
        return -1

    cdef inline double number_at(self, Py_ssize_t row, Py_ssize_t column) noexcept:
        return self.numbers[self.sorted_positions[row] * self.width + column]

    cdef object value(self, Py_ssize_t row, object column):
        cdef Py_ssize_t col
        if type(column) is str:
            found = self.column_positions.get(column)
            if found is None:
                # Preserve tuple.index's exception, including its message.
                col = self.columns.index(column)
            else:
                col = found
            return self.values[row][col]
        if isinstance(column, str):
            return self.values[row][self.columns.index(column)]
        return self.values[row][column]


cdef TableView view(object table):
    return <TableView>table._native_cache.get(TableView)


cdef class GroupedView:
    cdef long long[::1] groups, keys, offsets
    cdef double[::1] values

    def __init__(self, groups, keys, offsets, values):
        self.groups = array('q', groups)
        self.keys = array('q', keys)
        self.offsets = array('q', offsets)
        self.values = array('d', values)


cdef bint f64_accepts(object function):
    return function is _sum_f64 and PY_VERSION_HEX >= 0x030c0000


# Preserve explicit conversion semantics, including replaced callables.

# Primitive full-key lookup over the existing owned Table.
from cpython.object cimport PyObject_Hash
from cpython.unicode cimport PyUnicode_Compare
import sys as _key_sys

cdef enum:
    _KEY_INT = 1
    _KEY_BOOL = 2
    _KEY_TEXT = 3
    _KEY_FLOAT_INT = 4
    _KEY_FLOAT = 5
    _KEY_OBJECT = 6

cdef union FloatKeyBits:
    double value
    unsigned long long bits


import numpy as _key_numpy
cdef tuple _numpy_integer_keys = tuple(set((
    _key_numpy.int8, _key_numpy.int16, _key_numpy.int32, _key_numpy.int64,
    _key_numpy.uint8, _key_numpy.uint16, _key_numpy.uint32, _key_numpy.uint64,
    _key_numpy.intp, _key_numpy.uintp, _key_numpy.longlong, _key_numpy.ulonglong)))
cdef tuple _numpy_float_keys = (_key_numpy.float16, _key_numpy.float32, _key_numpy.float64)
cdef object _numpy_bool_key = _key_numpy.bool_


cdef void _key_real(KeyPart* part, double number, object original) noexcept:
    cdef FloatKeyBits storage
    part.number = 0
    part.text = <PyObject*>original
    if not isfinite(number):
        part.kind = _KEY_OBJECT
    elif -9223372036854775808.0 <= number < 9223372036854775808.0 and floor(number) == number:
        part.kind = _KEY_FLOAT_INT
        part.number = <long long>number
    else:
        part.kind = _KEY_FLOAT
        storage.value = number
        part.number = <long long>storage.bits


cdef void key_int(KeyPart* part, long long value, bint boolean) noexcept:
    part.kind = _KEY_BOOL if boolean else _KEY_INT
    part.number = value
    part.text = NULL

cdef void key_text(KeyPart* part, object value) noexcept:
    part.number = 0
    part.text = <PyObject*>value
    if type(value) is str:
        part.kind = _KEY_TEXT
    elif type(value) is float:
        _key_real(part, PyFloat_AS_DOUBLE(value), value)
    else:
        part.kind = _KEY_OBJECT

cdef bint _native_parts(const KeyPart* parts, Py_ssize_t count) noexcept:
    cdef Py_ssize_t i
    for i in range(count):
        if parts[i].kind not in (_KEY_INT, _KEY_BOOL, _KEY_TEXT, _KEY_FLOAT_INT, _KEY_FLOAT):
            return False
    return True

cdef unsigned long long _parts_hash(const KeyPart* parts, Py_ssize_t count, bint tuple_key) except? 0:
    cdef unsigned long long value = 1469598103934665603
    cdef Py_ssize_t i
    value = (value ^ <unsigned long long>tuple_key) * <unsigned long long>1099511628211
    value = (value ^ <unsigned long long>count) * <unsigned long long>1099511628211
    for i in range(count):
        if parts[i].kind == _KEY_TEXT:
            value = (value ^ 3) * <unsigned long long>1099511628211
            value = (value ^ <unsigned long long>PyObject_Hash(<object>parts[i].text)) * <unsigned long long>1099511628211
        elif parts[i].kind == _KEY_FLOAT:
            value = (value ^ 4) * <unsigned long long>1099511628211
            value = (value ^ <unsigned long long>parts[i].number) * <unsigned long long>1099511628211
        else:
            # bool and int have equal-key semantics in the existing Table.
            value = (value ^ 1) * <unsigned long long>1099511628211
            value = (value ^ <unsigned long long>parts[i].number) * <unsigned long long>1099511628211
    return value

cdef class KeyedView:
    cdef signed char[::1] boolean_modes
    # The original tuple keeps every object referenced by parts alive.
    cdef tuple keys, columns, raw_values
    cdef KeyPart* parts
    cdef long long[::1] slots
    cdef unsigned long long[::1] hashes
    cdef double[::1] values
    cdef unsigned char[::1] numeric
    cdef Py_ssize_t width, value_width, rows, mask
    cdef bint tuple_keys, admitted, pandas_columns

    def __cinit__(self, table):
        cdef KeyPart* parts
        cdef Py_ssize_t row, slot, offset, i, capacity
        cdef unsigned long long code
        cdef int overflow
        cdef object key, part, kind, value
        cdef tuple input_row
        spec = table._index_error
        if spec is not None:
            spec = spec._index
        self.pandas_columns = False if spec is None else spec.get('pandas_columns', False)
        self.boolean_modes = array('b', () if spec is None else spec['boolean_modes'])
        self.keys = table._keys
        self.rows = len(self.keys)
        self.columns = table._columns
        self.raw_values = table._values
        self.value_width = len(self.columns)
        self.width = 0
        self.tuple_keys = False
        if not self.keys:
            self.admitted = True
            return
        self.tuple_keys = type(self.keys[0]) is tuple
        self.width = len(self.keys[0]) if self.tuple_keys else 1
        if self.tuple_keys and spec is not None and not spec['multi']:
            # A plain object Index can contain tuple labels. Pandas treats
            # these as scalar keys and does not validate their components.
            self.boolean_modes = array('b')
        if (self.width and self.rows > _key_sys.maxsize // sizeof(KeyPart) // self.width
                or self.rows > _key_sys.maxsize // sizeof(double) // self.value_width):
            raise MemoryError()
        self.parts = <KeyPart*>PyMem_Malloc(max(1, self.rows * self.width) * sizeof(KeyPart))
        if self.parts == NULL:
            raise MemoryError()
        capacity = 8
        while capacity < self.rows * 2:
            capacity *= 2
        if capacity > _key_sys.maxsize // 8:
            return
        self.slots = array('q', [-1] * capacity)
        self.hashes = array('Q', [0] * capacity)
        self.mask = capacity - 1
        for row in range(self.rows):
            key = self.keys[row]
            if (type(key) is tuple) != self.tuple_keys:
                return
            if self.tuple_keys and len(key) != self.width:
                return
            parts = self.parts + row * self.width
            for i in range(self.width):
                part = (<tuple>key)[i] if self.tuple_keys else key
                kind = type(part)
                if kind is int or kind is bool:
                    key_int(&parts[i], PyLong_AsLongLongAndOverflow(part, &overflow), kind is bool)
                    if overflow:
                        return
                else:
                    key_text(&parts[i], part)
                    if parts[i].kind == _KEY_OBJECT:
                        return
            code = _parts_hash(parts, self.width, self.tuple_keys)
            slot = <Py_ssize_t>(code & <unsigned long long>self.mask)
            while self.slots[slot] >= 0:
                slot = (slot + 1) & self.mask
            self.slots[slot] = row
            self.hashes[slot] = code
        # These slots serve explicit float reads. Keep the original objects
        # for ordinary selection and materialization.
        cdef Array numbers = clone(array('d'), self.rows * self.value_width, False)
        cdef Array valid = clone(array('B'), self.rows * self.value_width, True)
        offset = 0
        for input_row in self.raw_values:
            for value in input_row:
                if PyFloat_CheckExact(value):
                    numbers.data.as_doubles[offset] = PyFloat_AS_DOUBLE(value)
                    valid.data.as_uchars[offset] = 1
                elif type(value) in _numpy_integer_keys:
                    numbers.data.as_doubles[offset] = PyFloat_AsDouble(value)
                    valid.data.as_uchars[offset] = 1
                offset += 1
        self.values = numbers
        self.numeric = valid
        self.admitted = True

    def __dealloc__(self):
        PyMem_Free(self.parts)

    cdef bint equal(self, Py_ssize_t row, const KeyPart* parts) except -1:
        cdef Py_ssize_t i
        cdef const KeyPart* stored = self.parts + row * self.width
        for i in range(self.width):
            if self.boolean_modes.shape[0] and self.boolean_modes[i] and (
                    (parts[i].kind == _KEY_BOOL) != (stored[i].kind == _KEY_BOOL)):
                return False
            if parts[i].kind == _KEY_TEXT:
                if (stored[i].kind != _KEY_TEXT
                        or (parts[i].text != stored[i].text
                            and PyUnicode_Compare(<object>parts[i].text, <object>stored[i].text) != 0)):
                    return False
            elif parts[i].kind == _KEY_FLOAT:
                if stored[i].kind != _KEY_FLOAT or parts[i].number != stored[i].number:
                    return False
            else:
                if stored[i].kind not in (_KEY_INT, _KEY_BOOL, _KEY_FLOAT_INT) or parts[i].number != stored[i].number:
                    return False
        return True

    cdef Py_ssize_t find(self, const KeyPart* parts, Py_ssize_t count, bint tuple_key) except? -2:
        cdef unsigned long long code
        cdef Py_ssize_t slot, row
        if not self.rows or count != self.width or tuple_key != self.tuple_keys:
            return -1
        code = _parts_hash(parts, count, tuple_key)
        slot = <Py_ssize_t>(code & <unsigned long long>self.mask)
        while self.slots[slot] >= 0:
            row = self.slots[slot]
            if code == self.hashes[slot] and self.equal(row, parts):
                return row
            slot = (slot + 1) & self.mask
        return -1


cdef Py_ssize_t _owned_position(object table, object key) except? -2:
    cdef object spec = table._index_error
    cdef object position
    cache = table._native_cache
    rectangular = cache.get(_rectangular_storage)
    contiguous = cache.get(_contiguous_storage)
    if rectangular is not None or contiguous is not None:
        try:
            return table._compact_position(key, rectangular, contiguous)
        except KeyError:
            return -1
    if spec is not None:
        spec = spec._index
        if not _valid_index_key(key, spec):
            return -1
        key = _index_key(key, spec)
    position = table._positions.get(key)
    return -1 if position is None else position


cdef inline bint _key_type_is_one_of(object kind, tuple supported) noexcept:
    for candidate in supported:
        if kind is candidate:
            return True
    return False


cdef void _pack_part(object value, KeyPart* part) except *:
    cdef int overflow = 0
    cdef long long number
    kind = type(value)
    if (kind is int or kind is bool) and -(1 << 63) <= value < (1 << 63):
        key_int(part, value, kind is bool)
    elif kind is str or kind is float:
        key_text(part, value)
    elif kind is _numpy_bool_key:
        key_int(part, PyObject_IsTrue(value), True)
        part.text = <PyObject*>value
    elif _key_type_is_one_of(kind, _numpy_integer_keys):
        number = PyLong_AsLongLongAndOverflow(PyNumber_Index(value), &overflow)
        if overflow:
            key_text(part, value)
        else:
            key_int(part, number, False)
            part.text = <PyObject*>value
    elif _key_type_is_one_of(kind, _numpy_float_keys):
        _key_real(part, PyFloat_AsDouble(value), value)
    else:
        key_text(part, value)


cdef KeyPart* _pack_key(object key, KeyPart* small, Py_ssize_t* count,
                         bint* tuple_key) except? NULL:
    cdef KeyPart* parts = small
    cdef Py_ssize_t i
    tuple_key[0] = type(key) is tuple
    if not tuple_key[0]:
        count[0] = 1
        _pack_part(key, small)
        return small if _native_parts(small, 1) else NULL
    count[0] = len(key)
    if count[0] > 4:
        if count[0] > _key_sys.maxsize // sizeof(KeyPart):
            raise MemoryError()
        parts = <KeyPart*>PyMem_Malloc(count[0] * sizeof(KeyPart))
        if parts == NULL:
            raise MemoryError()
    try:
        for i in range(count[0]):
            _pack_part((<tuple>key)[i], &parts[i])
        if _native_parts(parts, count[0]):
            return parts
    except:
        if parts != small:
            PyMem_Free(parts)
        raise
    if parts != small:
        PyMem_Free(parts)
    return NULL


cdef Py_ssize_t _indexed_key_position(object table, object key) except? -3:
    cdef KeyPart[4] small
    cdef Py_ssize_t count
    cdef bint tuple_key
    cdef KeyPart* parts = _pack_key(key, &small[0], &count, &tuple_key)
    cdef KeyedView data
    if parts == NULL:
        return -2
    try:
        data = table._native_cache.get(KeyedView)
        if data is not None and data.admitted:
            return data.find(parts, count, tuple_key)
        return _owned_position(table, key)
    finally:
        if parts != &small[0]:
            PyMem_Free(parts)


cdef Py_ssize_t indexed_position(object table, object key) except? -2:
    """Native unique-row position, or -1 when the key domain is not admitted."""
    cdef Py_ssize_t position = _indexed_key_position(table, key)
    if position == -1:
        raise KeyError(key)
    return -1 if position == -2 else position


from builtins import int as _typed_int, bool as _typed_bool, str as _typed_str
from modelx_helpers import typed_inputs as _typed_inputs_module
from modelx_helpers import pdn as _pdn_module
from modelx_helpers.typed_inputs import _typed_value as _typed_python_value, _MISSING as _typed_missing
cdef dict _typed_inputs_bindings = vars(_typed_inputs_module)
cdef dict _pdn_bindings = vars(_pdn_module)


cdef class ContiguousView:
    cdef long long low, high
    cdef const long long[:, ::1] integers
    cdef const double[:, ::1] floats
    cdef tuple columns
    cdef bint integral

    def __init__(self, long long low, data, tuple columns):
        self.low = low
        self.high = low + (data.shape[0] - 1)
        self.columns = columns
        self.integral = data.dtype == _key_numpy.dtype('int64')
        if self.integral:
            self.integers = data
        else:
            self.floats = data

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline bint read(self, long long key, object column, double* value) except -1:
        cdef Py_ssize_t pos, row
        if key < self.low or key > self.high or type(column) is not str:
            return False
        if <PyObject*>column == PyTuple_GET_ITEM(self.columns, 0):
            pos = 0
        else:
            try:
                pos = self.columns.index(column)
            except ValueError:
                return False
        row = <Py_ssize_t>(key - self.low)
        value[0] = <double>self.integers[row, pos] if self.integral else self.floats[row, pos]
        return True


@cython.final
cdef class NativeTable:
    """A Cython receiver for a Table owned by an ordinary Cell cache."""

    def __init__(self, owner):
        cdef TableView data
        cdef KeyedView indexed
        if not isinstance(owner, Table):
            raise TypeError('declared Table Cell returned a different type')
        self._owner = owner
        # M1 (W1.2): the typed handle built at preparation (compiled pdn.table) or by the first
        # table_handle() request; a bind only finds it and never writes the table's cache
        self.handle = owner._native_cache.get(TableHandle) if type(owner) is Table else None
        if type(owner) is Table:
            self._numeric_view = None
            self._contiguous_view = owner._native_cache.get(ContiguousView)
            if _contiguous_storage in owner._native_cache:
                return
            if _rectangular_storage in owner._native_cache:
                return
        if type(owner) is Table:
            keys = owner._keys
            if keys and (type(keys[0]) is int or len(keys) == 1 and type(keys[0]) is str):
                data = view(owner)
                spec = owner._index_error
                if spec is None and data is not None and data.numeric:
                    self._numeric_view = data
                    return
                if spec is not None and spec._index['multi']:
                    return
                cached = owner._native_cache.get(KeyedView)
                if data is not None and data.integral and data.contiguous:
                    self._input_keys = data
                    self._input_positions = &data.sorted_positions[0]
                    self._input_low = data.first_integer
                    self._input_high = data.last_integer
                if cached is not None:
                    indexed = <KeyedView>cached
                    if not indexed.admitted:
                        return
                    self._indexed_view = indexed
                    self._input_values = &indexed.values[0]
                    self._input_numeric = &indexed.numeric[0]
                    self._input_width = indexed.value_width
                    if self._input_keys is None and len(keys) == 1 and type(keys[0]) is str:
                        self._single_label = keys[0]

    cpdef object row(self, object key):
        if type(self._owner) is Table:
            return row(_row, (), self._owner, key)
        return self._owner.row(key)

    cpdef object slice(self, object key):
        if type(self._owner) is Table:
            return row(_row, (), self._owner, key)
        return self._owner.slice(key)

    @property
    def index(self):
        """The helper's own index (D54): one forwarding point, so every read, operator and
        numpy protocol of the index (``k in index``, ``index + 2``, ``np.asarray(index)``)
        answers exactly as live."""
        return self._owner.index

    def __getitem__(self, key):
        return self._owner[key]

    def __iter__(self):
        return iter(self._owner)

    def __len__(self):
        return len(self._owner)

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def __modelx_cython_owner__(self):
        """Expose the semantic owner when passing a view to another package."""
        return _semantic_owner(self._owner)

    def __modelx_table_owner__(self):
        return _semantic_owner(self._owner)

    def __modelx_table_owner_abi__(self):
        return 1

    def __eq__(self, other):
        return _semantic_owner(self._owner) == other

    # auto-tables (D104): operators answered by the owning helper Table (pandas on a frame-built table,
    # NotImplemented, hence TypeError, on a pdn.table table)
    def __lt__(self, other):
        return _owner_operator(self, '__lt__', other)

    def __le__(self, other):
        return _owner_operator(self, '__le__', other)

    def __gt__(self, other):
        return _owner_operator(self, '__gt__', other)

    def __ge__(self, other):
        return _owner_operator(self, '__ge__', other)

    def __add__(self, other):
        return _owner_operator(self, '__add__', other)

    def __radd__(self, other):
        return _owner_operator(self, '__radd__', other)

    def __sub__(self, other):
        return _owner_operator(self, '__sub__', other)

    def __rsub__(self, other):
        return _owner_operator(self, '__rsub__', other)

    def __mul__(self, other):
        return _owner_operator(self, '__mul__', other)

    def __rmul__(self, other):
        return _owner_operator(self, '__rmul__', other)

    def __truediv__(self, other):
        return _owner_operator(self, '__truediv__', other)

    def __rtruediv__(self, other):
        return _owner_operator(self, '__rtruediv__', other)

    def __floordiv__(self, other):
        return _owner_operator(self, '__floordiv__', other)

    def __rfloordiv__(self, other):
        return _owner_operator(self, '__rfloordiv__', other)

    def __mod__(self, other):
        return _owner_operator(self, '__mod__', other)

    def __rmod__(self, other):
        return _owner_operator(self, '__rmod__', other)

    def __pow__(self, other, mod=None):
        return _owner_operator(self, '__pow__', other)

    def __rpow__(self, other, mod=None):
        return _owner_operator(self, '__rpow__', other)

    def __and__(self, other):
        return _owner_operator(self, '__and__', other)

    def __rand__(self, other):
        return _owner_operator(self, '__rand__', other)

    def __or__(self, other):
        return _owner_operator(self, '__or__', other)

    def __ror__(self, other):
        return _owner_operator(self, '__ror__', other)

    def __xor__(self, other):
        return _owner_operator(self, '__xor__', other)

    def __rxor__(self, other):
        return _owner_operator(self, '__rxor__', other)

    def __neg__(self):
        return getattr(_semantic_owner(self._owner), '__neg__')()

    def __pos__(self):
        return getattr(_semantic_owner(self._owner), '__pos__')()

    def __abs__(self):
        return getattr(_semantic_owner(self._owner), '__abs__')()

    def __invert__(self):
        return getattr(_semantic_owner(self._owner), '__invert__')()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return NativeTable, (_semantic_owner(self._owner),)


Table.register(NativeTable)


def _owner_operator(NativeTable table, str name, object other):
    if type(other) is NativeTable:
        other = _semantic_owner((<NativeTable>other)._owner)
    return getattr(_semantic_owner(table._owner), name)(other)


cdef NativeTable bind_native_table(object value, str cell):
    """Bind a Table or keep a local wrapper already owned by another Cell."""
    if type(value) is NativeTable:
        return <NativeTable>value
    return table_from_value(value, cell)


cdef NativeTable _bind_owned(object value, str cell):
    value_type = type(value)
    if value_type is NativeTable:
        return <NativeTable>value
    if value_type is Table or Table in value_type.__mro__:
        return NativeTable(value)
    if isinstance(value, Table):
        abi = getattr(value_type, '__modelx_table_owner_abi__', None)
        owner = (getattr(value_type, '__modelx_cython_owner__', None)
                 or getattr(value_type, '__modelx_table_owner__', None))
        if callable(abi) and callable(owner) and abi(value) == 1:
            value = owner(value)
            if Table in type(value).__mro__:
                return NativeTable(value)
    raise TypeError(f'{cell} expected a Table result, got {value_type.__name__}. '
                    "Rebuild with a representative sample or set return_type='object'.")


def _schema_mismatch(found, expected):
    """Why a table of schema ``found`` cannot replace one the compiled code was built for, or None.

    The native reads depend on the key levels (count and kind) and on the kinds of the columns
    traced at build time; rows, key range, storage form and extra columns may differ.
    """
    keys, want = tuple(found.get('keys') or ()), tuple(expected.get('keys') or ())
    if keys != want:
        return (f'its key levels have kinds {keys!r} where the compiled model was built for '
                f'{want!r}; the index must have the same levels with the same kinds')
    columns = found.get('columns') or {}
    for name, kind in (expected.get('columns') or {}).items():
        if name not in columns:
            return (f'it has no column {name!r}; the compiled model was built for the columns '
                    f'{sorted(expected.get("columns") or ())!r} (extra columns are allowed)')
        if columns[name] != kind:
            return (f'its column {name!r} holds {columns[name]!r} values where the compiled model '
                    f'reads {kind!r}')
    return None


cpdef NativeTable table_from_value(object value, str cell, object schema=None):
    """Bind a formula result to the native helper Table provider.

    Accepts a helper Table, a view owned by a helper Table or a compatible pandas DataFrame.
    The frame validator raises TableAdmissionError for unsupported values. When supplied,
    ``schema`` checks the resulting table against the key and column kinds used by the Cell.
    """
    source = getattr(type(value), '__modelx_table_cell_source__', None)
    if source is not None:
        value = source(value)
    elif not isinstance(value, Table) and type(value).__module__.split('.')[0] == 'pandas':
        from modelx_cython.runtime.table_admission import admit
        value = admit(value, cell)
    table = _bind_owned(value, cell)
    if schema is not None:
        why = _schema_mismatch(_immutable_inputs.table_schema(_semantic_owner(table._owner)), schema)
        if why is not None:
            raise ValueError(f'{cell}: the assigned table does not fit the compiled model: {why}')
    return table


import pandas as _pandas
cdef object _pandas_isna = _pandas.isna
_isna_origin = getattr(_pandas_isna, '__module__', None)
if (_pandas_isna is not getattr(_pandas, 'isnull', None)
        or not isinstance(_isna_origin, str) or _isna_origin.partition('.')[0] != 'pandas'):
    # A replacement installed before import must retain its own semantics.
    _pandas_isna = object()


from cpython.object cimport PyTypeObject
from modelx_helpers.table_operations import row as _row

# Cython 3.2.9's bundled declaration takes PyObject*, whereas Python 3.12's
# actual API takes PyTypeObject*. Keep the real signature for GCC14 as well.
cdef extern from "Python.h":
    object _record_alloc "PyType_GenericAlloc"(PyTypeObject*, Py_ssize_t)


cdef object row(object function, tuple keywords, object table, object key):
    cdef Py_ssize_t position
    cdef tuple metadata
    cdef object prepared, index, spec
    if function is not _row or type(table) is not Table:
        return forward_call(function, keywords, ('table', 'key'), (table, key), omitted_argument)
    metadata = table._metadata
    if metadata is None or metadata[0] != 'frame':
        raise ValueError('row requires table(..., rows=True) with pandas row metadata')
    if metadata[4] is None:
        raise ValueError('row requires table(..., rows=True)')
    prepared = metadata[10] if len(metadata) > 10 else None
    if prepared is not None and type(key) is int:
        # A matching exact integer label can read the prepared row through
        # the existing key index. Other domains keep their pandas key rules.
        index = table._positions.get(key)
        if index is not None and type(table._keys[index]) is int:
            return prepared[index]
    if type(key) is bool and table._index_error is not None:
        spec = table._index_error._index
        if not spec.get('row_bool_labels', spec['boolean_modes'][0] == 1):
            raise KeyError(f'{key}: boolean label can not be used without a boolean index')
    position = indexed_position(table, key)
    if position < 0:
        return forward_call(function, keywords, ('table', 'key'), (table, key), omitted_argument)
    if prepared is not None:
        return prepared[position]
    return _row(table, key)


import numpy as _group_numpy


# ---- M1 typed table handles (W1.2; interface in _native_inputs.pxd, M1-INTERFACE.md) -----
# A TableHandle is built once per helper Table object from the helper's storage contract
# (`table_storage`, W1.H), with every sub-table of every key prefix built eagerly as a view of
# the same arrays (D25 Q2). Nothing is written after construction, so reads take no lock, and
# no read touches a Python attribute, method or dict of a shared object: keys are normalised
# into C parts and looked up in C indexes, values are read from C arrays. The few reads that
# must run Python code (exotic label objects, asof on an index not in increasing order, cells
# of object arrays, row_record) are counted in `handle_fallbacks()`.

from cpython.bool cimport PyBool_Check
from cpython.float cimport PyFloat_Check, PyFloat_FromDouble
from cpython.long cimport PyLong_FromLongLong, PyLong_AsDouble
from cpython.unicode cimport PyUnicode_Check
from cpython.tuple cimport PyTuple_Check, PyTuple_GET_SIZE
from cpython.object cimport PyObject_TypeCheck, PyTypeObject
from cpython.number cimport PyNumber_Float, PyNumber_Long
from cpython.bytearray cimport PyByteArray_AS_STRING
from cpython.sequence cimport PySequence_Fast_ITEMS
from libc.math cimport isnan, isinf
from libc.string cimport memcpy

cdef extern from *:
    """
    /* Process-wide debug counter, with the one atomics mechanism of _native_vectors.pxd:
       CPython's cpython/pyatomic.h from 3.13 (_Py_atomic_add_int64,
       _Py_atomic_load_int64_relaxed; underscore-prefixed interpreter API); before 3.13 every
       caller holds the GIL (the read runs Python object code), so a plain counter is exact. */
    static int64_t mx_handle_fallbacks_ = 0;
    #if PY_VERSION_HEX >= 0x030D0000
    static void mx_handle_fallback(void) { (void)_Py_atomic_add_int64(&mx_handle_fallbacks_, 1); }
    static long long mx_handle_fallback_count(void) { return (long long)_Py_atomic_load_int64_relaxed(&mx_handle_fallbacks_); }
    #else
    static void mx_handle_fallback(void) { mx_handle_fallbacks_ += 1; }
    static long long mx_handle_fallback_count(void) { return (long long)mx_handle_fallbacks_; }
    #endif
    static inline Py_hash_t mx_mix64(unsigned long long x) {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; x ^= x >> 31;
        return (Py_hash_t)(x >> 1);
    }
    static inline Py_hash_t mx_hash_double(double d) {
        unsigned long long u; memcpy(&u, &d, sizeof(u)); return mx_mix64(u ^ 0x9e3779b97f4a7c15ULL);
    }
    static inline Py_hash_t mx_hash_step(Py_hash_t acc, Py_hash_t h) {
        return (Py_hash_t)(((unsigned long long)acc ^ (unsigned long long)h) * 1000003ULL);
    }
    """
    void mx_handle_fallback() noexcept nogil
    long long mx_handle_fallback_count() noexcept nogil
    Py_hash_t mx_mix64(unsigned long long x) noexcept nogil
    Py_hash_t mx_hash_double(double d) noexcept nogil
    Py_hash_t mx_hash_step(Py_hash_t acc, Py_hash_t h) noexcept nogil

cdef extern from "Python.h":
    # PyObject* spellings: no reference counting on shared labels during a read.
    int mx_unicode_compare "PyUnicode_Compare"(PyObject* left, PyObject* right)
    int mx_rich_equal "PyObject_RichCompareBool"(PyObject* left, PyObject* right, int op) except -1
    PyObject* mx_tuple_item "PyTuple_GET_ITEM"(PyObject* tuple, Py_ssize_t i) noexcept nogil
    Py_ssize_t mx_tuple_size "PyTuple_GET_SIZE"(PyObject* tuple) noexcept nogil
    bint mx_tuple_check_exact "PyTuple_CheckExact"(PyObject* value) noexcept nogil
    bint mx_long_check_exact "PyLong_CheckExact"(PyObject* value) noexcept nogil
    ctypedef struct PyMemberDef:
        Py_ssize_t offset
    ctypedef struct PyMemberDescrObject:
        PyMemberDef* d_member

cdef enum:
    HK_NONE = 0         # never matches (an int beyond 64 bits, an unhashable object)
    HK_INT = 1
    HK_FLOAT = 2
    HK_STR = 3
    HK_MISSING = 4      # NaN, None, pd.NA, pd.NaT under a pandas index spec
    HK_OBJ = 5          # any other label object: == and hash (counted fallback)

cdef enum:
    HC_F64 = 1
    HC_I64 = 2
    HC_BOOL = 3
    HC_OBJ = 4

cdef enum:
    HB_NO = 0
    HB_BOOL = 1
    HB_CINT = 2         # a C integer part: a bool on a bool-only level when 0 or 1

cdef object _table_storage_fn = getattr(_immutable_inputs, 'table_storage', None)
cdef object _no_extreme = object()
cdef PyTypeObject* _np_integer_type = <PyTypeObject*>_group_numpy.integer
cdef PyTypeObject* _np_floating_type = <PyTypeObject*>_group_numpy.floating
cdef PyTypeObject* _np_bool_type = <PyTypeObject*>_group_numpy.bool_
cdef object _np_float64 = _group_numpy.dtype('float64')
cdef object _np_int64 = _group_numpy.dtype('int64')
cdef object _pd_na = getattr(_pandas, 'NA', _missing)
cdef object _pd_nat = getattr(_pandas, 'NaT', _missing)
cdef Py_ssize_t _record_items_offset = -1


cdef Py_ssize_t _slot_offset(object owner_type, str name):
    """Byte offset of a __slots__ member, so a hot read takes the immutable field directly."""
    descr = owner_type.__dict__.get(name)
    if descr is None or type(descr).__name__ != 'member_descriptor':
        return -1
    return (<PyMemberDescrObject*><PyObject*>descr).d_member.offset


_record_items_offset = _slot_offset(Record, '_items')


# ---- D132: the Record of a table row built natively (collect and direct share row_record) ----------
# The fields ``immutable_inputs._record`` sets, written into the slots of a new Record (not yet
# shared, so no lock); the values are the helper's own frozen row objects, so types and values are
# those of ``T.loc[key]``. Used only for root rows-form tables without preserved scalar types and
# without frame metadata (those keep the helper's ``loc``).
from cpython.dict cimport PyDict_New, PyDict_SetItem
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cpython.ref cimport Py_INCREF
cdef extern from "Python.h":
    object PyDictProxy_New(object mapping)

cdef object _object_new = object.__new__
cdef Py_ssize_t _rec_index_offset = _slot_offset(Record, '_index')
cdef Py_ssize_t _rec_groups_offset = _slot_offset(Record, '_native_groups')
cdef Py_ssize_t _rec_metadata_offset = _slot_offset(Record, '_metadata')
cdef Py_ssize_t _rec_preserve_offset = _slot_offset(Record, '_preserve_scalar_types')
cdef Py_ssize_t _rec_frame_row_offset = _slot_offset(Record, '_frame_row')   # -1 when the helper has none
cdef bint _native_records = (_record_items_offset >= 0 and _rec_index_offset >= 0 and _rec_groups_offset >= 0
                             and _rec_metadata_offset >= 0 and _rec_preserve_offset >= 0)


cdef inline void _put_slot(object record, Py_ssize_t offset, object value):
    Py_INCREF(value)
    (<PyObject**>(<char*><PyObject*>record + offset))[0] = <PyObject*>value


cdef class _FrameRowMaker:
    """``frame.loc[label]`` of a frame-built table, made only when a Python caller asks (D119 memo)."""
    cdef object table, label

    def __call__(self):
        from modelx_helpers.immutable_inputs import _frame_of
        return _frame_of(self.table).loc[self.label]


cdef void _native_rows_init(TableHandle t, object owner) except *:
    """Keep ``t._rowrefs`` only where the native Record equals the helper's ``loc[key]`` Record."""
    if t._rowrefs is None:
        return
    metadata = getattr(owner, '_metadata', None)
    if (not _native_records or t._preserve or t._prefix != () or t.base != 0 or len(t._rowrefs) != t.rows
            or (metadata is not None and metadata[0] == 'frame' and metadata[4] is not None)
            or not all(type(values) is tuple for values in t._rowrefs)):
        t._rowrefs = None
        return
    t._frame_owner = None
    if _rec_frame_row_offset >= 0:
        try:
            from modelx_helpers.immutable_inputs import _frame_of
        except ImportError:
            return
        if _frame_of(owner) is not None:
            t._frame_owner = owner


cdef object _native_row_record(TableHandle self, Py_ssize_t row):
    cdef PyObject* values = PyTuple_GET_ITEM(self._rowrefs, row)        # borrowed: held by the handle
    cdef PyObject* columns = <PyObject*>self._columns
    cdef Py_ssize_t j, width = self.width
    cdef _FrameRowMaker maker
    if mx_tuple_size(values) != width:
        mx_handle_fallback()
        return self.helper().loc[_h_label(self, row)]
    items = PyTuple_New(width)
    index = PyDict_New()
    for j in range(width):
        name = <object>mx_tuple_item(columns, j)
        value = <object>mx_tuple_item(values, j)
        pair = (name, value)
        Py_INCREF(pair)
        PyTuple_SET_ITEM(items, j, pair)
        PyDict_SetItem(index, name, value)
    record = _object_new(Record)
    _put_slot(record, _record_items_offset, items)
    _put_slot(record, _rec_index_offset, PyDictProxy_New(index))
    _put_slot(record, _rec_groups_offset, PyDict_New())
    _put_slot(record, _rec_metadata_offset, None)
    _put_slot(record, _rec_preserve_offset, False)
    if self._frame_owner is not None:
        maker = _FrameRowMaker.__new__(_FrameRowMaker)
        maker.table = self._frame_owner
        maker.label = _h_label(self, row)
        _put_slot(record, _rec_frame_row_offset, (maker, None))
    return record


def handle_fallbacks():
    """Number of handle reads that ran Python code (W1.2 debug counter, process-wide)."""
    return mx_handle_fallback_count()


# ---- key parts ------------------------------------------------------------------------

cdef inline int _part_object(HandlePart* q, object k) except -1:
    q.kind = HK_OBJ
    q.boolean = HB_NO
    q.o = <PyObject*>k
    try:
        q.h = hash(k)
    except TypeError:           # unhashable: never a label (the helper raises KeyError too)
        q.kind = HK_NONE
    return 0


cdef inline int _part_double(HandlePart* q, double d, object original, bint spec) except -1:
    q.boolean = HB_NO
    if isnan(d):
        if spec:
            q.kind = HK_MISSING
            q.h = 0x5bd1e995
            return 0
        return _part_object(q, original)
    if d == floor(d) and -9223372036854775808.0 <= d < 9223372036854775808.0:
        q.kind = HK_INT
        q.i = <long long>d
        q.h = mx_mix64(<unsigned long long>q.i)
    else:
        q.kind = HK_FLOAT
        q.f = d
        q.h = mx_hash_double(d)
    return 0


cdef int _part_from(HandlePart* q, object k, bint spec) except -1:
    """Normalise a key part as pandas compares labels: equal numbers match whatever their type
    (int, IntEnum, numpy scalar, integral float), bools carry a flag for the boolean label
    modes, NaN/None/NA/NaT are one missing label under an index spec."""
    cdef int overflow = 0
    cdef long long v
    q.boolean = HB_NO
    if PyLong_Check(k):
        if PyBool_Check(k):
            q.kind = HK_INT
            q.boolean = HB_BOOL
            q.i = 1 if k is True else 0
            q.h = mx_mix64(<unsigned long long>q.i)
            return 0
        v = PyLong_AsLongLongAndOverflow(k, &overflow)
        if overflow:
            q.kind = HK_NONE
            return 0
        q.kind = HK_INT
        q.i = v
        q.h = mx_mix64(<unsigned long long>v)
        return 0
    if PyFloat_Check(k):
        return _part_double(q, PyFloat_AS_DOUBLE(k), k, spec)
    if PyUnicode_Check(k):
        q.kind = HK_STR
        q.o = <PyObject*>k
        q.h = hash(k)
        return 0
    if k is None:
        if spec:
            q.kind = HK_MISSING
            q.h = 0x5bd1e995
            return 0
        return _part_object(q, k)
    if PyObject_TypeCheck(k, _np_bool_type):
        q.kind = HK_INT
        q.boolean = HB_BOOL
        q.i = 1 if PyObject_IsTrue(k) else 0
        q.h = mx_mix64(<unsigned long long>q.i)
        return 0
    if PyObject_TypeCheck(k, _np_integer_type):
        return _part_from(q, PyNumber_Index(k), spec)
    if PyObject_TypeCheck(k, _np_floating_type):
        return _part_double(q, PyFloat_AsDouble(k), k, spec)
    if spec and (k is _pd_na or k is _pd_nat):
        q.kind = HK_MISSING
        q.h = 0x5bd1e995
        return 0
    return _part_object(q, k)


cdef inline void _part_cint(HandlePart* q, long long v) noexcept:
    q.kind = HK_INT
    q.boolean = HB_CINT
    q.i = v
    q.h = mx_mix64(<unsigned long long>v)


cdef inline int _set_part(HandlePart* q, TableKey p, bint spec) except -1:
    if TableKey is object:
        return _part_from(q, p, spec)
    else:
        _part_cint(q, p)
        return 0


cdef inline object _original(TableKey p):
    if TableKey is object:
        return p
    else:
        return PyLong_FromLongLong(p)


cdef inline bint _mode_ok(const HandlePart* q, unsigned char mode) noexcept:
    if mode == 0 or q.kind == HK_MISSING:
        return True
    if q.boolean == HB_CINT:
        return mode == 2 or q.i == 0 or q.i == 1
    if mode == 1:
        return q.boolean == HB_BOOL
    return q.boolean == HB_NO


cdef inline int _part_eq(const HandlePart* a, const HandlePart* q) except -1:
    if a.kind != q.kind or a.h != q.h:
        return 0
    if a.kind == HK_INT:
        return a.i == q.i
    if a.kind == HK_FLOAT:
        return a.f == q.f
    if a.kind == HK_STR:
        return a.o == q.o or mx_unicode_compare(a.o, q.o) == 0
    if a.kind == HK_MISSING:
        return 1
    if a.kind == HK_OBJ:
        if a.o == q.o:
            return 1
        mx_handle_fallback()
        return mx_rich_equal(a.o, q.o, 2)       # Py_EQ
    return 0


cdef inline Py_hash_t _hparts_hash(const HandlePart* parts, Py_ssize_t n) noexcept nogil:
    cdef Py_hash_t h = 0x345678
    cdef Py_ssize_t j
    for j in range(n):
        h = mx_hash_step(h, parts[j].h)
    return mx_mix64(<unsigned long long>h)


cdef Py_ssize_t _keys_find(const HandleKeys* k, const HandlePart* q) except -2:
    """Row (or axis position) of the key q (k.nparts parts) in k; -1 when absent."""
    cdef Py_ssize_t j, lo, hi, mid, r
    cdef unsigned long long offset, pos
    cdef size_t i
    cdef const HandlePart* stored
    cdef long long v
    cdef int same
    for j in range(k.nparts):
        if q[j].kind == HK_NONE or not _mode_ok(&q[j], k.modes[j]):
            return -1
    if k.scheme == TABLE_DENSE:
        if q[0].kind != HK_INT:
            return -1
        # unsigned difference: no signed overflow at +-2**63 or below a negative first key (D15)
        offset = <unsigned long long>q[0].i - <unsigned long long>k.first
        return <Py_ssize_t>offset if offset < <unsigned long long>k.count else -1
    if k.scheme == TABLE_SORTED:
        if q[0].kind != HK_INT:
            return -1
        v = q[0].i
        lo = 0
        hi = k.count
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if k.skeys[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        if lo < k.count and k.skeys[lo] == v:
            return k.srows[lo]
        return -1
    if k.grid != NULL:
        # D86: the dense product index (same rows as the hash index below, which it replaces)
        offset = 0
        for j in range(k.nparts):
            if q[j].kind != HK_INT:
                return -1
            pos = <unsigned long long>q[j].i - <unsigned long long>k.gfirst[j]
            if pos >= k.gspan[j]:
                return -1
            offset = offset * k.gspan[j] + pos
        return k.grid[offset] - 1
    if k.count == 0:
        return -1
    i = <size_t>_hparts_hash(q, k.nparts) & k.mask
    while True:
        r = k.slots[i]
        if r == 0:
            return -1
        r -= 1
        stored = &k.parts[r * k.pstride + k.poff]
        same = 1
        for j in range(k.nparts):
            if not _part_eq(&stored[j], &q[j]):
                same = 0
                break
        if same:
            return r
        i = (i + 1) & k.mask


# ---- construction ------------------------------------------------------------------------

cdef void* _alloc(list hold, Py_ssize_t nbytes) except NULL:
    """Zeroed memory owned by a bytearray in `hold` (never resized, so the pointer stays valid)."""
    cdef object buffer = bytearray(nbytes if nbytes > 0 else 1)
    hold.append(buffer)
    return <void*>PyByteArray_AS_STRING(buffer)


cdef const unsigned char* _modes(list hold, object spec, Py_ssize_t levels, unsigned char default) except NULL:
    cdef unsigned char* modes = <unsigned char*>_alloc(hold, levels)
    cdef Py_ssize_t j
    found = spec['boolean_modes'] if spec is not None else ()
    for j in range(levels):
        modes[j] = <unsigned char>found[j] if j < len(found) else default
    return modes


cdef void _hash_index(HandleKeys* k, list hold) except *:
    """Open-addressing slots over k's stored keys (unique, as table keys are)."""
    cdef size_t size = 8, i
    cdef Py_ssize_t r
    cdef Py_ssize_t* slots
    while size < <size_t>(2 * k.count):
        size <<= 1
    slots = <Py_ssize_t*>_alloc(hold, <Py_ssize_t>(size * sizeof(Py_ssize_t)))
    for r in range(k.count):
        i = <size_t>_hparts_hash(&k.parts[r * k.pstride + k.poff], k.nparts) & (size - 1)
        while slots[i] != 0:
            i = (i + 1) & (size - 1)
        slots[i] = r + 1
    k.slots = slots
    k.mask = size - 1


# D86: the largest dense product index built for a multi-part key (slots of one Py_ssize_t), the default
# of the compiler's max_dense_entries (D74); a larger box keeps the hash index.
GRID_MAX_SLOTS = 65536


cdef str _grid_index(HandleKeys* k, list hold, Py_ssize_t max_slots):
    """D86: build k.grid over a HASHED key of two or more parts when every stored part is a plain int
    (no bool label, no level that matches bools only) and the key box holds at most max_slots positions.
    Duplicate keys keep the first row, as the hash index finds it. Returns '' when built, else why not."""
    cdef Py_ssize_t r, j, n = k.nparts
    cdef const HandlePart* part
    cdef long long* lo
    cdef unsigned long long* span
    cdef unsigned long long size = 1, offset
    cdef Py_ssize_t* grid
    if n < 2 or k.count == 0:
        return 'fewer than two key parts or no rows'
    for j in range(n):
        if k.modes[j] == 1:
            return 'a level matches bool labels only'
    lo = <long long*>_alloc(hold, n * sizeof(long long))
    hi = [0] * n
    for r in range(k.count):
        for j in range(n):
            part = &k.parts[r * k.pstride + k.poff + j]
            if part.kind != HK_INT or part.boolean != HB_NO:
                return 'a key part is not a plain integer'
            if r == 0 or part.i < lo[j]:
                lo[j] = part.i
            if r == 0 or part.i > hi[j]:
                hi[j] = part.i
    span = <unsigned long long*>_alloc(hold, n * sizeof(unsigned long long))
    for j in range(n):
        if hi[j] - lo[j] + 1 > max_slots:
            return f'key box above {max_slots:,} positions'
        span[j] = <unsigned long long>(hi[j] - lo[j] + 1)
        size *= span[j]
        if size > <unsigned long long>max_slots:
            return f'key box above {max_slots:,} positions'
    grid = <Py_ssize_t*>_alloc(hold, <Py_ssize_t>(size * sizeof(Py_ssize_t)))
    for r in range(k.count):
        offset = 0
        for j in range(n):
            offset = offset * span[j] + (<unsigned long long>k.parts[r * k.pstride + k.poff + j].i
                                         - <unsigned long long>lo[j])
        if grid[offset] == 0:
            grid[offset] = r + 1
    k.gfirst = lo
    k.gspan = span
    k.grid = grid
    return ''


cdef void _index_single(TableHandle t, HandleKeys* k, list hold, const HandlePart* parts,
                        Py_ssize_t count, Py_ssize_t pstride, Py_ssize_t poff,
                        const unsigned char* modes, object labels) except *:
    """A one-level key index: DENSE when the keys are consecutive ascending ints, SORTED for
    other ints, HASHED otherwise. Sets t's int-key facts and index extremes (as the helper's
    index.min()/max(): the smallest and largest label)."""
    cdef Py_ssize_t r
    cdef bint ints = True, increasing = True
    cdef long long* okeys
    cdef long long* skeys
    cdef Py_ssize_t* srows
    k.count = count
    k.nparts = 1
    k.pstride = pstride
    k.poff = poff
    k.parts = parts
    k.modes = modes
    for r in range(count):
        if parts[r * pstride + poff].kind != HK_INT or parts[r * pstride + poff].boolean != HB_NO:
            ints = False
            break
    t._int_keys = ints and count > 0
    if t._int_keys:
        okeys = <long long*>_alloc(hold, count * sizeof(long long))
        for r in range(count):
            okeys[r] = parts[r * pstride + poff].i
            if r and okeys[r] <= okeys[r - 1]:
                increasing = False
        t._okeys = okeys
        t._increasing = increasing
        if increasing and <unsigned long long>okeys[count - 1] - <unsigned long long>okeys[0] == <unsigned long long>(count - 1):
            k.scheme = TABLE_DENSE
            k.first = okeys[0]
            k.last = okeys[count - 1]
        else:
            k.scheme = TABLE_SORTED
            values = [okeys[r] for r in range(count)]
            order = sorted(range(count), key=values.__getitem__)
            skeys = <long long*>_alloc(hold, count * sizeof(long long))
            srows = <Py_ssize_t*>_alloc(hold, count * sizeof(Py_ssize_t))
            for r in range(count):
                srows[r] = order[r]
                skeys[r] = okeys[srows[r]]
            k.skeys = skeys
            k.srows = srows
            k.first = skeys[0]
            k.last = skeys[count - 1]
        t._first_obj = PyLong_FromLongLong(k.first)
        t._last_obj = PyLong_FromLongLong(k.last)
    else:
        k.scheme = TABLE_HASHED
        _hash_index(k, hold)
        t._first_obj, t._last_obj = _extremes(labels)


cdef tuple _extremes(object labels):
    """(min, max) of the labels as the helper's index.min()/max(); NaN when empty."""
    if not len(labels):
        return float('nan'), float('nan')
    try:
        return min(labels), max(labels)
    except TypeError:
        return _no_extreme, _no_extreme


cdef HandlePart* _label_parts(list hold, object labels, bint spec) except NULL:
    cdef Py_ssize_t r, count = len(labels)
    cdef HandlePart* parts = <HandlePart*>_alloc(hold, count * sizeof(HandlePart))
    hold.append(labels)                 # the parts borrow str and object labels
    for r in range(count):
        _part_from(&parts[r], labels[r], spec)
    return parts


cdef TableHandle _empty_like(TableHandle source):
    """A new handle sharing source's data arrays and identity (keys and children unset)."""
    cdef TableHandle t = TableHandle.__new__(TableHandle)
    t.owner = source.owner
    t._hold = source._hold
    t._prefix = source._prefix
    t._columns = source._columns
    t.width = source.width
    t._ckind = source._ckind
    t._dbl = source._dbl
    t._int = source._int
    t._obj = source._obj
    t._array = source._array
    t._stride = source._stride
    t._colbase = source._colbase
    t._spec_missing = source._spec_missing
    t._preserve = source._preserve
    t._first_obj = _no_extreme
    t._last_obj = _no_extreme
    t._children = ()
    return t


cdef TableHandle _clone(TableHandle source):
    cdef TableHandle t = _empty_like(source)
    t.scheme = source.scheme
    t.rows = source.rows
    t.levels = source.levels
    t.base = source.base
    t._children = source._children
    t._labels = source._labels
    t._first_obj = source._first_obj
    t._rowrefs = source._rowrefs
    t._frame_owner = source._frame_owner
    t._last_obj = source._last_obj
    t._keys = source._keys
    t._ckeys = source._ckeys
    t._axes = source._axes
    t._spans = source._spans
    t._int_keys = source._int_keys
    t._increasing = source._increasing
    t._okeys = source._okeys
    return t


cdef void _product_children(TableHandle t, list hold, tuple axes, list axis_handles) except *:
    """Eager depth-1 children of a PRODUCT handle: by position on its first axis."""
    cdef Py_ssize_t position
    cdef TableHandle child, axis
    cdef list children = []
    for position in range(len(axes[0])):
        child = _empty_like(t)
        child._prefix = t._prefix + (axes[0][position],)
        child.base = t.base + position * t._spans[0]
        child.rows = t._spans[0]
        child.levels = t.levels - 1
        if child.levels > 1:
            child.scheme = TABLE_PRODUCT
            child._axes = t._axes + 1
            child._spans = t._spans + 1
            child._labels = axes[1:]
            child._first_obj, child._last_obj = _product_extremes(axes[1:])
            _product_children(child, hold, axes[1:], axis_handles[1:])
        else:
            axis = <TableHandle>axis_handles[1]
            child.scheme = axis._keys.scheme
            child._keys = axis._keys
            child._labels = axis._labels
            child._int_keys = axis._int_keys
            child._increasing = axis._increasing
            child._okeys = axis._okeys
            child._first_obj = axis._first_obj
            child._last_obj = axis._last_obj
        children.append(child)
    t._children = tuple(children)


cdef tuple _product_extremes(tuple axes):
    try:
        return tuple(min(axis) for axis in axes), tuple(max(axis) for axis in axes)
    except (TypeError, ValueError):
        return _no_extreme, _no_extreme


cdef void _rows_children(TableHandle t, list hold, HandlePart* parts, Py_ssize_t levels,
                         Py_ssize_t depth, tuple labels, const unsigned char* modes) except *:
    """Eager children of a multi-level row-stored handle at key depth `depth`: rows are laid out
    so that each prefix is one run, so a child is the run [start, start + count) of t."""
    cdef Py_ssize_t start = 0, end, r, count
    cdef TableHandle child
    cdef list children = []
    cdef list firsts = []
    cdef HandlePart* heads
    cdef const HandlePart* row_parts = parts + t.base * levels
    while start < t.rows:
        end = start + 1
        while end < t.rows and _part_eq(&row_parts[end * levels + depth],
                                         &row_parts[start * levels + depth]):
            end += 1
        count = end - start
        child = _empty_like(t)
        child._prefix = t._prefix + (labels[t.base + start][depth],)
        child.base = t.base + start
        child.rows = count
        child.levels = t.levels - 1
        if child.levels > 1:
            child.scheme = TABLE_HASHED
            child._keys.scheme = TABLE_HASHED
            child._keys.count = count
            child._keys.nparts = child.levels
            child._keys.pstride = levels
            child._keys.poff = depth + 1
            child._keys.parts = parts + child.base * levels
            child._keys.modes = modes + depth + 1
            _hash_index(&child._keys, hold)
            child._labels = tuple(labels[child.base + r][depth + 1:] for r in range(count))
            child._first_obj, child._last_obj = _extremes(child._labels)
            _rows_children(child, hold, parts, levels, depth + 1, labels, modes)
        else:
            child._labels = tuple(labels[child.base + r][depth + 1] for r in range(count))
            _index_single(child, &child._keys, hold, parts + child.base * levels, count,
                          levels, depth + 1, modes + depth + 1, child._labels)
            child.scheme = child._keys.scheme
        children.append(child)
        firsts.append(start)
        start = end
    t._children = tuple(children)
    heads = <HandlePart*>_alloc(hold, len(firsts) * sizeof(HandlePart))
    for r in range(len(firsts)):
        heads[r] = row_parts[<Py_ssize_t>firsts[r] * levels + depth]
    t._ckeys.scheme = TABLE_HASHED
    t._ckeys.count = len(firsts)
    t._ckeys.nparts = 1
    t._ckeys.pstride = 1
    t._ckeys.poff = 0
    t._ckeys.parts = heads
    t._ckeys.modes = modes + depth
    _hash_index(&t._ckeys, hold)


cdef object _canonical(const HandlePart* q, object label):
    """A Python value equal for exactly the parts _part_eq calls equal (for grouping)."""
    if q.kind == HK_INT:
        return (1, q.i)
    if q.kind == HK_FLOAT:
        return (2, q.f)
    if q.kind == HK_STR:
        return (3, <object>q.o)
    if q.kind == HK_MISSING:
        return (4,)
    return (5, label)


cdef list _hierarchical_order(list rows, Py_ssize_t depth, Py_ssize_t levels, list canon):
    """Rows grouped by part `depth` in first-appearance order, recursively: every key prefix
    becomes one run; rows keep their relative order within a run."""
    cdef dict groups
    cdef list result
    if depth >= levels - 1 or len(rows) < 2:
        return rows
    groups = {}
    for r in rows:
        groups.setdefault(canon[r][depth], []).append(r)
    if len(groups) == 1:
        return _hierarchical_order(rows, depth + 1, levels, canon)
    result = []
    for group in groups.values():
        result.extend(_hierarchical_order(group, depth + 1, levels, canon))
    return result


cdef void _row_columns(TableHandle t, list hold, tuple values, list order, Py_ssize_t width) except *:
    """Typed column arrays of a row-stored table in handle row order: a float, int or bool
    column gets a C array; every cell also stays available as the stored object."""
    cdef Py_ssize_t rows = len(order), r, c, i
    cdef unsigned char* kinds = <unsigned char*>_alloc(hold, width)
    cdef double* numbers = NULL
    cdef long long* integers = NULL
    cdef int overflow = 0
    cdef bint any_float = False, any_int = False
    cdef unsigned char kind, this
    for c in range(width):
        kind = HC_F64
        for r in range(rows):
            value = values[<Py_ssize_t>order[r]][c]
            if type(value) is float:
                this = HC_F64
            elif type(value) is bool:
                this = HC_BOOL
            elif type(value) is int:
                PyLong_AsLongLongAndOverflow(value, &overflow)
                this = HC_OBJ if overflow else HC_I64
            else:
                this = HC_OBJ
            if r == 0:
                kind = this
            elif this != kind:
                kind = HC_OBJ
            if kind == HC_OBJ:
                break
        kinds[c] = kind
        any_float = any_float or kind == HC_F64
        any_int = any_int or kind == HC_I64 or kind == HC_BOOL
    if any_float:
        numbers = <double*>_alloc(hold, rows * width * sizeof(double))
    if any_int:
        integers = <long long*>_alloc(hold, rows * width * sizeof(long long))
    flat = tuple([values[<Py_ssize_t>order[r]][c] for r in range(rows) for c in range(width)])
    hold.append(flat)
    for r in range(rows):
        for c in range(width):
            i = r * width + c
            value = flat[i]
            if kinds[c] == HC_F64:
                numbers[i] = PyFloat_AS_DOUBLE(value)
            elif kinds[c] == HC_I64 or kinds[c] == HC_BOOL:
                integers[i] = PyLong_AsLongLongAndOverflow(value, &overflow)
    t._ckind = kinds
    t._dbl = numbers
    t._int = integers
    t._obj = PySequence_Fast_ITEMS(flat) if rows and width else NULL
    t._stride = width


cdef void _init_contiguous(TableHandle t, list hold, tuple storage) except *:
    cdef Py_ssize_t rows, c
    cdef const double[:, ::1] numbers
    cdef const long long[:, ::1] integers
    cdef unsigned char* kinds
    cdef HandlePart* parts
    start, data, spec = storage[1], storage[2], storage[4]
    rows = len(data)
    if not -(1 << 63) <= start <= start + rows - 1 < (1 << 63):
        raise OverflowError('contiguous table keys exceed 64-bit integers')
    t.levels = 1
    t.rows = rows
    kinds = <unsigned char*>_alloc(hold, t.width)
    if data.dtype == _np_float64 and data.ndim == 2:
        array = _group_numpy.ascontiguousarray(data)
        hold.append(array)
        numbers = array
        t._dbl = &numbers[0, 0] if rows else NULL
        kind = HC_F64
    elif data.dtype == _np_int64 and data.ndim == 2:
        array = _group_numpy.ascontiguousarray(data)
        hold.append(array)
        integers = array
        t._int = &integers[0, 0] if rows else NULL
        kind = HC_I64
    else:
        kind = HC_OBJ
        t._array = data
    for c in range(t.width):
        kinds[c] = kind
    if t._preserve:
        t._array = data             # obj() returns the stored numpy scalar, as the helper
    t._ckind = kinds
    t._stride = t.width
    # DENSE keys start .. start + rows - 1; contiguous tables reject bool keys (helper _find_row)
    t.scheme = TABLE_DENSE
    t._keys.scheme = TABLE_DENSE
    t._keys.count = rows
    t._keys.nparts = 1
    t._keys.first = start
    t._keys.last = start + rows - 1
    t._keys.modes = _modes(hold, spec, 1, 2)
    t._int_keys = rows > 0
    t._increasing = True
    t._first_obj = start
    t._last_obj = start + rows - 1


cdef void _init_rectangular(TableHandle t, list hold, tuple storage) except *:
    cdef Py_ssize_t levels, j
    cdef const double[::1] numbers
    cdef unsigned char* kinds
    cdef HandleKeys* axes_keys
    cdef Py_ssize_t* spans
    cdef TableHandle axis
    cdef const unsigned char* modes
    axes, data, spec = tuple(storage[1]), storage[2], storage[4]
    levels = len(axes)
    t.levels = levels
    array = _group_numpy.ascontiguousarray(data, dtype=_np_float64).reshape(-1)
    hold.append(array)
    numbers = array
    t._dbl = &numbers[0] if len(array) else NULL
    kinds = <unsigned char*>_alloc(hold, 1)
    kinds[0] = HC_F64
    t._ckind = kinds
    t._stride = 1
    t.rows = len(array)
    modes = _modes(hold, spec, levels, 0)
    axes_keys = <HandleKeys*>_alloc(hold, levels * sizeof(HandleKeys))
    spans = <Py_ssize_t*>_alloc(hold, levels * sizeof(Py_ssize_t))
    axis_handles = []
    for j in range(levels):
        labels = tuple(axes[j])
        axis = TableHandle.__new__(TableHandle)
        axis._labels = labels
        _index_single(axis, &axis._keys, hold, _label_parts(hold, labels, t._spec_missing),
                      len(labels), 1, 0, modes + j, labels)
        axes_keys[j] = axis._keys
        axis_handles.append(axis)
    hold.append(tuple(axis_handles))
    spans[levels - 1] = 1
    for j in range(levels - 2, -1, -1):
        spans[j] = spans[j + 1] * len(axes[j + 1])
    t.scheme = TABLE_PRODUCT
    t._axes = axes_keys
    t._spans = spans
    t._labels = axes
    t._first_obj, t._last_obj = _product_extremes(axes)
    _product_children(t, hold, axes, axis_handles)


cdef void _init_rows(TableHandle t, list hold, tuple storage) except *:
    cdef Py_ssize_t levels, rows, r, j
    cdef HandlePart* original
    cdef HandlePart* parts
    cdef const unsigned char* modes
    keys, values, spec = storage[1], storage[2], storage[5]
    rows = len(keys)
    if spec is not None:
        levels = len(spec['boolean_modes']) if spec['multi'] else 1
    else:
        levels = len(keys[0]) if keys and type(keys[0]) is tuple else 1
    t.levels = levels
    t.rows = rows
    modes = _modes(hold, spec, levels, 0)
    if levels == 1:
        _row_columns(t, hold, values, list(range(rows)), t.width)
        t._labels = tuple(keys)
        t._rowrefs = tuple(values)
        _index_single(t, &t._keys, hold, _label_parts(hold, t._labels, t._spec_missing),
                      rows, 1, 0, modes, t._labels)
        t.scheme = t._keys.scheme
        return
    hold.append(keys)
    original = <HandlePart*>_alloc(hold, rows * levels * sizeof(HandlePart))
    canon = []
    for r in range(rows):
        key = keys[r]
        if type(key) is not tuple or len(key) != levels:
            raise TypeError('multi-level table key is not a tuple of one part per level')
        for j in range(levels):
            _part_from(&original[r * levels + j], key[j], t._spec_missing)
        canon.append(tuple(_canonical(&original[r * levels + j], key[j]) for j in range(levels)))
    order = _hierarchical_order(list(range(rows)), 0, levels, canon)
    parts = <HandlePart*>_alloc(hold, rows * levels * sizeof(HandlePart))
    for r in range(rows):
        memcpy(&parts[r * levels], &original[<Py_ssize_t>order[r] * levels], levels * sizeof(HandlePart))
    _row_columns(t, hold, values, order, t.width)
    t._labels = tuple([keys[<Py_ssize_t>order[r]] for r in range(rows)])
    t._rowrefs = tuple([values[<Py_ssize_t>order[r]] for r in range(rows)])
    t.scheme = TABLE_HASHED
    t._keys.scheme = TABLE_HASHED
    t._keys.count = rows
    t._keys.nparts = levels
    t._keys.pstride = levels
    t._keys.poff = 0
    t._keys.parts = parts
    t._keys.modes = modes
    _hash_index(&t._keys, hold)
    why = _grid_index(&t._keys, hold, GRID_MAX_SLOTS)
    t._key_index = 'grid' if not why else 'hashed: ' + why
    t._first_obj, t._last_obj = _extremes(keys)
    _rows_children(t, hold, parts, levels, 0, t._labels, modes)


cdef double _object_f64(PyObject* value) except? -1:
    """float(value) for a stored cell object."""
    if PyFloat_Check(<object>value):
        return PyFloat_AS_DOUBLE(<object>value)
    if PyLong_Check(<object>value):
        return PyLong_AsDouble(<object>value)
    mx_handle_fallback()
    return PyFloat_AS_DOUBLE(PyNumber_Float(<object>value))


cdef long long _double_i64(double value) except? -1:
    """int(value) for a float cell, in a C long long."""
    if isnan(value):
        raise ValueError('cannot convert float NaN to integer')
    if isinf(value):
        raise OverflowError('cannot convert float infinity to integer')
    if not -9223372036854775808.0 <= value < 9223372036854775808.0:
        raise OverflowError('Python int too large to convert to C long long')
    return <long long>value


cdef long long _object_i64(PyObject* value) except? -1:
    cdef int overflow = 0
    cdef long long result
    if PyLong_Check(<object>value):
        result = PyLong_AsLongLongAndOverflow(<object>value, &overflow)
    elif PyFloat_Check(<object>value):
        return _double_i64(PyFloat_AS_DOUBLE(<object>value))
    else:
        mx_handle_fallback()
        result = PyLong_AsLongLongAndOverflow(PyNumber_Long(<object>value), &overflow)
    if overflow:
        raise OverflowError('Python int too large to convert to C long long')
    return result


cdef long long _h_extreme_part(TableHandle self, PyObject* key, Py_ssize_t k, bint last) except? -1:
    """Part ``k`` of the smallest (largest when ``last``) key as a C integer (diag-run8). The held
    extreme tuple and its part are read as borrowed pointers, so neither the shared key nor its
    part gains a reference; without one (empty table, no held extreme) the key comes from the
    helper as in ``first_key()[k]`` / ``last_key()[k]`` and converts like a C integer cast of it."""
    cdef PyObject* value
    if mx_tuple_check_exact(key) and 0 <= k < mx_tuple_size(key):
        value = mx_tuple_item(key, k)
        if mx_long_check_exact(value):
            return _object_i64(value)
    found = (self.last_key() if last else self.first_key())[k]
    return <long long>found


cdef inline Py_ssize_t _h_cell(TableHandle self, Py_ssize_t row, Py_ssize_t col) except -1:
    if <size_t>row >= <size_t>self.rows or <size_t>col >= <size_t>self.width:
        raise IndexError(f'table cell ({row}, {col}) is out of range')
    return (self.base + row) * self._stride + self._colbase + col

cdef Py_ssize_t _h_row(TableHandle self, const HandlePart* q, Py_ssize_t n) except -2:
    cdef Py_ssize_t j, position, row = 0
    if n != self.levels:
        return -1
    if self.scheme == TABLE_PRODUCT:
        for j in range(n):
            position = _keys_find(&self._axes[j], &q[j])
            if position < 0:
                return -1
            row += position * self._spans[j]
        return row
    return _keys_find(&self._keys, q)

cdef PyObject* _h_child(PyObject* parent, const HandlePart* q) except? NULL:
    """Borrowed child of `parent` for one prefix part (NULL when absent): the walk takes no
    reference on the shared handles; the root keeps every child alive."""
    cdef Py_ssize_t position
    if (<TableHandle>parent).levels < 2:
        return NULL
    if (<TableHandle>parent).scheme == TABLE_PRODUCT:
        position = _keys_find(&(<TableHandle>parent)._axes[0], q)
    else:
        position = _keys_find(&(<TableHandle>parent)._ckeys, q)
    if position < 0:
        return NULL
    return mx_tuple_item(<PyObject*>(<TableHandle>parent)._children, position)


cdef TableHandle _h_sub(TableHandle self, const HandlePart* q, Py_ssize_t n):
    cdef PyObject* handle = <PyObject*>self
    cdef Py_ssize_t j
    if n >= self.levels:
        raise TypeError(f'sub_{n} needs fewer key parts than the {self.levels} levels of the table')
    for j in range(n):
        handle = _h_child(handle, &q[j])
        if handle == NULL:
            return None
    if (<TableHandle>handle)._colbase != self._colbase or (<TableHandle>handle).width != self.width:
        return _column_of(<TableHandle>handle, self._colbase, self._columns)
    return <TableHandle>handle        # the one reference a sub-table read takes

cdef bint _h_has(TableHandle self, const HandlePart* q, Py_ssize_t n) except -1:
    cdef PyObject* handle = <PyObject*>self
    cdef Py_ssize_t j
    if n == self.levels:
        return _h_row(self, q, n) >= 0
    if n > self.levels or n == 0:
        return False
    if self.scheme == TABLE_PRODUCT:
        for j in range(n):
            if _keys_find(&self._axes[j], &q[j]) < 0:
                return False
        return True
    for j in range(n):
        handle = _h_child(handle, &q[j])
        if handle == NULL:
            return False
    return True

cdef int _h_unpack(TableHandle self, object key, HandlePart* q) except -1:
    """A whole tuple key passed as one part of a multi-level table: its parts, else 0."""
    cdef Py_ssize_t j, n
    if not (PyTuple_Check(key) and self.levels > 1):
        return 0
    n = PyTuple_GET_SIZE(key)
    if n == 0 or n > 6:
        return 0
    for j in range(n):
        _part_from(&q[j], <object>PyTuple_GET_ITEM(key, j), self._spec_missing)
    return <int>n

cdef int _h_int_index(TableHandle self, str what) except -1:
    if self.levels != 1 or not self._int_keys:
        raise TypeError(f'{what} needs a one-level table with integer keys')
    return 0

cdef object _h_label(TableHandle self, Py_ssize_t row):
    cdef Py_ssize_t j, position
    if self.scheme == TABLE_PRODUCT:
        parts = []
        for j in range(self.levels):
            position = (row // self._spans[j]) % self._axes[j].count
            parts.append(self._labels[j][position])
        return tuple(parts)
    if self._labels is None:
        return self._keys.first + row
    return self._labels[row]

cdef inline PyObject* _h_value(RecordHandle self, Py_ssize_t slot) except NULL:
    cdef PyObject* items = <PyObject*>self._items     # borrowed: no reference on the shared pairs
    if <size_t>slot >= <size_t>mx_tuple_size(items):
        raise IndexError(f'record slot {slot} is out of range')
    return mx_tuple_item(mx_tuple_item(items, slot), 1)


@cython.final
cdef class TableHandle:
    """Typed read handle of one helper Table (or of one of its sub-tables): see the pxd."""

    def __init__(self, owner):
        cdef list hold = []
        if type(owner) is NativeTable:
            owner = (<NativeTable>owner)._owner
        if _table_storage_fn is None:
            raise TypeError('typed table handles need modelx_helpers.immutable_inputs.table_storage')
        storage = tuple(_table_storage_fn(owner))
        self.owner = owner
        self._hold = hold
        self._prefix = ()
        self._columns = tuple(storage[-2])
        self.width = len(self._columns)
        self._spec_missing = storage[-1] is not None
        self._preserve = bool(getattr(owner, '_preserve_scalar_types', False))
        self._first_obj = _no_extreme
        self._last_obj = _no_extreme
        self._children = ()
        if storage[0] == 'contiguous':
            _init_contiguous(self, hold, storage)
        elif storage[0] == 'rectangular':
            _init_rectangular(self, hold, storage)
        elif storage[0] == 'rows':
            _init_rows(self, hold, storage)
        else:
            raise TypeError(f'unknown table storage {storage[0]!r}')
        _native_rows_init(self, owner)
        _attach_table_objects(self)

    # -- columns and cells ------------------------------------------------------------------

    cdef Py_ssize_t col(self, object name) except -2:
        # borrowed: the handle owns its column tuple; a shared handle is read by every worker on
        # every per-step read, so no reference count of the tuple is touched here (diag-4w)
        cdef PyObject* columns = <PyObject*>self._columns
        cdef Py_ssize_t j
        for j in range(self.width):
            if <PyObject*>name == mx_tuple_item(columns, j):
                return j
        if PyUnicode_Check(name):
            for j in range(self.width):
                if mx_unicode_compare(<PyObject*>name, mx_tuple_item(columns, j)) == 0:
                    return j
        raise KeyError(name)


    cdef double f64(self, Py_ssize_t row, Py_ssize_t col) except? -1:
        cdef Py_ssize_t i = _h_cell(self, row, col)
        cdef unsigned char kind = self._ckind[self._colbase + col]
        if kind == HC_F64:
            return self._dbl[i]
        if kind == HC_I64 or kind == HC_BOOL:
            return <double>self._int[i]
        if self._obj != NULL:
            return _object_f64(self._obj[i])
        mx_handle_fallback()
        return float(self._array[self.base + row, self._colbase + col])

    cdef long long i64(self, Py_ssize_t row, Py_ssize_t col) except? -1:
        cdef Py_ssize_t i = _h_cell(self, row, col)
        cdef unsigned char kind = self._ckind[self._colbase + col]
        if kind == HC_I64 or kind == HC_BOOL:
            return self._int[i]
        if kind == HC_F64:
            return _double_i64(self._dbl[i])
        if self._obj != NULL:
            return _object_i64(self._obj[i])
        mx_handle_fallback()
        return int(self._array[self.base + row, self._colbase + col])

    cdef object obj(self, Py_ssize_t row, Py_ssize_t col):
        cdef Py_ssize_t i = _h_cell(self, row, col)
        cdef unsigned char kind = self._ckind[self._colbase + col]
        if self._obj != NULL:
            return <object>self._obj[i]
        if self._array is not None:
            mx_handle_fallback()
            if self._preserve:
                return self._array[self.base + row, self._colbase + col]
            return self._array.item(self.base + row, self._colbase + col)
        if kind == HC_F64:
            return PyFloat_FromDouble(self._dbl[i])
        if kind == HC_BOOL:
            return self._int[i] != 0
        return PyLong_FromLongLong(self._int[i])

    # -- keys -----------------------------------------------------------------------------


    cdef Py_ssize_t row_1(self, TableKey p0) except -2:
        cdef HandlePart q[6]
        cdef Py_ssize_t row, n = 0
        if TableKey is object:
            n = _h_unpack(self, p0, q)
        if n == 0:
            _set_part(&q[0], p0, self._spec_missing)
            n = 1
        row = _h_row(self, q, n)
        if row < 0:
            raise KeyError(_original(p0))
        return row

    cdef Py_ssize_t row_2(self, TableKey p0, TableKey p1) except -2:
        cdef HandlePart q[2]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        row = _h_row(self, q, 2)
        if row < 0:
            raise KeyError((_original(p0), _original(p1)))
        return row

    cdef Py_ssize_t row_3(self, TableKey p0, TableKey p1, TableKey p2) except -2:
        cdef HandlePart q[3]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        row = _h_row(self, q, 3)
        if row < 0:
            raise KeyError((_original(p0), _original(p1), _original(p2)))
        return row

    cdef Py_ssize_t row_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except -2:
        cdef HandlePart q[4]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        row = _h_row(self, q, 4)
        if row < 0:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3)))
        return row

    cdef Py_ssize_t row_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4) except -2:
        cdef HandlePart q[5]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        row = _h_row(self, q, 5)
        if row < 0:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3),
                            _original(p4)))
        return row

    cdef Py_ssize_t row_6(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4, TableKey p5) except -2:
        cdef HandlePart q[6]
        cdef Py_ssize_t row
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        _set_part(&q[5], p5, self._spec_missing)
        row = _h_row(self, q, 6)
        if row < 0:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3),
                            _original(p4), _original(p5)))
        return row

    cdef bint has_1(self, TableKey p0) except -1:
        cdef HandlePart q[6]
        cdef Py_ssize_t n = 0
        if TableKey is object:
            n = _h_unpack(self, p0, q)
        if n == 0:
            _set_part(&q[0], p0, self._spec_missing)
            n = 1
        return _h_has(self, q, n)

    cdef bint has_2(self, TableKey p0, TableKey p1) except -1:
        cdef HandlePart q[2]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        return _h_has(self, q, 2)

    cdef bint has_3(self, TableKey p0, TableKey p1, TableKey p2) except -1:
        cdef HandlePart q[3]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        return _h_has(self, q, 3)

    cdef bint has_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except -1:
        cdef HandlePart q[4]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        return _h_has(self, q, 4)

    cdef bint has_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4) except -1:
        cdef HandlePart q[5]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        return _h_has(self, q, 5)

    cdef bint has_6(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4, TableKey p5) except -1:
        cdef HandlePart q[6]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        _set_part(&q[5], p5, self._spec_missing)
        return _h_has(self, q, 6)

    cdef TableHandle sub_1(self, TableKey p0):
        cdef HandlePart q[1]
        _set_part(&q[0], p0, self._spec_missing)
        child = _h_sub(self, q, 1)
        if child is None:
            raise KeyError(_original(p0))
        return child

    cdef TableHandle sub_2(self, TableKey p0, TableKey p1):
        cdef HandlePart q[2]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        child = _h_sub(self, q, 2)
        if child is None:
            raise KeyError((_original(p0), _original(p1)))
        return child

    cdef TableHandle sub_3(self, TableKey p0, TableKey p1, TableKey p2):
        cdef HandlePart q[3]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        child = _h_sub(self, q, 3)
        if child is None:
            raise KeyError((_original(p0), _original(p1), _original(p2)))
        return child

    cdef TableHandle sub_4(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3):
        cdef HandlePart q[4]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        child = _h_sub(self, q, 4)
        if child is None:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3)))
        return child

    cdef TableHandle sub_5(self, TableKey p0, TableKey p1, TableKey p2, TableKey p3, TableKey p4):
        cdef HandlePart q[5]
        _set_part(&q[0], p0, self._spec_missing)
        _set_part(&q[1], p1, self._spec_missing)
        _set_part(&q[2], p2, self._spec_missing)
        _set_part(&q[3], p3, self._spec_missing)
        _set_part(&q[4], p4, self._spec_missing)
        child = _h_sub(self, q, 5)
        if child is None:
            raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3),
                            _original(p4)))
        return child

    # -- index ------------------------------------------------------------------------------


    cdef long long first_int(self) except? -1:
        if self.rows == 0:
            raise ValueError('cannot convert float NaN to integer')
        _h_int_index(self, 'index.min()')
        return self._keys.first

    cdef long long last_int(self) except? -1:
        if self.rows == 0:
            raise ValueError('cannot convert float NaN to integer')
        _h_int_index(self, 'index.max()')
        return self._keys.last

    cdef object first_key(self):
        if self._first_obj is _no_extreme:
            mx_handle_fallback()
            return self.helper().index.min()
        return self._first_obj

    cdef object last_key(self):
        if self._last_obj is _no_extreme:
            mx_handle_fallback()
            return self.helper().index.max()
        return self._last_obj

    cdef long long first_part_int(self, Py_ssize_t k) except? -1:
        return _h_extreme_part(self, <PyObject*>self._first_obj, k, False)

    cdef long long last_part_int(self, Py_ssize_t k) except? -1:
        return _h_extreme_part(self, <PyObject*>self._last_obj, k, True)

    cdef Py_ssize_t asof_row(self, long long k) except -2:
        """Row of index.asof(k): the last key <= k; KeyError(k) below the first key (D25a)."""
        cdef Py_ssize_t lo = 0, hi, mid
        if self.levels == 1 and self._int_keys:
            if self._keys.scheme == TABLE_DENSE:
                if k >= self._keys.last:
                    return self.rows - 1
                if k < self._keys.first:
                    raise KeyError(k)
                return <Py_ssize_t>(<unsigned long long>k - <unsigned long long>self._keys.first)
            if self._increasing:
                hi = self.rows
                while lo < hi:
                    mid = lo + (hi - lo) // 2
                    if k < self._okeys[mid]:
                        hi = mid
                    else:
                        lo = mid + 1
                if lo == 0:
                    raise KeyError(k)
                return lo - 1
        mx_handle_fallback()        # labels not in increasing order: the helper (pandas) answers
        return self.row_1(<object>self.helper().index.asof(k))

    cdef long long asof_int(self, long long k) except? -1:
        cdef Py_ssize_t row
        if self.levels == 1 and self._int_keys:
            if self._keys.scheme == TABLE_DENSE:
                if k >= self._keys.last:
                    return self._keys.last
                if k < self._keys.first:
                    raise KeyError(k)
                return k
            if self._increasing:
                return self._okeys[self.asof_row(k)]
        mx_handle_fallback()
        return self.helper().index.asof(k)

    cdef Py_ssize_t searchsorted_right(self, long long k) noexcept:
        """index.searchsorted(k, side="right") on a one-level int index; -1 when not applicable
        (the read is then not typed). Bisects the keys in row order, as the helper."""
        cdef Py_ssize_t lo = 0, hi, mid
        if self.levels != 1 or not self._int_keys:
            return -1
        if self._keys.scheme == TABLE_DENSE:
            if k < self._keys.first:
                return 0
            if k >= self._keys.last:
                return self.rows
            return <Py_ssize_t>(<unsigned long long>k - <unsigned long long>self._keys.first) + 1
        hi = self.rows
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if k < self._okeys[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    cdef long long key_int(self, Py_ssize_t i) except? -1:
        cdef Py_ssize_t position = i + self.rows if i < 0 else i
        _h_int_index(self, 'index[i]')
        if not 0 <= position < self.rows:
            raise IndexError(f'index {i} is out of bounds for axis 0 with size {self.rows}')
        if self._keys.scheme == TABLE_DENSE:
            return self._keys.first + position
        return self._okeys[position]

    # -- escapes ----------------------------------------------------------------------------


    cdef object row_record(self, Py_ssize_t row):
        """The row as the helper's Record (T.loc[k]). Root rows-form tables: built here from the helper's
        own row tuple, as ``immutable_inputs._row_record`` builds it, without the Python ``loc`` (D132);
        otherwise built by the helper."""
        if <size_t>row >= <size_t>self.rows:
            raise IndexError(f'table row {row} is out of range')
        if self._rowrefs is not None:
            return _native_row_record(self, row)
        mx_handle_fallback()
        record = self.helper().loc[_h_label(self, row)]
        if self._colbase or self.width != len(record):
            return record[self._columns[0]] if self.width == 1 else record
        return record

    cdef TableHandle column_view(self, Py_ssize_t col):
        if <size_t>col >= <size_t>self.width:
            raise IndexError(f'table column {col} is out of range')
        return _column_of(self, self._colbase + col, (self._columns[col],))

    cdef object table_object(self):
        """The bound table of this handle (D45): a NativeTable whose handle is this handle, the
        same object on every call; Python reads on it go to the helper's table (a sub-table is
        materialised by the helper's per-prefix memo on first Python use only)."""
        if self._table_obj is None:
            raise TypeError('a column view is not a table')
        return self._table_obj

    # -- Python side (tests, receipts) -------------------------------------------------------

    def helper(self):
        """The helper Table this handle reads (a sub-table is materialised by the helper's memo)."""
        if not self._prefix:
            return self.owner
        return self.owner.loc[self._prefix if len(self._prefix) > 1 else self._prefix[0]]

    @property
    def key_index(self):
        """D86: how a multi-level full key is found: 'grid' (dense product index), or why it stays
        hashed; None for other schemes."""
        return self._key_index

    @property
    def scheme_name(self):
        return {TABLE_DENSE: 'dense', TABLE_SORTED: 'sorted', TABLE_HASHED: 'hashed',
                TABLE_PRODUCT: 'product'}.get(self.scheme, '?')

    @property
    def shape(self):
        return (self.rows, self.width, self.levels, self.base)

    @property
    def prefix(self):
        return self._prefix

    @property
    def children(self):
        return self._children

    def memory(self):
        """Bytes of this handle tree. `children`: what the eager sub-tables add (their handle
        objects, own key indexes and label tuples); `shared`: everything else (typed columns,
        key parts, the root's indexes, read by every handle of the table); `handles`: count."""
        import sys as _sys
        cdef TableHandle handle
        arrays = 0
        for item in self._hold:
            if type(item) is bytearray:
                arrays += len(<bytearray>item)
            elif type(item) is tuple:
                arrays += _sys.getsizeof(item)
            elif hasattr(item, 'nbytes'):
                arrays += item.nbytes
        count, objects, owned, labels = 0, 0, 0, 0
        stack = [(child, self.scheme) for child in self._children]
        while stack:
            handle, parent = stack.pop()
            count += 1
            objects += _sys.getsizeof(handle)
            if parent != TABLE_PRODUCT:             # PRODUCT children share their axes' arrays
                owned += _own_key_bytes(handle)
                labels += _sys.getsizeof(handle._labels)
            stack.extend((child, handle.scheme) for child in handle._children)
        children = objects + owned + labels
        total = arrays + _sys.getsizeof(self) + objects + labels
        return {'handles': 1 + count, 'children': children, 'shared': total - children, 'total': total}


cdef Py_ssize_t _own_key_bytes(TableHandle h):
    """Bytes of the key arrays a row-stored sub-table owns (its full-key and child indexes)."""
    cdef Py_ssize_t size = 0
    if h.scheme == TABLE_HASHED:
        size += (h._keys.mask + 1) * sizeof(Py_ssize_t)
        if h.levels > 1:
            size += (h._ckeys.mask + 1) * sizeof(Py_ssize_t) + h._ckeys.count * sizeof(HandlePart)
    elif h.scheme == TABLE_SORTED:
        size += h.rows * (sizeof(long long) + sizeof(Py_ssize_t))
    if h._okeys != NULL:
        size += h.rows * sizeof(long long)
    return size


@cython.final
cdef class _HandleTable:
    """Owner of the table object of a sub-table handle: the helper's sub-table, asked from the
    helper (memoised per prefix there) when Python code first reads it, then held here.

    Holding it matters: a per-policy table-valued cell (``return T.loc[prefix]``, D53) is read
    by per-step formulas (``sub["c"]``, ``sub.index``); asking the helper each time ran
    ``owner.loc[prefix]`` (its memo lookup) on every read, twice to three times the cost of
    the read itself (diag-uc-fr). The write is benign when two threads race: both store the
    helper's memoised object, the same one."""
    cdef TableHandle handle
    cdef object _sub

    cdef inline object _helper(self):
        cdef object sub = self._sub
        if sub is None:
            sub = self.handle.helper()
            self._sub = sub
        return sub

    def _table(self):
        return self._helper()

    def __getattr__(self, name):
        return getattr(self._helper(), name)

    def __getitem__(self, key):
        return self._helper()[key]

    def __iter__(self):
        return iter(self._helper())

    def __len__(self):
        return len(self._helper())

    def __contains__(self, key):
        return key in self._helper()

    def __eq__(self, other):
        return self._helper() == other

    def __hash__(self):
        return id(self)

    def __reduce__(self):
        return self._helper().__reduce__()


Table.register(_HandleTable)


cdef inline object _semantic_owner(object owner):
    if type(owner) is _HandleTable:
        return (<_HandleTable>owner)._helper()
    return owner


cdef void _attach_table_objects(TableHandle root) except *:
    """Give every handle of a table its NativeTable, while the tree is still private."""
    cdef TableHandle handle
    cdef NativeTable table
    cdef _HandleTable lazy
    stack = [root]
    while stack:
        handle = <TableHandle>stack.pop()
        table = NativeTable.__new__(NativeTable)
        if handle._prefix:
            lazy = _HandleTable.__new__(_HandleTable)
            lazy.handle = handle
            table._owner = lazy
        else:
            table._owner = handle.owner
        table.handle = handle
        handle._table_obj = table
        stack.extend(handle._children)


cdef TableHandle _column_of(TableHandle source, Py_ssize_t colbase, tuple columns):
    cdef TableHandle view = _clone(source)
    view._colbase = colbase
    view._columns = columns
    view.width = len(columns)
    view._table_obj = None
    return view


@cython.final
cdef class RecordHandle:
    """Typed field reads of one helper Record: field(name) once, then f64/i64/obj by slot."""

    def __init__(self, owner):
        _record_init(self, owner)

    cdef Py_ssize_t field(self, object name) except -2:
        cdef Py_ssize_t j, n = mx_tuple_size(<PyObject*>self._items)
        cdef PyObject* key
        for j in range(n):
            if <PyObject*>name == mx_tuple_item(mx_tuple_item(<PyObject*>self._items, j), 0):
                return j
        for j in range(n):
            key = mx_tuple_item(mx_tuple_item(<PyObject*>self._items, j), 0)
            if PyUnicode_Check(name) and PyUnicode_Check(<object>key):
                if mx_unicode_compare(<PyObject*>name, key) == 0:
                    return j
            elif not PyUnicode_Check(name) and not PyUnicode_Check(<object>key):
                if mx_rich_equal(<PyObject*>name, key, 2):
                    mx_handle_fallback()
                    return j
        raise KeyError(name)


    cdef double f64(self, Py_ssize_t slot) except? -1:
        return _object_f64(_h_value(self, slot))

    cdef long long i64(self, Py_ssize_t slot) except? -1:
        return _object_i64(_h_value(self, slot))

    cdef object obj(self, Py_ssize_t slot):
        return <object>_h_value(self, slot)


cdef void _record_init(RecordHandle handle, object owner) except *:
    cdef PyObject* items
    handle.owner = owner
    if isinstance(owner, Record) and _record_items_offset >= 0:
        # the Record's own immutable pairs, read from its slot (no attribute lookup)
        items = (<PyObject**>(<char*><PyObject*>owner + _record_items_offset))[0]
        if items != NULL and PyTuple_Check(<object>items):
            handle._items = <tuple>items
            return
    if isinstance(owner, Record):
        handle._items = owner._items
        return
    mx_handle_fallback()
    handle._items = tuple(owner.items())


from cpython.object cimport PyObject_GenericGetAttr
cdef extern from "Python.h":
    # PyObject* forms: no reference count is taken on the shared dict or its value
    PyObject* _mx_dict_get "PyDict_GetItemWithError"(PyObject* mapping, PyObject* key) except? NULL
    bint _mx_dict_exact "PyDict_CheckExact"(PyObject* value)
    ctypedef struct _mx_PyMemberDef "PyMemberDef":
        Py_ssize_t offset
    ctypedef struct _mx_PyMemberDescrObject "PyMemberDescrObject":
        _mx_PyMemberDef* d_member

cdef Py_ssize_t _native_cache_offset = -1


cdef PyObject* _table_native_cache(object owner) except? NULL:
    """The `_native_cache` slot of an exact helper Table, borrowed, read at its offset without a lock.

    In free-threaded CPython a slot read through its member descriptor takes the object's
    critical section, and every worker reads the slot of the same shared Data table on every
    per-step table read (diag-4w: DIA_US_S, four workers spent most samples waiting there).
    The slot is assigned only while the table is built, before it is shared (immutable_inputs
    and the compiled pdn.table), and never reassigned, so a plain load sees its one value; the
    table (held by the caller) keeps it alive. NULL (no exception) when the slot is unset or
    its offset is unknown: the caller takes the generic lookup."""
    global _native_cache_offset
    if _native_cache_offset < 0:
        descriptor = Table.__dict__.get('_native_cache')
        if type(descriptor).__name__ != 'member_descriptor':
            return NULL
        _native_cache_offset = (<_mx_PyMemberDescrObject*><PyObject*>descriptor).d_member.offset
    return (<PyObject**>((<char*><PyObject*>owner) + _native_cache_offset))[0]


cdef object _bound_handle(object owner):
    """The handle of a helper Table, built once per table object and kept in its native cache
    (at preparation for tables built by compiled code, else at the first table_handle request)."""
    cdef dict cache
    cdef PyObject* slot
    cdef PyObject* found
    if <PyObject*>_mx_Py_TYPE(owner) == _TABLE_TYPE:
        # Table defines __getattr__, so an ordinary attribute read takes the slot getattr hook;
        # an exact Table's slot is read directly and its cached handle borrowed (it is stored
        # once with setdefault and never removed), so no lock and no reference count of the
        # shared cache dict is touched (diag-4w)
        slot = _table_native_cache(owner)
        if slot != NULL and _mx_dict_exact(slot):
            found = _mx_dict_get(slot, <PyObject*>TableHandle)
            if found != NULL:
                return <object>found
            cache = <dict>slot
        else:
            cache = <dict>PyObject_GenericGetAttr(owner, '_native_cache')
    else:
        cache = owner._native_cache
    handle = cache.get(TableHandle)
    if handle is None:
        handle = cache.setdefault(TableHandle, TableHandle(owner))
    return handle


cdef TableHandle table_handle(object value):
    """The handle of a bound table: NativeTable.handle, else the table's own handle.

    A NativeTable without a handle (bound before its table had one) takes it at its first native
    read: the handle is found or built once per table through the table's cache (``_bound_handle``:
    built at most once per table, published with ``setdefault``, so every racing thread gets the same
    object) and stored into the NativeTable under a critical section on it, only while the field is
    still None. Later reads return the field without the counted path. A table whose storage has no
    handle raises here, at the first native read, as before."""
    cdef TableHandle handle
    if type(value) is NativeTable:
        if (<NativeTable>value).handle is not None:
            return (<NativeTable>value).handle
        mx_handle_fallback()        # once per table: its first native read
        handle = _bound_handle((<NativeTable>value)._owner)
        with cython.critical_section(value):
            if (<NativeTable>value).handle is None:
                (<NativeTable>value).handle = handle
        return (<NativeTable>value).handle
    elif type(value) is TableHandle:
        return <TableHandle>value
    mx_handle_fallback()        # not a bound table: the handle is found through its cache
    # an exact helper Table first: `Table` is a Mapping (an ABC), and isinstance() through
    # the ABC machinery cost more than the read it guards, on every per-step read of a Data
    # table held as a plain Table (diag-us: SPIA_US_S mort_rate_base, improvement_rate)
    if <PyObject*>_mx_Py_TYPE(value) != _TABLE_TYPE and not isinstance(value, Table):
        raise TypeError(f'expected a Table, got {type(value).__name__}')
    return _bound_handle(value)


cdef RecordHandle record_handle(object value):
    cdef RecordHandle handle = RecordHandle.__new__(RecordHandle)
    _record_init(handle, value)
    return handle


# ---- W1.9: borrowed sub-tables and index labels ----------------------------------------------

cdef inline TableRef _sub_ref(TableHandle h, const HandlePart* q, Py_ssize_t n) except? NULL:
    """Borrowed shared child for n prefix parts, NULL when absent (no reference is taken)."""
    cdef PyObject* handle = <PyObject*>h
    cdef Py_ssize_t j
    if n >= h.levels:
        raise TypeError(f'sub_{n} needs fewer key parts than the {h.levels} levels of the table')
    for j in range(n):
        handle = _h_child(handle, &q[j])
        if handle == NULL:
            return NULL
    if (<TableHandle>handle)._colbase != h._colbase or (<TableHandle>handle).width != h.width:
        raise TypeError('a column view has no borrowed sub-table')    # never lowered on views
    return handle


cdef TableRef sub_ref_1(TableHandle h, TableKey p0) except NULL:
    cdef HandlePart q[1]
    _set_part(&q[0], p0, h._spec_missing)
    found = _sub_ref(h, q, 1)
    if found == NULL:
        raise KeyError(_original(p0))
    return found


cdef TableRef sub_ref_2(TableHandle h, TableKey p0, TableKey p1) except NULL:
    cdef HandlePart q[2]
    _set_part(&q[0], p0, h._spec_missing)
    _set_part(&q[1], p1, h._spec_missing)
    found = _sub_ref(h, q, 2)
    if found == NULL:
        raise KeyError((_original(p0), _original(p1)))
    return found


cdef TableRef sub_ref_3(TableHandle h, TableKey p0, TableKey p1, TableKey p2) except NULL:
    cdef HandlePart q[3]
    _set_part(&q[0], p0, h._spec_missing)
    _set_part(&q[1], p1, h._spec_missing)
    _set_part(&q[2], p2, h._spec_missing)
    found = _sub_ref(h, q, 3)
    if found == NULL:
        raise KeyError((_original(p0), _original(p1), _original(p2)))
    return found


cdef TableRef sub_ref_4(TableHandle h, TableKey p0, TableKey p1, TableKey p2, TableKey p3) except NULL:
    cdef HandlePart q[4]
    _set_part(&q[0], p0, h._spec_missing)
    _set_part(&q[1], p1, h._spec_missing)
    _set_part(&q[2], p2, h._spec_missing)
    _set_part(&q[3], p3, h._spec_missing)
    found = _sub_ref(h, q, 4)
    if found == NULL:
        raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3)))
    return found


cdef TableRef sub_ref_5(TableHandle h, TableKey p0, TableKey p1, TableKey p2, TableKey p3,
                        TableKey p4) except NULL:
    cdef HandlePart q[5]
    _set_part(&q[0], p0, h._spec_missing)
    _set_part(&q[1], p1, h._spec_missing)
    _set_part(&q[2], p2, h._spec_missing)
    _set_part(&q[3], p3, h._spec_missing)
    _set_part(&q[4], p4, h._spec_missing)
    found = _sub_ref(h, q, 5)
    if found == NULL:
        raise KeyError((_original(p0), _original(p1), _original(p2), _original(p3), _original(p4)))
    return found


cdef inline Py_ssize_t _key_position(TableHandle h, Py_ssize_t i) except -1:
    cdef Py_ssize_t position = i + h.rows if i < 0 else i
    if not 0 <= position < h.rows:
        raise IndexError(f'index {i} is out of bounds for axis 0 with size {h.rows}')
    return position


cdef object key_label(TableHandle h, Py_ssize_t i):
    """The i-th label of the index in row order (a tuple on several levels), as the helper's
    ``index[i]``; negative i counts from the end."""
    return _h_label(h, _key_position(h, i))


cdef double key_f64(TableHandle h, Py_ssize_t i) except? -1:
    """``float(index[i])``: a float label is read without a reference on it."""
    cdef Py_ssize_t position = _key_position(h, i)
    cdef PyObject* label
    if h.scheme != TABLE_PRODUCT and h._labels is not None and h.levels == 1:
        label = mx_tuple_item(<PyObject*>h._labels, position)
        if PyFloat_CheckExact(<object>label):
            return PyFloat_AS_DOUBLE(<object>label)
    return float(_h_label(h, position))


cdef dict _level_extremes = {}     # (id(handle), level, last) -> (handle, value): written once per key


cdef object level_extreme(TableHandle h, object level, bint last):
    """``index.get_level_values(level).min()`` (``max()`` when ``last``) of the handle's table, as
    the helper (pandas) answers it, computed once per handle and level: the first call asks the
    helper (a counted fallback), later calls read the memo. The memo holds the handle, so its id
    is not reused while the entry lives; concurrent first calls store the same value."""
    key = (<Py_ssize_t><void*>h, level, last)
    found = _level_extremes.get(key)
    if found is not None and (<tuple>found)[0] is h:
        return (<tuple>found)[1]
    mx_handle_fallback()
    values = h.helper().index.get_level_values(level)
    value = values.max() if last else values.min()
    _level_extremes[key] = (h, value)
    return value


def level_extreme_of(TableHandle h, level, last=False):
    """Python access to level_extreme (tests)."""
    return level_extreme(h, level, last)


def key_label_of(TableHandle h, Py_ssize_t i):
    """Python access to key_label (tests)."""
    return key_label(h, i)


def key_f64_of(TableHandle h, Py_ssize_t i):
    """Python access to key_f64 (tests)."""
    return key_f64(h, i)


def sub_ref_of(TableHandle h, *parts):
    """Python access to sub_ref_n (tests): the shared child handle it borrows."""
    cdef TableRef found
    n = len(parts)
    if n == 1:
        found = sub_ref_1[object](h, parts[0])
    elif n == 2:
        found = sub_ref_2[object](h, parts[0], parts[1])
    elif n == 3:
        found = sub_ref_3[object](h, parts[0], parts[1], parts[2])
    else:
        raise TypeError('sub_ref_of takes 1 to 3 parts')
    return <object>found


def table_handle_of(value):
    """Python access to table_handle (tests and receipts)."""
    return table_handle(value)
