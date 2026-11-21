"""Python access to the table cells of a compiled model (api-tables).

A compiled table cell keeps the value its formula returned: a plain pandas DataFrame, which compiled code reads
through the table prepared from it (``runtime._tables``), or a native table. Python callers see it through a
per-name property generated on the Space class: ``space.cell()`` and ``space.cell[()]`` return the value, the
same object on every call, as live modelx returns the cached value.

As upstream modelx-cython, a compiled model offers no clear and no assignment of a table cell; its References
are plain attributes.
"""

import threading

_lock = threading.Lock()
_NO_FRAME = object()
_binders = {}       # top-level package of a compiled model -> its runtime _tables.bind


def register_prepared(bind=None, package=None):
    with _lock:
        if bind is not None:
            _binders[package] = bind


def bind_frame(space, value):
    """T-4: prepare ``value`` (a frame cell's computed or assigned frame) in the model's PreparedTable cache and
    return what an item reading it holds (T-4.2): its PreparedTable, or a table whose every slot is -1 (not a
    preparable frame); None when the model has no fast reads."""
    bind = _binders.get(type(space).__module__.split('.')[0])
    return bind(value) if bind is not None else None


_enums = {}


def enum_member(module, qualname, value):
    """The member of the IntEnum ``module.qualname`` whose value is ``value`` (D103).

    A compiled cell holds an admitted enum as its C integer; this is what a Python caller,
    an object-valued use or a collected output receives, as in live modelx.
    """
    cls = _enums.get((module, qualname))
    if cls is None:
        import importlib
        cls = importlib.import_module(module)
        for part in qualname.split('.'):
            cls = getattr(cls, part)
        _enums[module, qualname] = cls
    member = cls._value2member_map_.get(value)
    return member if member is not None else cls(value)

def frame_of(table, path):
    """What a Python caller gets for an internal table, the same object on every request, equal to what
    live modelx returns (D130): the plain DataFrame the table was admitted from (a formula's result or
    an assigned frame), which may be edited in place (compiled code reads the edit after a clear); for a
    table the model text builds with the helper (``pdn.table``), the helper Table itself, as live.

    Only the standard provider's ``NativeTable`` has a frame; another provider's object (bound by its own
    result binder) is returned as it is, as before D101.
    """
    frame = getattr(table, '_mx_python_frame', _NO_FRAME)
    if frame is _NO_FRAME:
        return table
    if frame is None:
        with _lock:
            frame = table._mx_python_frame
            if frame is None:
                from modelx_helpers.immutable_inputs import table_frame
                owner = table.__modelx_table_owner__()
                frame = table_frame(owner)          # a table admitted from a frame
                if frame is None:
                    frame = owner                   # a helper Table: what the formula returned, as live
                table._mx_python_frame = frame
    return frame


class TableCell:
    """What ``space.cell`` gives a Python caller for a compiled table cell."""
    __slots__ = ('_space', '_name', '_path')

    def __init__(self, space, name, path):
        self._space, self._name, self._path = space, name, path

    def __call__(self):
        return frame_of(self._space._mx_table_cell_value(self._name), self._path)

    def __getitem__(self, key):
        if key != ():
            raise KeyError(f'{self._path} has no parameters: use {self._path}[()]')
        return self()

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'<compiled table cell {self._path}>'


def row_series(record, path):
    """What a Python caller gets for a cell whose value is a row of a table admitted from a frame: the
    plain pandas row (``frame.loc[key]``, as live gives it), the same object on every call of the
    cached cell (D130, superseding D119's read-only Series). Any other value is returned as it is.
    Compiled code keeps reading the Record."""
    from modelx_helpers.immutable_inputs import record_row
    found = record_row(record)
    if found is None:
        return record
    row, cached = found
    if cached is None:
        with _lock:
            cached = record_row(record)[1]
            if cached is None:
                cached = row()
                object.__setattr__(record, '_frame_row', (row, cached))
    return cached
