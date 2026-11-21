"""What an item class holds from shared inputs: one registry per class (R-4).

Compiled per-item code holds what it reads from a shared input. The first read in an item's lifetime
resolves the value into a field of the item, and later reads use that field borrowed. The fields are
reset with the item's per-policy caches.

A class's ``HoldTable`` has one entry per held value, in three kinds:

- ``handle``: the TableHandle or RecordHandle of a zero-argument cached table or record Cell
  (``_mx_th_<n>``; None until resolved), keyed by the source expression of the Cell call;
- ``column``: the column index, or record field slot, of one literal name read through a held handle
  (``_mx_tc_<n>``; 0 until resolved), keyed by (held field, name);
- ``row``: the row of a key fixed per item and read through a held handle (``_mx_tr_<n>``; 0 until
  resolved), keyed by (held field, key parts);
- ``array``: the slice of a zero-argument cached Cell with a typed-memoryview result read by subscript
  (``_mx_ta_<n>`` of the Cell's memoryview type, valid while ``_mx_ta_<n>_ready``), keyed by (receiver, Cell);
  see ``held_arrays``.
- ``prepared`` and ``prepared_slot`` (T-3): the PreparedTable of a pandas frame and the slot of a column in it.

Which reads are held is decided in ``table_reads`` by traced type and data flow, never by how a read is
spelled. This module only names the fields and gives their declarations and reset lines.
"""

_KINDS = {
    # kind: (field prefix, C type of the .pxd, annotation of the .py, reset value)
    'handle': ('_mx_th_', 'object', 'object', 'None'),
    'column': ('_mx_tc_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
    'row': ('_mx_tr_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
    'array': ('_mx_ta_', None, None, None),     # typed per field; reset clears the ready flag
    # T-3/T-4.2 (frame_reads): the PreparedTable an item holds for one frame receiver (a table whose every slot is
    # -1 when the frame is not prepared), and whether it is bound (0 until bound), keyed by the receiver
    'prepared': ('_mx_tp_', 'object', 'object', 'None'),
    'prepared_slot': ('_mx_tq_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
    # T-4: one plus the level code of a key part fixed per item (-1 absent), keyed by (table field, level, part);
    # one plus the row of a row cell's key (-1: pandas), keyed by (table field, key)
    'prepared_code': ('_mx_tk_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
    'prepared_row': ('_mx_tw_', 'Py_ssize_t', '_mx_cy.Py_ssize_t', '0'),
}


class HoldTable:
    """The held fields of one item class, by kind and key, in the order they were first needed."""

    def __init__(self):
        self._fields = {kind: {} for kind in _KINDS}
        self.types = {}     # array field -> (C type of the .pxd, annotation of the .py)

    def field(self, kind, key, ctype=None, annotation=None):
        """The field holding ``key`` (created on first use, numbered per kind); an ``array`` field takes the
        memoryview's C type and annotation."""
        fields = self._fields[kind]
        field = fields.get(key)
        if field is None:
            field = fields[key] = f'{_KINDS[kind][0]}{len(fields)}'
            if kind == 'array':
                self.types[field] = (ctype, annotation)
        return field

    def fields(self):
        """(field, kind) of every held field: handles, then columns, then rows."""
        for kind, fields in self._fields.items():
            for field in fields.values():
                yield field, kind

    def __bool__(self):
        return any(self._fields.values())


def hold_table(cls_info):
    """The class's HoldTable, created on first use."""
    table = getattr(cls_info, 'holds', None)
    if table is None:
        table = cls_info.holds = HoldTable()
    return table


def _held(cls_info):
    table = getattr(cls_info, 'holds', None)
    return table.fields() if table is not None else ()


def _types(cls_info):
    return cls_info.holds.types


def pxd_declarations(cls_info):
    """``cdef`` lines of the class's .pxd for its held fields."""
    lines = []
    for field, kind in _held(cls_info):
        if kind == 'array':
            lines += [f'cdef {_types(cls_info)[field][0]} {field}\n', f'cdef bint {field}_ready\n']
        else:
            lines.append(f'cdef {_KINDS[kind][1]} {field}\n')
    return lines


def py_declarations(cls_info):
    """Annotated declarations of the class body of the .py for its held fields."""
    lines = []
    for field, kind in _held(cls_info):
        if kind == 'array':
            lines += [f'{field}: {_types(cls_info)[field][1]}\n', f'{field}_ready: _mx_cy.bint\n']
        else:
            lines.append(f'{field}: {_KINDS[kind][2]}\n')
    return lines


def reset_lines(cls_info, target='root'):
    """Reset of every held field of ``target``, with the per-policy caches (a held array: its ready flag; the
    slice it holds is replaced at the next read)."""
    return [f'{target}.{field}_ready = False' if kind == 'array' else f'{target}.{field} = {_KINDS[kind][3]}'
            for field, kind in _held(cls_info)]
