"""Typed-memoryview Cells read by subscript: the array is held per item (``holds`` kind ``array``).

A formula of an item class that subscripts a call of a zero-argument cached Cell whose traced result is a typed
memoryview, ``self.data.mort_table_array()[a]`` or ``self.rates()[t]``, would copy the owner's cached slice on
every evaluation: an atomic increment and decrement of the one acquisition count that every worker shares
(Term_US_SEC at four workers was slower than at one). The read becomes an element read of a field of the item
that holds the slice: the first read in the item's lifetime calls the Cell (its own first-fill discipline,
mv-race) and stores the slice in the field, once; later reads index the field, which acquires nothing.

Rule (recognised by traced type, never by spelling): the receiver is ``self`` or a Reference of ``self`` typed as
a compiled Space of this module; the Cell is cached, takes no arguments, and its traced result is a typed
memoryview; the formula has parameters (it runs many times per item); the reading class has items (its instances are items, used by one thread at a time, as their own
caches are). The held field is reset with the item's per-policy caches (``holds.reset_lines``), so a cleared or
reassigned input is read afresh by the next policy; an item that holds an array keeps the slice it took, as an
item keeps the table it first bound (D130, R-4), and the slice keeps its buffer alive.
"""
import libcst as cst

from .consts import MX_SELF
from .holds import hold_table


def _owner(cls_info, receiver):
    """The ClassInfo of the receiver of a Cell call: ``self`` or a typed Reference of ``self``, else None."""
    if isinstance(receiver, cst.Name) and receiver.value == MX_SELF:
        return cls_info
    if (isinstance(receiver, cst.Attribute) and isinstance(receiver.value, cst.Name)
            and receiver.value.value == MX_SELF):
        ref = cls_info.refs.get(receiver.attr.value)
        decl = getattr(ref, 'decl_type_expr', '') if ref is not None else ''
        if decl and getattr(ref, 'is_relative', False):
            return cls_info.module.classes.get(decl)
    return None


def _array_cell(cls_info, call):
    """(receiver code, Cell name, C type, annotation) of a zero-argument cached Cell call with a typed-memoryview
    result, else None."""
    if not (isinstance(call, cst.Call) and not call.args and isinstance(call.func, cst.Attribute)):
        return None
    owner = _owner(cls_info, call.func.value)
    if owner is None:
        return None
    cell = owner.cells.get(call.func.attr.value)
    if (cell is None or not cell.has_formula_def or cell.has_args() or not cell.has_typeinfo()
            or not cell.is_array_returned):
        return None
    ctype = cell.get_rettype_expr(c_style=True)
    if '[:' not in ctype:
        return None
    receiver = cst.Module([]).code_for_node(call.func.value)
    return receiver, call.func.attr.value, ctype, cell.get_rettype_expr()


def _element(annotation):
    """The element type of a memoryview annotation ``_mx_cy.const[_mx_cy.double][:]`` (or without ``const``)."""
    base = annotation.split('[:', 1)[0].rstrip('[')
    if base.startswith('_mx_cy.const[') and base.endswith(']'):
        base = base[len('_mx_cy.const['):-1]
    return base if base.startswith('_mx_cy.') and '[' not in base else None


class _HoldArrays(cst.CSTTransformer):
    def __init__(self, cls_info, functions):
        super().__init__()
        self.cls_info = cls_info
        self.functions = functions
        self.depth = 0      # nested functions and lambdas keep their reads (their self is not the item's)

    def visit_Lambda(self, node):
        self.depth += 1

    def leave_Lambda(self, original, updated):
        self.depth -= 1
        return updated

    def visit_FunctionDef(self, node):
        self.depth += 1

    def leave_FunctionDef(self, original, updated):
        self.depth -= 1
        return updated

    def leave_Subscript(self, original, updated):
        if self.depth != 1 or not all(isinstance(e.slice, cst.Index) for e in updated.slice):
            return updated
        found = _array_cell(self.cls_info, original.value)
        if found is None:
            return updated
        receiver, name, ctype, annotation = found
        element = _element(annotation)
        if element is None or ctype.count(':') != len(updated.slice):
            return updated      # a partial index is a sub-view, not an element: the read stays as written
        field = hold_table(self.cls_info).field('array', (receiver, name), ctype=ctype, annotation=annotation)
        hold = f'_mx_hold_{self.cls_info.name}{field}'
        read = f'_mx_held_{self.cls_info.name}{field}'
        ndim = len(updated.slice)
        if hold not in self.functions:
            params = ', '.join(f'i{k}: _mx_cy.Py_ssize_t' for k in range(ndim))
            index = ', '.join(f'i{k}' for k in range(ndim))
            self.functions[hold] = cst.parse_module(
                f'@_mx_cy.cfunc\n'
                f'def {hold}({MX_SELF}: {self.cls_info.name}) -> {annotation}:\n'
                f'    {MX_SELF}.{field} = {receiver}.{name}()\n'
                f'    {MX_SELF}.{field}_ready = True\n'
                f'    return {MX_SELF}.{field}\n'
                f'\n\n'
                f'@_mx_cy.cfunc\n'
                f'@_mx_cy.inline\n'
                f'def {read}({MX_SELF}: {self.cls_info.name}, {params}) -> {element}:\n'
                f'    if {MX_SELF}.{field}_ready:\n'
                f'        return {MX_SELF}.{field}[{index}]\n'
                f'    return {hold}({MX_SELF})[{index}]\n')
        index = ', '.join(cst.Module([]).code_for_node(e.slice.value) for e in updated.slice)
        return cst.parse_expression(f'{read}({MX_SELF}, {index})')


def lower(node, cls_info, cells, functions):
    """``node`` (a formula FunctionDef of ``cells`` of ``cls_info``) with its typed-memoryview Cell subscripts read
    through held fields; ``functions`` collects the module-level hold functions. Only a formula with parameters is
    lowered: it runs many times per item, while a zero-argument Cell runs once per item, where holding costs more
    than the one slice copy it saves."""
    if (cls_info.name not in cls_info.visitor.itemspace_classes
            or getattr(cls_info, 'collect_shared_item', False) or not cells.has_args()):
        return node
    return node.visit(_HoldArrays(cls_info, functions))
