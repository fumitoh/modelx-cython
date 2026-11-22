"""Vector cells: lists of floats and of integers lowered to typed C buffers (W1.6, D24).

A cell whose every traced value is a list of exact floats is a *float vector cell* ('V');
its cached value is an ``F64Vec`` (provider ``runtime/_native_vectors``). A list of exact
int64 integers gives an *integer vector cell* ('I', ``I64Vec``). A cell whose value is a
non-empty tuple of such lists is a *tuple of vectors* (('T', kinds)): a Python tuple of
typed vectors. The rule is keyed on the traced values only (``tracer.vector_shape``).

In every formula of a module with vector cells, a local is a vector when every assignment
to it is a vector expression; it is *owned* (writable) when every assignment allocates.
Lowered forms (``V`` float, ``I`` integer):

  allocation   [x] * n, n * [x]                -> vec_full(n, x)
               []  (then v.append(x))          -> an empty owned vector that grows
               [e for z in range(a, b)]        -> buffer + C loop  (statement value only)
               [e for a, b in zip(v, w)], [e for a in v]
               v[a:b]                          -> a new vector (a copy, as for a list)
  element      v[i]                            -> vec_get (Python index semantics, IndexError)
               v[i] = x  (owned v)             -> vec_set
               v.append(x)  (owned v)          -> vec_append
               a write or append to a vector the formula did not allocate is a build error
  length       len(v)                          -> v.n
  sums         sum(v), sum(v[a:b])             -> float: compensated, bit-identical to CPython;
                                                  integer: exact int64 (OverflowError)
               sum(e for a, b in zip(v, w) if c), sum(e for a in v)  (statement value only,
               e provably float)               -> C loop, the same compensated sum
  loops        for x in v / for a, b in zip(v, w) -> C loop over the buffers
               for i in range(a, b) whose reads and writes are v[i + c] on vectors the loop
               does not rebind: the bounds are tested once before the loop (loop versioning);
               the loop then reads and writes the buffers directly
  powers       x ** y with x and y provably real (vector elements, float cells, literals)
                                               -> _mx_sys._mx_pow (C pow, as powers.py)
  tuples       return a, b / a, b = cell(t) / cell(t)[k]

The element typer also knows ``min``/``max`` of typed scalars; with mixed float and integer
arguments the result is a *number* (``NUM``: Python returns one of the arguments unchanged), which
becomes a float only through arithmetic with a float, so a lowered sum's value and type stay those
of the Python sum.

A local bound only by ``for x in range(...)`` is a C ``long long`` in modules with vector
cells. Everything else on a vector runs through the vector's sequence protocol and is listed
in the admission report as ``vector_fallbacks``.
"""
import re

import libcst as cst

from .consts import CY_MOD, MX_SYS_MOD
from .helpers import module_name
from .syntax import dotted_name

PROVIDER = 'modelx_cython.runtime._native_vectors'
ALIAS = module_name(PROVIDER)
TYPES = {'V': f'{ALIAS}.F64Vec', 'I': f'{ALIAS}.I64Vec'}
SCALARS = {'V': f'{CY_MOD}.double', 'I': f'{CY_MOD}.longlong'}
FUNCS = {  # operation -> (float vector, integer vector)
    'new': ('vec_new', 'ivec_new'), 'full': ('vec_full', 'ivec_full'), 'of': ('vec_of', 'ivec_of'),
    'get': ('vec_get', 'ivec_get'), 'set': ('vec_set', 'ivec_set'),
    'append': ('vec_append', 'ivec_append'), 'slice': ('vec_slice', 'ivec_slice'),
    'sum': ('vec_sum', 'ivec_sum'), 'sum_range': ('vec_sum_range', 'ivec_sum_range'),
}
LL_MAX = '9223372036854775807'
LL_MIN = '(-9223372036854775807 - 1)'
FLOAT, INT = 'f', 'i'           # scalar kinds of the element typer
NUM = 'n'                       # min/max of mixed float and integer arguments: a float or an int
MATH_FLOAT = ('exp', 'log', 'sqrt', 'erf')      # math functions whose native form returns a C double


def fn(op, elem):
    return f'{ALIAS}.{FUNCS[op][elem == "I"]}'


def vector_type(kind):
    return TYPES[kind]


def _normal_shape(shape):
    if shape == 'E':
        return 'V'
    if isinstance(shape, tuple):
        return ('T', tuple(_normal_shape(k) for k in shape[1]))
    return shape


def cell_vector_shape(cell):
    """The vector shape of a cells (builder.CombinedCellsInfo.vector_shape)."""
    if not (cell.has_typeinfo() and cell.has_formula_def) or cell.has_spec_rettype or cell.is_special():
        return None
    if cell.body_has_closure or cell.formula_is_generator or cell.formula_has_captured_default:
        return None
    ret = cell._rt.ret_type
    if ret.is_array or ret.value_type not in (list, tuple):
        return None
    return _normal_shape(ret.vector_shape) if ret.vector_shape else None


CACHE_TYPE = f'{ALIAS}.VecCache'
MAX_CACHE_SLOTS = 1 << 20


MAX_CACHE_PARAMS = 3
CACHE_SLOT_BYTES = 8            # one pointer per slot (the vectors themselves are counted nowhere else either)


def _cache_window(cell):
    """(axes, entries) of a vector cell eligible for a VecCache by its form, else (None, reason)."""
    if not (cell.vector_shape and cell.has_formula_def and 1 <= len(cell.params) <= MAX_CACHE_PARAMS
            and not cell.is_locked
            and all(cell.get_argtype_expr(p, c_style=True) == 'long long' for p in cell.params)):
        return None, None
    try:
        axes = tuple(cell.associative_logical_axes)
    except (KeyError, AttributeError, TypeError):
        return None, 'the arguments are not all sampled C integers in a bounded range'
    if len(axes) != len(cell.params) or [a.key for a in axes] != list(cell.params):
        return None, 'the arguments are not all sampled C integers in a bounded range'
    entries = 1
    for axis in axes:
        if not 0 < axis.stop - axis.lower:
            return None, 'empty sampled window'
        entries *= int(axis.stop - axis.lower)
    return axes, entries


# The budget of the vector caches (D79; the values of the removed D74 window budget, D135).
VECTOR_CACHE_CELL_BYTES = 16 << 20
VECTOR_CACHE_SPACE_BYTES = 64 << 20


def vector_cache_plan(cls):
    """({cell name: (axes, entries, bytes)}, {cell name: reason}) for the vector cells of a Space:
    a vector cell with one to MAX_CACHE_PARAMS C integer parameters gets a dense VecCache within the
    budget: at most ``max_dense_entries`` slots (spec key of the Cell or Space, MAX_CACHE_SLOTS at
    most), VECTOR_CACHE_CELL_BYTES per Cell and VECTOR_CACHE_SPACE_BYTES per Space, largest first.
    A refused cell keeps the dictionary cache and its reason is reported (``cache_detail``)."""
    plan = getattr(cls, '_mx_vector_cache_plan', None)
    if plan is not None:
        return plan
    from .automatic import max_dense_entries
    chosen, refused, candidates = {}, {}, []
    for name, cell in cls.cells.items():
        axes, entries = _cache_window(cell)
        if axes is None:
            if entries is not None:
                refused[name] = entries
            continue
        limit = min(max_dense_entries(cell), MAX_CACHE_SLOTS)
        size = entries * CACHE_SLOT_BYTES
        window = ' x '.join(str(int(a.stop - a.lower)) for a in axes)
        if entries > limit:
            refused[name] = f'vector cache window {window} = {entries:,} entries exceeds max_dense_entries {limit:,}'
        elif size > VECTOR_CACHE_CELL_BYTES:
            refused[name] = (f'vector cache window {window} = {size:,} bytes exceeds the per-cell budget '
                             f'{VECTOR_CACHE_CELL_BYTES:,}')
        else:
            candidates.append((size, name, axes, entries))
    used = 0
    for size, name, axes, entries in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if used + size > VECTOR_CACHE_SPACE_BYTES:
            refused[name] = (f'Space budget {VECTOR_CACHE_SPACE_BYTES:,} bytes for vector caches exhausted '
                             f'({used:,} used, {size:,} needed)')
            continue
        used += size
        chosen[name] = (axes, entries, size)
    plan = (chosen, refused)
    try:
        cls._mx_vector_cache_plan = plan
    except AttributeError:      # a class info without attributes: recomputed per call
        pass
    return plan


def uses_vector_cache(cell):
    """Whether a vector cell caches its values in a VecCache (dense, by index) instead of a dict:
    one to MAX_CACHE_PARAMS C integer parameters, within the budget of ``vector_cache_plan``. With
    several parameters the slot is the row-major index over the sampled axes."""
    if _cache_window(cell)[0] is None or getattr(cell, 'parent', None) is None:
        return False
    return cell.name in vector_cache_plan(cell.parent)[0]


def cache_constructor(cell):
    axes = cell.associative_logical_axes
    if len(axes) == 1:
        return f'{CACHE_TYPE}({int(axes[0].stop - axes[0].lower)}, {int(axes[0].lower)})'
    size = 1
    for axis in axes:
        size *= int(axis.stop - axis.lower)
    return f'{CACHE_TYPE}({size}, 0)'


def _cache_key(cell, key_name):
    """Statements binding ``key_name`` to the slot of a multi-parameter vector cell's arguments:
    the row-major index when every argument is inside its sampled axis, else -1 (not cached)."""
    axes = cell.associative_logical_axes
    inside = ' and '.join(f'{int(a.lower)} <= {p} < {int(a.stop)}' for p, a in zip(cell.params, axes))
    index = f'({cell.params[0]} - {int(axes[0].lower)})'
    for p, a in zip(cell.params[1:], axes[1:]):
        index = f'({index} * {int(a.stop - a.lower)} + ({p} - {int(a.lower)}))'
    return (f'{key_name}: {CY_MOD}.longlong = -1\n'
            f'if {inside}:\n    {key_name} = {index}\n')


def cache_body(cell):
    """The public method of a vector cell with a VecCache: one acquire load on a hit."""
    cache = f'self._v_{cell.name}'
    value = '_mx_vector_value'
    while value in cell.params:
        value += '_'
    source = f'if {cache} is None:\n    {cache} = {cache_constructor(cell)}\n'
    if len(cell.params) == 1:
        key = cell.params[0]
    else:
        key = '_mx_vector_key'
        while key in cell.params:
            key += '_'
        source += _cache_key(cell, key)
    args = ', '.join(cell.params)
    source += (f'{value}: object = {ALIAS}.cache_get({cache}, {key})\n'
               f'if {value} is not None:\n    return {value}\n'
               f'return {cache}.put({key}, self._f_{cell.name}({args}))\n')
    return cst.IndentedBlock(body=cst.parse_module(source).body)


def module_vector_cells(module):
    return any(cell.vector_shape for cls in module.classes.values() for cell in cls.cells.values())


class _LocalAllocations(cst.CSTVisitor):
    """D94: a function that allocates a local list a formula may type as a vector: ``x = [c] * n`` /
    ``x = n * [c]`` with a numeric literal ``c``, or ``x = []`` with ``x.append(...)`` in the same function."""

    def __init__(self):
        self.found = False
        self._stack = []

    def visit_FunctionDef(self, node):
        self._stack.append((set(), set()))

    def leave_FunctionDef(self, node):
        empties, appended = self._stack.pop()
        if empties & appended:
            self.found = True

    def visit_Assign(self, node):
        if not self._stack or len(node.targets) != 1 or not isinstance(node.targets[0].target, cst.Name):
            return
        value, name = node.value, node.targets[0].target.value
        if isinstance(value, cst.List) and not value.elements:
            self._stack[-1][0].add(name)
        elif isinstance(value, cst.BinaryOperation) and isinstance(value.operator, cst.Multiply):
            for one in (value.left, value.right):
                if (isinstance(one, cst.List) and len(one.elements) == 1
                        and isinstance(one.elements[0], cst.Element) and _numeric_literal(one.elements[0].value)):
                    self.found = True

    def visit_Call(self, node):
        f = node.func
        if self._stack and isinstance(f, cst.Attribute) and f.attr.value == 'append' and isinstance(f.value, cst.Name):
            self._stack[-1][1].add(f.value.value)


def _numeric_literal(node):
    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
        node = node.expression
    return isinstance(node, (cst.Float, cst.Integer))


def module_uses_vectors(module):
    """Vector lowering runs in a module with vector cells, and (D94) in one whose formulas allocate local
    lists that may be typed vectors; there it changes only the formulas that do get a typed local vector."""
    cached = getattr(module, '_mx_uses_vectors', None)
    if cached is not None:
        return cached
    result = module_vector_cells(module)
    if not result:
        source = getattr(getattr(getattr(module, 'visitor', None), 'wrapper', None), 'module', None)
        if source is not None:
            scan = _LocalAllocations()
            source.visit(scan)
            result = scan.found
    try:
        module._mx_uses_vectors = result
    except AttributeError:
        pass
    return result


def providers(module):
    return {PROVIDER} if module_uses_vectors(module) else set()


def _shape_json(shape):
    return [shape[0], list(shape[1])] if isinstance(shape, tuple) else shape


class VectorWriteError(Exception):
    """A formula writes an element of a vector it did not allocate."""


# ---------------------------------------------------------------- syntax helpers
def _expr(text):
    return cst.parse_expression(text)


def _stmt(text):
    return cst.parse_statement(text)


def _code(node):
    return cst.Module([]).code_for_node(node)


def _is_name(node, value=None):
    return isinstance(node, cst.Name) and (value is None or node.value == value)


def _index(sub):
    """The index expression of a one-element non-slice subscript, else None."""
    if len(sub.slice) == 1 and isinstance(sub.slice[0].slice, cst.Index) and sub.slice[0].slice.star is None:
        return sub.slice[0].slice.value
    return None


def _slice(sub):
    if len(sub.slice) == 1 and isinstance(sub.slice[0].slice, cst.Slice) and sub.slice[0].slice.step is None:
        return sub.slice[0].slice
    return None


def _int_literal(node):
    if isinstance(node, cst.Integer):
        return int(node.value, 0)
    if (isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus)
            and isinstance(node.expression, cst.Integer)):
        return -int(node.expression.value, 0)
    return None


def _list_range(node):
    """diag-kr: the arguments of ``list(range(a))`` / ``list(range(a, b))``, else None."""
    if (isinstance(node, cst.Call) and _is_name(node.func, 'list') and _plain_args(node, 1, 1)
            and isinstance(inner := node.args[0].value, cst.Call) and _is_name(inner.func, 'range')
            and _plain_args(inner, 1, 2)):
        return [a.value for a in inner.args]
    return None


def _plain_args(call, low, high):
    return (low <= len(call.args) <= high
            and all(a.keyword is None and a.star == '' for a in call.args))


def _names(target):
    if isinstance(target, cst.Name):
        return [target.value]
    if isinstance(target, (cst.Tuple, cst.List)):
        return [n for e in target.elements for n in _names(e.value)]
    if isinstance(target, cst.StarredElement):
        return _names(target.value)
    return []


def _walk(node):
    found = []

    class W(cst.CSTVisitor):
        def on_visit(self, n):
            found.append(n)
            return True
    node.visit(W())
    return found


class _Rename(cst.CSTTransformer):
    def __init__(self, names):
        self.names = names

    def leave_Name(self, original, updated):
        new = self.names.get(updated.value)
        return updated.with_changes(value=new) if new else updated


# ---------------------------------------------------------------- facts of one formula
class _Facts(cst.CSTVisitor):
    """Bindings, element writes, appends and returns of one formula body."""

    def __init__(self):
        self.assigns = {}           # name -> [('value', expr) | ('unpack', expr, i, n) | ('aug', op, expr)
                                    #          | ('for', iter, i, n) | ('other',)]
        self.writes = []            # (base expr, value expr) of v[i] = x
        self.appends = []           # (base expr, value expr, call) of v.append(x)
        self.returns = []           # return value expressions
        self.statement_comps = set()
        self.concat_values = {}     # diag-kr: id of a returned concatenation -> its operands
        self.store_targets = set()  # ids of subscripts that are assignment targets
        self.comp_targets = set()

    def add(self, name, item):
        self.assigns.setdefault(name, []).append(item)

    def visit_Assign(self, node):
        if isinstance(node.value, cst.ListComp) and len(node.targets) == 1:
            self.statement_comps.add(id(node.value))
        for t in node.targets:
            tgt = t.target
            if isinstance(tgt, cst.Name):
                self.add(tgt.value, ('value', node.value))
            elif isinstance(tgt, (cst.Tuple, cst.List)):
                n = len(tgt.elements)
                for i, e in enumerate(tgt.elements):
                    if isinstance(e.value, cst.Name) and not isinstance(e, cst.StarredElement):
                        self.add(e.value.value, ('unpack', node.value, i, n))
                    else:
                        for name in _names(e.value):
                            self.add(name, ('other',))
                        if isinstance(e.value, cst.Subscript):
                            self.store_targets.add(id(e.value))
            elif isinstance(tgt, cst.Subscript):
                self.store_targets.add(id(tgt))
                self.writes.append((tgt.value, node.value))

    def concatenation(self, value):
        """diag-kr: ``A + B + ...`` whose every operand is a list comprehension or
        ``list(range(...))``: the operands, left to right, else None."""
        if not (isinstance(value, cst.BinaryOperation) and isinstance(value.operator, cst.Add)):
            return None
        parts = []
        def flatten(node):
            if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Add):
                return flatten(node.left) and flatten(node.right)
            if isinstance(node, cst.ListComp) or _list_range(node) is not None:
                parts.append(node)
                return True
            return False
        return parts if flatten(value) else None

    def visit_Return(self, node):
        if node.value is not None:
            self.returns.append(node.value)
        if isinstance(node.value, cst.ListComp):
            self.statement_comps.add(id(node.value))
        parts = self.concatenation(node.value)
        if parts is not None:
            # built left to right into one typed vector: the order of evaluation is kept
            self.concat_values[id(node.value)] = parts
            self.statement_comps.update(id(part) for part in parts if isinstance(part, cst.ListComp))
        if isinstance(node.value, cst.Tuple) and all(
                isinstance(e.value, (cst.Name, cst.Integer, cst.Float, cst.ListComp)) for e in node.value.elements):
            # no other element has effects, so building a comprehension first keeps the order
            self.statement_comps.update(id(e.value) for e in node.value.elements
                                        if isinstance(e.value, cst.ListComp))

    def visit_AugAssign(self, node):
        if isinstance(node.target, cst.Name):
            self.add(node.target.value, ('aug', node.operator, node.value))
        elif isinstance(node.target, cst.Subscript):
            self.store_targets.add(id(node.target))
            self.writes.append((node.target.value, None))

    def visit_AnnAssign(self, node):
        for name in _names(node.target):
            self.add(name, ('other',))

    def visit_For(self, node):
        names = _names(node.target)
        if isinstance(node.target, cst.Name):
            self.add(node.target.value, ('for', node.iter, 0, 1))
        elif isinstance(node.target, cst.Tuple) and all(isinstance(e.value, cst.Name) for e in node.target.elements):
            for i, e in enumerate(node.target.elements):
                self.add(e.value.value, ('for', node.iter, i, len(node.target.elements)))
        else:
            for name in names:
                self.add(name, ('other',))

    def visit_CompFor(self, node):
        # a comprehension target is local to the comprehension (language level 3)
        self.comp_targets.update(_names(node.target))

    def visit_NamedExpr(self, node):
        self.add(node.target.value, ('other',))

    def visit_Param(self, node):
        self.add(node.name.value, ('other',))

    def visit_With(self, node):
        for item in node.items:
            if item.asname is not None:
                for n in _names(item.asname.name):
                    self.add(n, ('other',))

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.add(node.name.name.value, ('other',))

    def visit_Global(self, node):
        for n in node.names:
            self.add(n.name.value, ('other',))

    def visit_Nonlocal(self, node):
        for n in node.names:
            self.add(n.name.value, ('other',))

    def visit_Del(self, node):
        for name in _names(node.target):
            self.add(name, ('other',))

    def visit_Import(self, node):
        for alias in node.names:
            self.add(_code(alias.asname.name if alias.asname else alias.name).split('.')[0], ('other',))

    def visit_ImportFrom(self, node):
        if not isinstance(node.names, cst.ImportStar):
            for alias in node.names:
                self.add(_code(alias.asname.name if alias.asname else alias.name), ('other',))

    def visit_Call(self, node):
        f = node.func
        if (isinstance(f, cst.Attribute) and f.attr.value == 'append' and len(node.args) == 1
                and node.args[0].keyword is None and node.args[0].star == ''):
            self.appends.append((f.value, node.args[0].value, node))

    def visit_FunctionDef(self, node):
        self.add(node.name.value, ('other',))
        return True

    def visit_ClassDef(self, node):
        self.add(node.name.value, ('other',))
        return False


# ---------------------------------------------------------------- the lowering
class VectorLowering(cst.CSTTransformer):
    """Rewrite one ``_f_`` formula (a method taking ``self``) of a module with vector cells."""

    def __init__(self, cls_info, cells, builtins, counter):
        self.cls_info = cls_info
        self.cells = cells
        self.shapes = {name: c.vector_shape for name, c in cls_info.cells.items() if c.vector_shape}
        self.builtins = builtins
        self.counter = counter          # shared per module: unique temporaries
        self.temps = set()              # the temporaries temp() made
        self.loop_renames = {}          # generated loop target -> its name in the formula
        self.env, self.scalars, self.owned = {}, {}, set()
        self.lowered, self.fallbacks = 0, []
        self.bounds_hoisted, self.bounds_checked, self.powers = 0, 0, 0
        self.decls = []
        self.comp_scalars = []          # stack of {name: scalar kind} inside expanded loops
        self.extreme_calls = {}          # diag-kr: min/max over a generator on vectors
        self.range_names = set()
        self.ref_scalars = {}
        self.param_scalars = {}         # D87: C-typed parameters, unless the formula rebinds them
        self.strict_locals = False      # D94: module without vector cells: untyped writes keep a list
        self.scope_stack = []           # per enclosing comprehension/lambda: expanded here

    # ---------------------------------------------------------------- kinds
    def cell_scalar(self, name):
        cell = self.cls_info.cells.get(name)
        if cell is None or self.shapes.get(name):
            return None
        try:
            ret = cell.get_rettype_expr(c_style=True)
        except Exception:       # a cell without type information
            return None
        return {'double': FLOAT, 'long long': INT}.get(ret)

    def kind(self, node):
        """'V', 'I', 'AV', 'AI' (fresh allocations), ('T', kinds) or None for an original node."""
        if isinstance(node, cst.Call):
            f = node.func
            if isinstance(f, cst.Attribute) and _is_name(f.value, 'self') and f.attr.value in self.shapes:
                return self.shapes[f.attr.value]
            return None
        if isinstance(node, cst.Name):
            return self.env.get(node.value)
        if isinstance(node, cst.Subscript):
            base = self.kind(node.value)
            if isinstance(base, tuple):
                k = _int_literal(_index(node)) if _index(node) is not None else None
                if k is not None and -len(base[1]) <= k < len(base[1]):
                    return base[1][k]
                return None
            if base in ('V', 'I', 'AV', 'AI') and _slice(node) is not None:
                return 'A' + base[-1]
            return None
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Multiply):
            for one, other in ((node.left, node.right), (node.right, node.left)):
                if (isinstance(one, cst.List) and len(one.elements) == 1
                        and isinstance(one.elements[0], cst.Element)
                        and not isinstance(other, (cst.List, cst.ListComp))):
                    elem = self.scalar(one.elements[0].value)
                    return {FLOAT: 'AV', INT: 'AI'}.get(elem, 'A?')
            return None
        if isinstance(node, cst.List) and not node.elements:
            return 'A?'
        if isinstance(node, cst.List):
            # D87: a display whose elements are all typed floats (all typed integers) is a fresh
            # vector written element by element; mixed or untyped elements keep the list path
            if not all(isinstance(e, cst.Element) for e in node.elements):
                return None
            kinds = {self.scalar(e.value) for e in node.elements}
            return {frozenset({FLOAT}): 'AV', frozenset({INT}): 'AI'}.get(frozenset(kinds))
        if isinstance(node, cst.ListComp):
            if id(node) in self.statement_comps and self.comp_ok(node):
                elem = self.comp_elem(node)
                return {FLOAT: 'AV', INT: 'AI'}.get(elem, 'A?')
            return None
        if isinstance(node, cst.IfExp):
            a, b = self.kind(node.body), self.kind(node.orelse)
            if a and b and isinstance(a, str) and isinstance(b, str) and a[-1] == b[-1] and a[-1] in 'VI':
                return 'A' + a[-1] if a[0] == b[0] == 'A' else a[-1]
        return None

    def elem(self, node):
        """'V' or 'I' when the node is a (typed) vector, else None. A slice is a vector only
        where it is stored into one (``destination``); elsewhere it is a list."""
        if isinstance(node, cst.Subscript) and _index(node) is None:
            return None
        k = self.kind(node)
        return k[-1] if isinstance(k, str) and k[-1] in 'VI' else None

    def scalar(self, node):
        """FLOAT, INT or None: the proven scalar kind of an expression (element typer)."""
        if isinstance(node, cst.Float):
            return FLOAT
        if isinstance(node, cst.Integer):
            return INT
        if isinstance(node, cst.Name):
            for frame in reversed(self.comp_scalars):
                if node.value in frame:
                    return frame[node.value]
            return self.scalars.get(node.value)
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, (cst.Minus, cst.Plus)):
            return self.scalar(node.expression)
        if isinstance(node, cst.BinaryOperation):
            a, b = self.scalar(node.left), self.scalar(node.right)
            if a is None or b is None:
                return None
            op = node.operator
            if isinstance(op, cst.Divide):
                return FLOAT
            if NUM in (a, b):
                if isinstance(op, (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo)):
                    return FLOAT if FLOAT in (a, b) else NUM
                return None
            if isinstance(op, cst.Power):
                return FLOAT if b == FLOAT else None
            if isinstance(op, (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo)):
                return FLOAT if FLOAT in (a, b) else INT
            return None
        if isinstance(node, cst.IfExp):
            a, b = self.scalar(node.body), self.scalar(node.orelse)
            return a if a == b else None
        if isinstance(node, cst.Subscript):
            e = self.elem(node.value)
            if e is not None and _index(node) is not None:
                return FLOAT if e == 'V' else INT
            return None
        if isinstance(node, cst.Attribute) and _is_name(node.value, 'self'):
            return self.ref_scalars.get(node.attr.value)
        if isinstance(node, cst.Call):
            f = node.func
            if isinstance(f, cst.Attribute) and _is_name(f.value, 'self'):
                return self.cell_scalar(f.attr.value)
            math = self.math_member(f)
            if math in MATH_FLOAT and len(node.args) == 1 and node.args[0].keyword is None \
                    and node.args[0].star == '' and self.scalar(node.args[0].value) in (FLOAT, INT):
                return FLOAT        # D87: math.exp/log/sqrt/erf of a real number is a float
            if (isinstance(f, cst.Name) and f.value in ('min', 'max') and f.value in self.builtins
                    and len(node.args) >= 2 and all(a.keyword is None and a.star == '' for a in node.args)):
                kinds = {self.scalar(a.value) for a in node.args}
                if None in kinds:
                    return None
                return kinds.pop() if len(kinds) == 1 else NUM
            if isinstance(f, cst.Name) and f.value in self.builtins and _plain_args(node, 1, 1):
                arg = node.args[0].value
                if f.value == 'len' and self.elem(arg):
                    return INT
                if f.value == 'sum':
                    e = self.elem(arg)
                    if e is None and isinstance(arg, cst.Subscript) and _slice(arg) is not None:
                        e = self.elem(arg.value)
                    if e is not None:
                        return FLOAT if e == 'V' else INT
        return None

    def math_member(self, f):
        """The math function name of ``self.<math Reference>.<fn>``, else None (helpers.module_math)."""
        if not (isinstance(f, cst.Attribute) and isinstance(f.value, cst.Attribute)
                and _is_name(f.value.value, 'self')):
            return None
        ref = getattr(self.cls_info, 'refs', {}).get(f.value.attr.value)
        member = getattr(ref, 'helper_members', {}).get(f.attr.value)
        return getattr(member, 'module_math', None)

    def iter_vectors(self, iterator):
        """The vector operands of a loop source over vectors, else None."""
        if self.elem(iterator):
            return [iterator]
        if (isinstance(iterator, cst.Call) and _is_name(iterator.func, 'zip') and 'zip' in self.builtins
                and iterator.args and all(a.keyword is None and a.star == '' and self.elem(a.value)
                                          for a in iterator.args)):
            return [a.value for a in iterator.args]
        return None

    def range_args(self, iterator, names=False):
        """The arguments of ``range(a[, b])``; with ``names``, a local bound only to one
        gives ``[NAME.start, NAME.stop]`` (read once, step 1)."""
        if (isinstance(iterator, cst.Call) and _is_name(iterator.func, 'range') and 'range' in self.builtins
                and _plain_args(iterator, 1, 2)):
            return [a.value for a in iterator.args]
        if names and _is_name(iterator) and iterator.value in self.range_names:
            return [_expr(f'{iterator.value}.start'), _expr(f'{iterator.value}.stop')]
        return None

    def targets(self, target, count):
        if count == 1 and isinstance(target, cst.Name):
            return [target.value]
        if isinstance(target, cst.Tuple) and len(target.elements) == count and all(
                isinstance(e.value, cst.Name) for e in target.elements):
            return [e.value.value for e in target.elements]
        return None

    def comp_ok(self, node, ifs_ok=False):
        comp = node.for_in
        if comp.inner_for_in is not None or comp.asynchronous is not None:
            return False
        if comp.ifs and not ifs_ok:
            return False
        vecs = self.iter_vectors(comp.iter)
        if vecs is not None:
            return self.targets(comp.target, len(vecs)) is not None
        return self.range_args(comp.iter, True) is not None and isinstance(comp.target, cst.Name)

    def loop_scalars(self, comp):
        """Scalar kinds of a comprehension's (or loop's) targets."""
        vecs = self.iter_vectors(comp.iter)
        if vecs is not None:
            names = self.targets(comp.target, len(vecs)) or []
            return {name: FLOAT if self.elem(v) == 'V' else INT for name, v in zip(names, vecs)}
        if self.range_args(comp.iter, True) is not None and isinstance(comp.target, cst.Name):
            return {comp.target.value: INT}
        return {}

    def comp_elem(self, node):
        self.comp_scalars.append(self.loop_scalars(node.for_in))
        try:
            return self.scalar(node.elt)
        finally:
            self.comp_scalars.pop()

    # ---------------------------------------------------------------- analysis
    def analyse(self, body):
        facts = _Facts()
        body.visit(facts)
        self.facts = facts
        self.statement_comps = facts.statement_comps
        assigns = facts.assigns
        for name, kind in self.param_scalars.items():
            if name not in assigns and name not in facts.comp_targets:
                self.scalars[name] = kind
        self.range_names = {name for name, items in assigns.items() if name not in facts.comp_targets
                            and len(items) == 1 and items[0][0] == 'value'
                            and self.range_args(items[0][1]) is not None}
        returned = self.returned_kinds(facts)
        changed, rounds = True, 0
        while changed and rounds < 50:
            changed, rounds = False, rounds + 1
            for name, items in assigns.items():
                if name in facts.comp_targets:
                    continue        # also bound inside a comprehension: never typed
                new = self.vector_kind(name, items, returned)
                if self.env.get(name) != new:
                    self.env[name] = new
                    changed = True
                s = self.scalar_kind(items)
                if self.scalars.get(name) != s:
                    self.scalars[name] = s
                    changed = True
        # an integer vector local whose writes or appends are provably floats stays a list
        for base, value in facts.writes + [(b, v) for b, v, _ in facts.appends]:
            if _is_name(base) and self.env.get(base.value) == 'I' and value is not None \
                    and self.scalar(value) == FLOAT:
                self.env[base.value] = None
                self.fallback(base, 'integer vector written with a float: kept as a list')
            # D94: a float vector local written or appended with a proven int or a mixed min/max stays a
            # list (the list would hold the int unchanged); in a module without vector cells, also with a
            # value not proven a float at all (an object the list would hold as it is)
            elif _is_name(base) and self.env.get(base.value) == 'V' and value is not None \
                    and (self.scalar(value) in (INT, NUM) or (self.strict_locals and self.scalar(value) != FLOAT)):
                self.env[base.value] = None
                self.fallback(base, 'float vector written with a value not proven a float: kept as a list')
            elif _is_name(base) and self.env.get(base.value) == 'I' and value is not None \
                    and self.strict_locals and self.scalar(value) != INT:
                self.env[base.value] = None
                self.fallback(base, 'integer vector written with a value not proven an integer: kept as a list')
        self.owned = {name for name, items in assigns.items() if self.env.get(name) in ('V', 'I')
                      and all(item[0] == 'value' and isinstance(self.kind(item[1]), str)
                              and self.kind(item[1])[0] == 'A' for item in items)}

    def returned_kinds(self, facts):
        """Kinds that a local takes from the cell's own traced shape when it is returned;
        'X' for a local returned where the cell's value is not a vector (it stays a list)."""
        shape = self.cells.vector_shape
        out = {}
        for value in facts.returns:
            if _is_name(value):
                out.setdefault(value.value, set()).add(shape if shape in ('V', 'I') else 'X')
            elif isinstance(value, cst.Tuple):
                kinds = shape[1] if isinstance(shape, tuple) and len(shape[1]) == len(value.elements) else None
                for i, e in enumerate(value.elements):
                    if _is_name(e.value):
                        out.setdefault(e.value.value, set()).add(kinds[i] if kinds else 'X')
        return {name: kinds.pop() if len(kinds) == 1 else 'X' for name, kinds in out.items()}

    def _item_kinds(self, items):
        kinds = []
        for item in items:
            if item[0] == 'value':
                kinds.append(self.kind(item[1]))
            elif item[0] == 'unpack':
                k = self.kind(item[1])
                kinds.append(k[1][item[2]] if isinstance(k, tuple) and len(k[1]) == item[3] else None)
            else:
                kinds.append(None)
        return kinds

    def vector_kind(self, name, items, returned):
        kinds = self._item_kinds(items)
        if (kinds and any(k is None for k in kinds) and self.env.get(name) is None
                and all(item[0] == 'value' for item in items)):
            # diag-kr: a vector rebuilt from itself (``val = [v * g for v in val]``) is typed when
            # assuming the kind its other assignments give makes every assignment that kind: the
            # first assignment executed is one of the others, so the assumption holds inductively
            known = {k[-1] for k in kinds if isinstance(k, str)} - {'?'}
            if len(known) == 1:
                candidate = known.pop()
                self.env[name] = candidate
                try:
                    trial = self._item_kinds(items)
                finally:
                    self.env[name] = None
                if all(isinstance(k, str) and k[-1] in (candidate, '?') for k in trial):
                    kinds = trial
        if not kinds or any(k is None for k in kinds) or returned.get(name) == 'X':
            return None
        if all(isinstance(k, tuple) for k in kinds):
            return kinds[0] if len(set(kinds)) == 1 else None
        if any(isinstance(k, tuple) for k in kinds):
            return None
        elems = {k[-1] for k in kinds} - {'?'}
        if not elems and name in returned:
            elems = {returned[name]}
        if '?' in {k[-1] for k in kinds} and not elems:
            elems = self.appended_kinds(name)
        if len(elems) != 1:
            return None
        if name in returned and returned[name] not in elems:
            return None
        return elems.pop()

    def appended_kinds(self, name):
        kinds = {self.scalar(v) for b, v, _ in self.facts.appends if _is_name(b, name)}
        kinds |= {self.scalar(v) for b, v in self.facts.writes if _is_name(b, name) and v is not None}
        if len(kinds) == 1 and kinds <= {FLOAT, INT}:
            return {'V' if kinds.pop() == FLOAT else 'I'}
        return set()

    def scalar_kind(self, items):
        kinds = set()
        for item in items:
            if item[0] == 'value':
                kinds.add(self.scalar(item[1]))
            elif item[0] == 'aug':
                kinds.add(self._aug_kind(item))
            elif item[0] == 'for':
                it = item[1]
                if self.range_args(it) is not None and item[3] == 1:
                    kinds.add(INT)
                else:
                    vecs = self.iter_vectors(it)
                    if vecs is not None and len(vecs) == item[3]:
                        kinds.add(FLOAT if self.elem(vecs[item[2]]) == 'V' else INT)
                    else:
                        kinds.add(None)
            else:
                kinds.add(None)
        return kinds.pop() if len(kinds) == 1 else None

    def _aug_kind(self, item):
        op, value = item[1], item[2]
        v = self.scalar(value)
        if v not in (FLOAT, INT):
            return None
        if isinstance(op, cst.DivideAssign):
            return FLOAT
        if isinstance(op, (cst.AddAssign, cst.SubtractAssign, cst.MultiplyAssign)):
            return v if v == FLOAT else INT
        return None

    # ---------------------------------------------------------------- helpers
    def temp(self, label):
        self.counter[0] += 1
        name = f'_mx_v{self.counter[0]}_{label}'
        self.temps.add(name)
        return name

    def declare(self, name, typ):
        self.decls.append(_stmt(f'{name}: {typ}\n'))

    def fallback(self, node, why):
        self.fallbacks.append(f'{why}: {_code(node)[:80]}')

    def call(self, op, elem, *args):
        return cst.Call(_expr(fn(op, elem)), [cst.Arg(a) for a in args])

    # ---------------------------------------------------------------- expressions
    def leave_Subscript(self, original, updated):
        if id(original) in self.facts.store_targets:
            return updated
        e = self.elem(original.value)
        if e is not None:
            index = _index(updated)
            if index is not None:
                self.lowered += 1
                return self.call('get', e, updated.value, index)
            if _slice(original) is None:
                self.fallback(original, 'slice of a vector with a step: a list through the sequence protocol')
            elif id(original) not in self.sum_slices and id(original) not in self.alloc_slices:
                self.fallback(original, 'slice of a vector outside sum or an assignment: '
                                        'a list through the sequence protocol')
            return updated
        k = self.kind(original)
        if k in ('V', 'I') and isinstance(self.kind(original.value), tuple):
            return self.call('of', k, updated)
        return updated

    def scalar_only_power(self, node):
        """D91: a power whose operands read no vector, vector element or local the vector typer
        typed (only parameters, self/Reference attributes, Cell calls and literals), outside any
        comprehension this lowering expands: powers.py types it as before, with its one-entry
        memo where enabled; the vector lowering leaves it alone."""
        if self.comp_scalars:
            return False
        allowed = set(self.cells.params) | {'self'}

        class Probe(cst.CSTVisitor):
            ok = True

            def visit_Attribute(inner, item):
                item.value.visit(inner)     # the attribute name itself is not a variable
                return False

            def visit_Subscript(inner, item):
                inner.ok = False
                return False

            def visit_Name(inner, item):
                if item.value not in allowed and item.value not in ('True', 'False', 'None', 'float', 'int'):
                    inner.ok = False

            def visit_Call(inner, item):
                if isinstance(item.func, cst.Name) and item.func.value not in ('float', 'int'):
                    if item.func.value not in allowed:
                        inner.ok = False
                elif not isinstance(item.func, cst.Name):
                    item.func.visit(inner)
                for arg in item.args:
                    arg.value.visit(inner)
                return False

            def visit_Lambda(inner, item):
                inner.ok = False
                return False

        probe = Probe()
        node.left.visit(probe)
        node.right.visit(probe)
        return probe.ok

    def leave_BinaryOperation(self, original, updated):
        if isinstance(original.operator, cst.Power) and self.scalar_only_power(original):
            return updated
        if isinstance(original.operator, cst.Power) and self.scalar(original.right) == FLOAT \
                and self.scalar(original.left) in (FLOAT, INT) and self.in_lowered_scope(original):
            self.powers += 1
            base = updated.left
            if self.scalar(original.left) == INT:
                base = cst.Call(_expr(f'{CY_MOD}.cast'), [cst.Arg(_expr(f'{CY_MOD}.double')),
                                                          cst.Arg(base.with_changes(lpar=(), rpar=()))])
            return cst.Call(_expr(f'{MX_SYS_MOD}._mx_pow'), [cst.Arg(base), cst.Arg(updated.right)],
                            lpar=updated.lpar, rpar=updated.rpar)
        return updated

    def in_lowered_scope(self, node):
        """Powers are rewritten where every name resolves to the formula's own bindings:
        outside comprehensions, or inside comprehensions this lowering expands."""
        return all(self.scope_stack)

    def visit_ListComp(self, node):
        expand = id(node) in self.statement_comps and self.comp_ok(node)
        self.scope_stack.append(expand)
        if expand:
            self.comp_scalars.append(self.loop_scalars(node.for_in))
            self.expanding.add(id(node))
        return True

    def visit_GeneratorExp(self, node):
        expand = id(node) in self.sum_gens
        self.scope_stack.append(expand)
        if expand:
            self.comp_scalars.append(self.loop_scalars(node.for_in))
            self.expanding.add(id(node))
        return True

    def visit_SetComp(self, node):
        self.scope_stack.append(False)
        return True

    def visit_DictComp(self, node):
        self.scope_stack.append(False)
        return True

    def visit_Lambda(self, node):
        self.scope_stack.append(False)
        return True

    def leave_SetComp(self, original, updated):
        self.scope_stack.pop()
        return updated

    def leave_DictComp(self, original, updated):
        self.scope_stack.pop()
        return updated

    def leave_Lambda(self, original, updated):
        self.scope_stack.pop()
        return updated

    def leave_GeneratorExp(self, original, updated):
        self.scope_stack.pop()
        if id(original) in self.expanding:
            self.comp_scalars.pop()
        elif self.iter_vectors(original.for_in.iter):
            self.fallback(original, 'generator over vectors not lowered')
        return updated

    def leave_ListComp(self, original, updated):
        self.scope_stack.pop()
        if id(original) in self.expanding:
            self.comp_scalars.pop()
        if id(original) not in self.expanding and self.iter_vectors(original.for_in.iter):
            self.fallback(original, 'list comprehension over vectors not lowered')
        return updated

    def leave_Call(self, original, updated):
        f = original.func
        if isinstance(f, cst.Attribute) and f.attr.value == 'append' and len(original.args) == 1:
            e = self.elem(f.value)
            if e is not None:
                base = f.value
                if not (_is_name(base) and base.value in self.owned):
                    raise VectorWriteError(
                        f'{self.cells.model_path}: append to vector {_code(base)!r}, which this formula '
                        f'did not allocate; cached vectors are immutable: build a new one')
                self.lowered += 1
                return self.call('append', e, updated.func.value, updated.args[0].value)
        if isinstance(f, cst.Name) and f.value in self.builtins and _plain_args(original, 1, 1):
            arg_o, arg_u = original.args[0].value, updated.args[0].value
            e = self.elem(arg_o)
            if f.value == 'len' and e:
                self.lowered += 1
                return cst.Attribute(arg_u, cst.Name('n'))
            if f.value == 'sum':
                if e:
                    self.lowered += 1
                    return self.call('sum', e, arg_u)
                if id(arg_o) in self.sum_slices:
                    sl = _slice(arg_u)
                    lo = sl.lower if sl.lower is not None else _expr('0')
                    hi = sl.upper if sl.upper is not None else _expr(LL_MAX)
                    self.lowered += 1
                    return self.call('sum_range', self.elem(arg_o.value), arg_u.value, lo, hi)
        if isinstance(f, cst.Name) and f.value in self.builtins | {'min', 'max', 'any', 'all', 'sorted', 'list',
                                                                  'tuple', 'enumerate', 'reversed', 'map'}:
            for a in original.args:
                if self.elem(a.value) and f.value not in ('len', 'sum', 'zip', 'range'):
                    self.fallback(original, f'{f.value}() over a vector runs through the sequence protocol')
        return updated

    # ---------------------------------------------------------------- statements
    def leave_SimpleStatementLine(self, original, updated):
        if len(original.body) != 1:
            return updated
        so, su = original.body[0], updated.body[0]
        if isinstance(so, cst.AugAssign) and isinstance(so.target, cst.Subscript) and self.elem(so.target.value):
            return self.element_update(original, updated, so, su)
        if isinstance(so, cst.Assign) and len(so.targets) == 1:
            tgt_o = so.targets[0].target
            if isinstance(tgt_o, cst.Subscript) and self.elem(tgt_o.value):
                base = tgt_o.value
                self.check_owned(base, 'assignment to an element of')
                if _index(tgt_o) is not None:
                    self.lowered += 1
                    tgt_u = su.targets[0].target
                    call = self.call('set', self.elem(base), tgt_u.value, _index(tgt_u), su.value)
                    return updated.with_changes(body=[cst.Expr(call)])
                raise VectorWriteError(f'{self.cells.model_path}: slice assignment to vector '
                                       f'{_code(base)!r} is not supported: write the elements')
            if isinstance(tgt_o, (cst.Tuple, cst.List)) and isinstance(self.kind(so.value), tuple):
                names = [e.value for e in tgt_o.elements]
                if all(_is_name(n) and self.env.get(n.value) in ('V', 'I') for n in names):
                    temps = [self.temp('u') for _ in names]
                    stmts = [_stmt(f'{", ".join(temps)} = {_code(su.value)}\n')]
                    stmts += [_stmt(f'{n.value} = {fn("of", self.env[n.value])}({t})\n')
                              for n, t in zip(names, temps)]
                    self.lowered += 1
                    return cst.FlattenSentinel([updated.with_changes(body=s.body) for s in stmts])
            if _is_name(tgt_o) and self.env.get(tgt_o.value) in ('V', 'I'):
                pre, value = self.destination(so.value, su.value, self.env[tgt_o.value])
                return self.emit(updated, su.with_changes(value=value), pre)
        if isinstance(so, (cst.Assign, cst.Return)) and id(getattr(so, 'value', None)) in self.extreme_calls:
            name, elem = self.extreme_calls[id(so.value)]
            pre, result = self.expand_extreme(so.value.args[0].value, su.value.args[0].value, name, elem)
            su = su.with_changes(value=result)
            if isinstance(so, cst.Return):
                su = self.wrap_return(so, su, [])[1]
            return self.emit(updated, su, pre)
        if isinstance(so, (cst.Assign, cst.Return)) and id(so.value) in self.leading_sums:
            # diag-us2 (D93): the sum evaluated first in a +/- chain, computed just before the statement
            lead_o, lead_u = _leading_operand(so.value), _leading_operand(su.value)
            pre, result = self.expand_sum(lead_o.args[0].value, lead_u.args[0].value)
            su = su.with_changes(value=_replace_leading(su.value, result))
            if isinstance(so, cst.Return):
                su = self.wrap_return(so, su, [])[1]
            return self.emit(updated, su, pre)
        if isinstance(so, (cst.Assign, cst.Return, cst.AugAssign)) and id(getattr(so, 'value', None)) in self.sum_calls:
            pre, result = self.expand_sum(so.value.args[0].value, su.value.args[0].value)
            su = su.with_changes(value=result)
            if isinstance(so, cst.Return):
                su = self.wrap_return(so, su, [])[1]
            return self.emit(updated, su, pre)
        if isinstance(so, cst.Return) and so.value is not None:
            pre, su = self.wrap_return(so, su, [])
            return self.emit(updated, su, pre)
        return updated

    def emit(self, line, stmt, pre):
        line = line.with_changes(body=[stmt])
        return cst.FlattenSentinel([*pre, line]) if pre else line

    def check_owned(self, base, what):
        if not (_is_name(base) and base.value in self.owned):
            raise VectorWriteError(
                f'{self.cells.model_path}: {what} vector {_code(base)!r}, which this formula did not '
                f'allocate; cached vectors are immutable: allocate a new one ([0.0] * n) and write that')

    def element_update(self, line, updated, so, su):
        """v[i] op= x on an owned vector: one index evaluation, then get and set."""
        base, e = so.target.value, self.elem(so.target.value)
        self.check_owned(base, 'update of an element of')
        ops = {cst.AddAssign: '+', cst.SubtractAssign: '-', cst.MultiplyAssign: '*'}
        ops.update({cst.DivideAssign: '/'} if e == 'V' else {cst.FloorDivideAssign: '//', cst.ModuloAssign: '%'})
        op = ops.get(type(so.operator))
        index = _index(su.target)
        if op is None or index is None:
            raise VectorWriteError(f'{self.cells.model_path}: unsupported element update of vector '
                                   f'{_code(base)!r}: {_code(so).strip()}')
        k = self.temp('k')
        self.declare(k, f'{CY_MOD}.longlong')
        v = _code(su.target.value)
        self.lowered += 1
        return cst.FlattenSentinel([
            updated.with_changes(body=[_stmt(f'{k} = {_code(index)}\n').body[0]]),
            updated.with_changes(body=[_stmt(
                f'{fn("set", e)}({v}, {k}, {fn("get", e)}({v}, {k}) {op} ({_code(su.value)}))\n').body[0]])])

    def destination(self, value_o, value_u, e):
        """(statements, value) storing value_o into a typed vector of kind e: allocations
        are built in place, typed vectors pass, anything else is converted (vec_of)."""
        k = self.kind(value_o)
        if isinstance(value_o, cst.ListComp) and isinstance(k, str) and k[0] == 'A':
            return self.expand_comp(value_o, value_u, e)
        if isinstance(value_o, cst.BinaryOperation) and isinstance(k, str) and k[0] == 'A':
            if isinstance(value_u.left, cst.List):
                item, count = value_u.left.elements[0].value, value_u.right
            else:
                item, count = value_u.right.elements[0].value, value_u.left
            self.lowered += 1
            return [], self.call('full', e, count, item)
        if isinstance(value_o, cst.List) and not value_o.elements:
            self.lowered += 1
            return [], self.call('new', e, cst.Integer('0'))
        if isinstance(value_o, cst.List) and k == 'A' + e:
            return self.display(value_u, e)
        if isinstance(value_o, cst.Subscript) and id(value_o) in self.alloc_slices and self.elem(value_o.value) == e:
            sl = _slice(value_u)
            lo = sl.lower if sl.lower is not None else _expr('0')
            hi = sl.upper if sl.upper is not None else _expr(LL_MAX)
            self.lowered += 1
            return [], self.call('slice', e, value_u.value, lo, hi)
        if self.elem(value_o) == e and not (isinstance(k, str) and k[0] == 'A'):
            return [], value_u
        if _is_name(value_u) and value_u.value in self.temps:
            return [], value_u
        concat = self.concatenation(value_o, value_u, e)
        if concat is not None:
            return concat
        return [], self.call('of', e, value_u)

    def display(self, value_u, e):
        """D87: ``[a, b, ...]`` of typed scalars of kind e into a new typed vector: the elements are
        evaluated left to right into typed locals first (an exception leaves no vector), then the
        vector is allocated and written; no Python list."""
        pre, temps = [], []
        for element in value_u.elements:
            t = self.temp('e')
            self.declare(t, SCALARS[e])
            pre.append(_stmt(f'{t} = ({_code(element.value)})\n'))
            temps.append(t)
        r = self.temp('d')
        self.declare(r, TYPES[e])
        pre.append(_stmt(f'{r} = {fn("new", e)}({len(temps)})\n'))
        pre += [_stmt(f'{r}.data[{i}] = {t}\n') for i, t in enumerate(temps)]
        self.lowered += 1
        return pre, cst.Name(r)

    def concatenation(self, value_o, value_u, e):
        """diag-kr: ``[... for ...] + list(range(a, b)) + ...`` stored into a typed vector of kind
        e, each operand built in turn (left to right, as Python evaluates them) and copied into
        one allocation, instead of Python lists converted by vec_of. None when an operand is not
        of kind e."""
        parts_o = self.facts.concat_values.get(id(value_o))
        if parts_o is None or 'list' not in self.builtins or 'range' not in self.builtins:
            return None
        parts_u = []
        def flatten(node):
            if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Add):
                flatten(node.left)
                flatten(node.right)
            else:
                parts_u.append(node)
        flatten(value_u)
        if len(parts_u) != len(parts_o):
            return None
        for part in parts_o:
            if isinstance(part, cst.ListComp):
                if self.kind(part) != 'A' + e:
                    return None
            elif e != 'I' or any(self.scalar(arg) != INT for arg in _list_range(part)):
                return None
        pre, names = [], []
        for po, pu in zip(parts_o, parts_u):
            if isinstance(po, cst.ListComp):
                more, value = self.expand_comp(po, pu, e)
                pre += more
                names.append(value.value)
                continue
            args = [_code(a) for a in _list_range(pu)]
            lo, hi = ('0', args[0]) if len(args) == 1 else (args[0], args[1])
            lo_t, hi_t, n, i, r = (self.temp('clo'), self.temp('chi'), self.temp('cn'), self.temp('ci'),
                                   self.temp('cr'))
            for name in (lo_t, hi_t):
                self.declare(name, f'{CY_MOD}.longlong')
            for name in (n, i):
                self.declare(name, f'{CY_MOD}.Py_ssize_t')
            self.declare(r, TYPES[e])
            pre += [_stmt(f'{lo_t} = ({lo}\n)\n'), _stmt(f'{hi_t} = ({hi}\n)\n'),
                    _stmt(f'{n} = {hi_t} - {lo_t} if {hi_t} > {lo_t} else 0\n'),
                    _stmt(f'{r} = {fn("new", e)}({n})\n'),
                    _stmt(f'for {i} in range({n}):\n    {r}.data[{i}] = {lo_t} + {i}\n')]
            names.append(r)
        out, at, i = self.temp('cat'), self.temp('cat_at'), self.temp('cat_i')
        self.declare(out, TYPES[e])
        self.declare(at, f'{CY_MOD}.Py_ssize_t')
        self.declare(i, f'{CY_MOD}.Py_ssize_t')
        pre.append(_stmt(f'{out} = {fn("new", e)}({" + ".join(f"{name}.n" for name in names)})\n'))
        pre.append(_stmt(f'{at} = 0\n'))
        for name in names:
            pre.append(_stmt(f'for {i} in range({name}.n):\n    {out}.data[{at} + {i}] = {name}.data[{i}]\n'))
            pre.append(_stmt(f'{at} += {name}.n\n'))
        self.lowered += 1
        return pre, cst.Name(out)

    def wrap_return(self, so, su, pre):
        shape = self.cells.vector_shape
        if shape in ('V', 'I'):
            more, value = self.destination(so.value, su.value, shape)
            return pre + more, su.with_changes(value=value)
        if isinstance(shape, tuple) and isinstance(so.value, cst.Tuple) and isinstance(su.value, cst.Tuple) \
                and len(so.value.elements) == len(shape[1]):
            elements = []
            for eo, eu, k in zip(so.value.elements, su.value.elements, shape[1]):
                more, value = self.destination(eo.value, eu.value, k)
                pre = pre + more
                elements.append(eu.with_changes(value=value))
            return pre, su.with_changes(value=su.value.with_changes(elements=elements))
        if self.kind(so.value) in ('V', 'I'):
            # a typed vector returned where the cell's value is not a vector: give a list back
            return pre, su.with_changes(value=cst.Call(cst.Attribute(su.value, cst.Name('tolist'))))
        return pre, su

    def loop_head(self, comp_o, comp_u, tag):
        """Statements before the loop, the loop header text, loop-variable lines and renames."""
        pre, renames = [], {}
        vecs_o = self.iter_vectors(comp_o.iter)
        i = self.temp(f'{tag}i')
        n = self.temp(f'{tag}n')
        self.declare(i, f'{CY_MOD}.Py_ssize_t')
        self.declare(n, f'{CY_MOD}.Py_ssize_t')
        if vecs_o is not None:
            vecs_u = ([comp_u.iter] if len(vecs_o) == 1 and self.elem(comp_o.iter)
                      else [a.value for a in comp_u.iter.args])
            names = self.targets(comp_o.target, len(vecs_o))
            xs = []
            for v_o, v in zip(vecs_o, vecs_u):
                x = self.temp(f'{tag}x')
                self.declare(x, TYPES[self.elem(v_o)])
                pre.append(_stmt(f'{x} = {_code(v)}\n'))
                xs.append((x, self.elem(v_o)))
            pre.append(_stmt(f'{n} = {xs[0][0]}.n\n'))
            for x, _ in xs[1:]:
                pre.append(_stmt(f'if {x}.n < {n}: {n} = {x}.n\n'))
            body = []
            for name, (x, e) in zip(names, xs):
                new = self.temp(f'{tag}{name}')
                self.declare(new, SCALARS[e])
                renames[name] = new
                self.loop_renames[new] = name
                body.append(f'{new} = {x}.data[{i}]')
            return pre, f'for {i} in range({n}):', body, renames, n, i
        args = [_code(a) for a in self.range_args(comp_u.iter, True)]
        lo, hi = (('0', args[0]) if len(args) == 1 else (args[0], args[1]))
        lo_t, hi_t = self.temp(f'{tag}lo'), self.temp(f'{tag}hi')
        self.declare(lo_t, f'{CY_MOD}.longlong')
        self.declare(hi_t, f'{CY_MOD}.longlong')
        z = self.temp(f'{tag}{comp_o.target.value}')
        self.declare(z, f'{CY_MOD}.longlong')
        renames[comp_o.target.value] = z
        self.loop_renames[z] = comp_o.target.value
        pre += [_stmt(f'{lo_t} = ({lo}\n)\n'), _stmt(f'{hi_t} = ({hi}\n)\n'),
                _stmt(f'{n} = {hi_t} - {lo_t} if {hi_t} > {lo_t} else 0\n')]
        return pre, f'for {i} in range({n}):', [f'{z} = {lo_t} + {i}'], renames, n, i

    def expand_comp(self, comp_o, comp_u, e):
        pre, head, body, renames, n, i = self.loop_head(comp_o.for_in, comp_u.for_in, 'c')
        r = self.temp('r')
        self.declare(r, TYPES[e])
        elt = _code(comp_u.elt.visit(_Rename(renames)))
        pre.append(_stmt(f'{r} = {fn("new", e)}({n})\n'))
        # the element may span lines (as written in the formula): parenthesised, closed on its own line
        loop = '\n'.join([head] + ['    ' + line for line in body] + [f'    {r}.data[{i}] = ({elt}\n)']) + '\n'
        pre.append(_stmt(loop))
        self.lowered += 1
        return pre, cst.Name(r)

    def expand_sum(self, gen_o, gen_u):
        pre, head, body, renames, n, i = self.loop_head(gen_o.for_in, gen_u.for_in, 's')
        s = self.temp('sum')
        self.declare(s, f'{ALIAS}.VecSum')
        elt = _code(gen_u.elt.visit(_Rename(renames)))
        pre.append(_stmt(f'{ALIAS}.sum_init({CY_MOD}.address({s}))\n'))
        add = f'{ALIAS}.sum_add({CY_MOD}.address({s}), {elt})'
        if gen_u.for_in.ifs:
            cond = ' and '.join(f'({_code(c.test.visit(_Rename(renames)))})' for c in gen_u.for_in.ifs)
            lines = body + [f'if {cond}:', f'    {add}']
        else:
            lines = body + [add]
        pre.append(_stmt('\n'.join([head] + ['    ' + line for line in lines]) + '\n'))
        self.lowered += 1
        return pre, _expr(f'{ALIAS}.sum_result({CY_MOD}.address({s}))')

    def expand_extreme(self, gen_o, gen_u, name, elem):
        """diag-kr: ``min``/``max`` of a generator over vectors, as Python computes it: the first
        element that passes the filters is the current extreme, a later one replaces it only when
        ``element < current`` (``>`` for max), and an empty result raises ValueError."""
        pre, head, body, renames, n, i = self.loop_head(gen_o.for_in, gen_u.for_in, 'm')
        best, seen, value = self.temp('best'), self.temp('seen'), self.temp('item')
        ctype = SCALARS['V' if elem == FLOAT else 'I']
        self.declare(best, ctype)
        self.declare(value, ctype)
        self.declare(seen, f'{CY_MOD}.bint')
        elt = _code(gen_u.elt.visit(_Rename(renames)))
        op = '<' if name == 'min' else '>'
        pre.append(_stmt(f'{seen} = False\n'))
        pre.append(_stmt(f'{best} = 0\n'))
        update = [f'{value} = {elt}', f'if not {seen} or {value} {op} {best}:',
                  f'    {best} = {value}', f'    {seen} = True']
        if gen_u.for_in.ifs:
            cond = ' and '.join(f'({_code(c.test.visit(_Rename(renames)))})' for c in gen_u.for_in.ifs)
            lines = body + [f'if {cond}:'] + ['    ' + line for line in update]
        else:
            lines = body + update
        pre.append(_stmt('\n'.join([head] + ['    ' + line for line in lines]) + '\n'))
        from .reductions import _EMPTY
        pre.append(_stmt(f'if not {seen}:\n    raise ValueError("{_EMPTY.format(name=name)}")\n'))
        self.lowered += 1
        return pre, cst.Name(best)

    # ---------------------------------------------------------------- for statements
    def visit_For(self, node):
        vecs = self.iter_vectors(node.iter)
        if vecs is not None and self.for_vector_ok(node, vecs):
            self.vector_fors.add(id(node))
        return True

    def for_vector_ok(self, node, vecs):
        names = self.targets(node.target, len(vecs))
        if names is None or node.asynchronous is not None:
            return False
        # the loop reads buffers fixed at its start: no append to them inside
        iterated = {v.value for v in vecs if _is_name(v)}
        for b, _, call in self.facts.appends:
            if _is_name(b) and b.value in iterated and any(c is call for c in _walk(node.body)):
                return False
        return True

    def leave_For(self, original, updated):
        if id(original) in self.vector_fors:
            return self.lower_vector_for(original, updated)
        if self.iter_vectors(original.iter):
            self.fallback(original.iter, 'for statement over vectors not lowered')
            return updated
        return self.version_range_for(original, updated)

    def lower_vector_for(self, original, updated):
        vecs_o = self.iter_vectors(original.iter)
        vecs_u = ([updated.iter] if len(vecs_o) == 1 and self.elem(original.iter)
                  else [a.value for a in updated.iter.args])
        names = self.targets(original.target, len(vecs_o))
        i, n = self.temp('fi'), self.temp('fn')
        self.declare(i, f'{CY_MOD}.Py_ssize_t')
        self.declare(n, f'{CY_MOD}.Py_ssize_t')
        pre, xs = [], []
        for v_o, v in zip(vecs_o, vecs_u):
            x = self.temp('fx')
            self.declare(x, TYPES[self.elem(v_o)])
            pre.append(_stmt(f'{x} = {_code(v)}\n'))
            xs.append(x)
        pre.append(_stmt(f'{n} = {xs[0]}.n\n'))
        for x in xs[1:]:
            pre.append(_stmt(f'if {x}.n < {n}: {n} = {x}.n\n'))
        binds = [_stmt(f'{name} = {x}.data[{i}]\n') for name, x in zip(names, xs)]
        body = updated.body.with_changes(body=[*binds, *updated.body.body])
        loop = updated.with_changes(target=cst.Name(i), iter=_expr(f'range({n})'), body=body)
        self.lowered += 1
        return cst.FlattenSentinel([*pre, loop])

    def version_range_for(self, original, updated):
        """Hoist the bounds tests of v[i + c] out of a range loop (loop versioning)."""
        if not (isinstance(original.target, cst.Name) and original.target.value in self.range_locals
                and self.range_args(original.iter) is not None and original.asynchronous is None):
            return updated
        i = original.target.value
        bound = set()           # names bound inside the loop
        for node in _walk(original.body):
            if isinstance(node, (cst.Assign, cst.AugAssign, cst.AnnAssign, cst.For, cst.CompFor,
                                 cst.NamedExpr, cst.Del)):
                targets = (node.targets if isinstance(node, cst.Assign) else [node])
                for t in targets:
                    bound.update(_names(t.target))
            if isinstance(node, (cst.FunctionDef, cst.Lambda, cst.ClassDef)):
                return updated
        appended = {b.value for b, _, call in self.facts.appends if _is_name(b)
                    and any(c is call for c in _walk(original.body))}
        if i in bound:
            return updated
        accesses = {}           # (vector name, offset) -> elem
        rewriter = _Unchecked(i, self, bound | appended, accesses)
        fast_body = updated.body.visit(rewriter)
        if not accesses or rewriter.mixed:
            return updated
        args = [a.value for a in updated.iter.args]
        lo_t, hi_t = self.temp('blo'), self.temp('bhi')
        self.declare(lo_t, f'{CY_MOD}.longlong')
        self.declare(hi_t, f'{CY_MOD}.longlong')
        pre = [cst.SimpleStatementLine([cst.Assign([cst.AssignTarget(cst.Name(lo_t))],
                                                   args[0] if len(args) == 2 else cst.Integer('0'))]),
               cst.SimpleStatementLine([cst.Assign([cst.AssignTarget(cst.Name(hi_t))], args[-1])])]
        guards = []
        for (v, c) in sorted(accesses):
            off = f' + {c}' if c > 0 else (f' - {-c}' if c < 0 else '')
            guards.append(f'{v} is not None and {ALIAS}.in_range({lo_t}{off}, {hi_t}{off}, {v}.n)')
        rng = _expr(f'range({lo_t}, {hi_t})')
        fast = updated.with_changes(iter=rng, body=fast_body)
        slow = updated.with_changes(iter=rng)
        test = _expr(' and '.join(f'({g})' for g in guards))
        self.bounds_hoisted += len(accesses)
        self.lowered += 1
        return cst.FlattenSentinel([*pre, cst.If(test=test, body=cst.IndentedBlock([fast]),
                                                 orelse=cst.Else(cst.IndentedBlock([slow])))])


class _Unchecked(cst.CSTTransformer):
    """In a range loop over ``i``: vec_get/vec_set(v, i + c) on vectors not bound in the
    loop become direct buffer reads and writes; records (v, c) for the guard."""

    def __init__(self, i, lowering, bound, accesses):
        self.i = i
        self.lowering = lowering
        self.bound = bound
        self.accesses = accesses
        self.mixed = False
        self.gets = {f'{ALIAS}.{FUNCS["get"][k]}': k for k in (0, 1)}
        self.sets = {f'{ALIAS}.{FUNCS["set"][k]}': k for k in (0, 1)}

    def offset(self, index):
        if _is_name(index, self.i):
            return 0
        if isinstance(index, cst.BinaryOperation) and isinstance(index.operator, (cst.Add, cst.Subtract)) \
                and not index.lpar:
            left, right = index.left, index.right
            if _is_name(left, self.i) and _int_literal(right) is not None:
                c = _int_literal(right)
                return c if isinstance(index.operator, cst.Add) else -c
            if isinstance(index.operator, cst.Add) and _is_name(right, self.i) and _int_literal(left) is not None:
                return _int_literal(left)
        return None

    def target(self, call):
        if len(call.args) < 2:
            return None
        v, index = call.args[0].value, call.args[1].value
        if not _is_name(v) or v.value in self.bound:
            return None
        c = self.offset(index)
        if c is None or abs(c) > 1 << 20:
            return None
        return v.value, c, index

    def visit_FunctionDef(self, node):
        return False

    def leave_Call(self, original, updated):
        name = dotted_name(updated.func)
        if name in self.gets:
            hit = self.target(updated)
            if hit:
                v, c, index = hit
                self.accesses[(v, c)] = True
                return cst.Subscript(cst.Attribute(cst.Name(v), cst.Name('data')),
                                     [cst.SubscriptElement(cst.Index(index))],
                                     lpar=updated.lpar, rpar=updated.rpar)
        return updated

    def leave_Expr(self, original, updated):
        value = updated.value
        if isinstance(value, cst.Call) and dotted_name(value.func) in self.sets:
            hit = self.target(value)
            if hit:
                v, c, index = hit
                self.accesses[(v, c)] = True
                return cst.Assign([cst.AssignTarget(cst.Subscript(
                    cst.Attribute(cst.Name(v), cst.Name('data')),
                    [cst.SubscriptElement(cst.Index(index))]))], value.args[2].value)
        return updated


class _CheckedFloatReads(cst.CSTTransformer):
    """Emit the bounds check at the use site for an index held in a C local."""

    def __init__(self, vectors, indices):
        self.vectors = vectors
        self.indices = indices

    def on_visit(self, node):
        # A nested scope can rebind a name with different typing.
        return not isinstance(node, (cst.FunctionDef, cst.ClassDef, cst.Lambda,
                                     cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp))

    def leave_Call(self, original, updated):
        if dotted_name(updated.func) != fn('get', 'V') or not _plain_args(updated, 2, 2):
            return updated
        vector, index = (arg.value for arg in updated.args)
        if (not isinstance(vector, cst.Name) or self.vectors.get(vector.value) != 'V'
                or not isinstance(index, cst.Name) or index.value not in self.indices):
            return updated
        # Both operands are stable C locals. The fallback still implements
        # Python's negative index and out-of-range behavior.
        v, i = vector.value, index.value
        return _expr(
            f'({v}.data[{i}] if {CY_MOD}.cast({CY_MOD}.ulonglong, {i}) '
            f'< {CY_MOD}.cast({CY_MOD}.ulonglong, {v}.n) else {_code(updated)})'
        )


def range_loop_locals(body, builtins):
    """Names bound only as the target of ``for NAME in range(...)`` (builtin range)."""
    kinds = {}

    class Collect(cst.CSTVisitor):
        def visit_For(inner, node):
            ok = (isinstance(node.target, cst.Name) and 'range' in builtins
                  and isinstance(node.iter, cst.Call) and _is_name(node.iter.func, 'range')
                  and _plain_args(node.iter, 1, 3) and node.asynchronous is None)
            for n in _names(node.target):
                kinds.setdefault(n, set()).add('range' if ok else 'other')

        def visit_AssignTarget(inner, node):
            for n in _names(node.target):
                kinds.setdefault(n, set()).add('other')

        def visit_AugAssign(inner, node):
            for n in _names(node.target):
                kinds.setdefault(n, set()).add('other')

        def visit_AnnAssign(inner, node):
            for n in _names(node.target):
                kinds.setdefault(n, set()).add('other')

        def visit_NamedExpr(inner, node):
            kinds.setdefault(node.target.value, set()).add('other')

        def visit_Param(inner, node):
            kinds.setdefault(node.name.value, set()).add('other')

        def visit_Global(inner, node):
            for n in node.names:
                kinds.setdefault(n.name.value, set()).add('other')

        def visit_Nonlocal(inner, node):
            for n in node.names:
                kinds.setdefault(n.name.value, set()).add('other')

        def visit_Del(inner, node):
            for n in _names(node.target):
                kinds.setdefault(n, set()).add('other')

        def visit_FunctionDef(inner, node):
            return False        # nested functions keep their own locals

        def visit_Lambda(inner, node):
            return False

        def visit_With(inner, node):
            for item in node.items:
                if item.asname is not None:
                    for n in _names(item.asname.name):
                        kinds.setdefault(n, set()).add('other')

        def visit_ExceptHandler(inner, node):
            if node.name is not None:
                kinds.setdefault(node.name.name.value, set()).add('other')

    body.visit(Collect())
    return sorted(name for name, k in kinds.items() if k == {'range'})


def derived_index_locals(facts, loops, params):
    """Locals every assignment of which is an int64 literal, a range loop local or another
    such local, or one of those ``// c`` or ``% c`` for a positive literal c: their values
    lie between the loop bounds and the literals, so a C ``long long`` cannot overflow."""
    candidates = {name for name, items in facts.assigns.items()
                  if name not in loops and name not in params and name not in facts.comp_targets
                  and items and all(item[0] == 'value' for item in items)}

    def ok(expr, known):
        if isinstance(expr, cst.Integer) or _int_literal(expr) is not None:
            v = _int_literal(expr)
            return v is not None and -(1 << 63) <= v < (1 << 63)
        if isinstance(expr, cst.Name):
            return expr.value in known
        if (isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, (cst.FloorDivide, cst.Modulo))
                and isinstance(expr.right, cst.Integer) and int(expr.right.value, 0) > 0):
            return ok(expr.left, known)
        return False

    found = set()
    changed = True
    while changed:
        changed = False
        known = loops | found
        for name in candidates - found:
            if all(ok(item[1], known) for item in facts.assigns[name]):
                found.add(name)
                changed = True
    # every assignment must be in the formula itself, not only proven from names bound later
    return sorted(found)


def _declare_at_top(node, decls):
    if not decls:
        return node
    stmts = list(node.body.body)
    start = 1 if stmts and isinstance(stmts[0], cst.SimpleStatementLine) and isinstance(
        stmts[0].body[0], cst.Expr) and isinstance(stmts[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString)) else 0
    return node.with_changes(body=node.body.with_changes(body=[*stmts[:start], *decls, *stmts[start:]]))


_TEMP = re.compile(r'_mx_v[0-9]+_')     # the lowering's temporaries (``VectorLowering.temp``)


class _ObjectUses(cst.CSTTransformer):
    """D154: a vector reaching Python code inside a compiled formula is a new list there (``.tolist()``).

    After the lowering, a typed vector (a vector local or temporary, or the result of a vector cell's entry
    ``self.<cell>(...)``) is used natively only as the argument of a provider operation, through ``.n`` or
    ``.data``, in ``is None`` / ``is not None``, as an assignment target, as the value assigned to a vector
    name, as the returned value, or as the element of a tuple of vectors (unpacked, indexed or returned).
    Every other use is Python code (the sequence protocol, a builtin, a comprehension, a pandas index): the
    vector becomes a list there, so the value and the Python semantics are those of live, which builds a list.
    A tuple of vectors in such a position becomes a tuple of lists."""
    METADATA_DEPENDENCIES = (__import__('libcst.metadata', fromlist=['ParentNodeProvider']).ParentNodeProvider,)

    def __init__(self, names, shapes, own=None):
        super().__init__()
        self.names, self.shapes = names, shapes     # {name: shape}, {vector cell: shape}
        self.own = own                              # the shape this formula returns, if it is a vector cell
        self.converted = []

    def _typed_target(self, target):
        if isinstance(target, cst.Name):    # a typed vector local, or a temporary of the lowering (consumed natively)
            return target.value in self.names or _TEMP.match(target.value) is not None
        if isinstance(target, cst.Tuple):
            return all(self._typed_target(e.value) for e in target.elements)
        return False

    def _shape(self, node):
        if isinstance(node, cst.Name):
            return self.names.get(node.value)
        if (isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute)
                and _is_name(node.func.value, 'self')):
            return self.shapes.get(node.func.attr.value)
        if isinstance(node, cst.Subscript):
            base = self._shape(node.value)
            if isinstance(base, tuple) and _index(node) is not None and _int_literal(_index(node)) is not None:
                k = _int_literal(_index(node))
                if -len(base[1]) <= k < len(base[1]):
                    return base[1][k]
        return None

    def _parent(self, node):
        return self.get_metadata(self.METADATA_DEPENDENCIES[0], node, None)

    def _native(self, node, shape):
        parent = self._parent(node)
        if isinstance(parent, cst.Arg):
            call = self._parent(parent)
            f = call.func if isinstance(call, cst.Call) else None
            return isinstance(f, cst.Attribute) and _is_name(f.value, ALIAS)
        if isinstance(parent, cst.Attribute) and parent.value is node:
            return parent.attr.value in ('n', 'data')
        if isinstance(parent, cst.ComparisonTarget) or isinstance(parent, cst.Comparison):
            comp = parent if isinstance(parent, cst.Comparison) else self._parent(parent)
            ops = [c.operator for c in comp.comparisons]
            return all(isinstance(o, (cst.Is, cst.IsNot)) for o in ops) and any(
                _is_name(c.comparator, 'None') for c in comp.comparisons)
        if isinstance(parent, cst.AssignTarget):
            return True
        if isinstance(parent, cst.Return):
            return self.own is not None and self.own == shape     # the formula returns this typed vector
        if isinstance(parent, cst.Assign) and parent.value is node:
            return all(self._typed_target(t.target) for t in parent.targets)
        if isinstance(parent, cst.AnnAssign):
            return self._typed_target(parent.target)
        if isinstance(parent, cst.Subscript) and parent.value is node and isinstance(shape, tuple):
            return True     # an element of a tuple of vectors: judged where the element is used
        if isinstance(parent, cst.Element):
            outer = self._parent(parent)
            above = self._parent(outer) if isinstance(outer, cst.Tuple) else None
            if isinstance(above, cst.Return):
                return isinstance(self.own, tuple)
            if isinstance(above, cst.Assign) and above.value is outer:
                return all(self._typed_target(t.target) for t in above.targets)
            return False
        return False

    def _leave(self, original, updated):
        shape = self._shape(original)
        if shape is None or self._native(original, shape):
            return updated
        self.converted.append(_code(original)[:80])
        if isinstance(shape, tuple):
            return _expr(f'tuple([_mx_vector.tolist() for _mx_vector in {_code(updated)}])')
        return cst.Call(cst.Attribute(updated, cst.Name('tolist')), ())

    def leave_Name(self, original, updated):
        return self._leave(original, updated)

    def leave_Call(self, original, updated):
        return self._leave(original, updated)

    def leave_Subscript(self, original, updated):
        return self._leave(original, updated)


def _object_uses(body, names, shapes, own=None):
    """``body`` (an IndentedBlock) with every Python use of a vector made a list (``_ObjectUses``)."""
    from libcst.metadata import MetadataWrapper
    wrapper = MetadataWrapper(cst.Module(body=body.body), unsafe_skip_copy=True)
    uses = _ObjectUses(names, shapes, own)
    module = wrapper.visit(uses)
    return body.with_changes(body=module.body), uses.converted


def lower(node, cls_info, cells, scope, counter, module, ref_kinds=None):
    """Lower the vector forms of one ``_f_`` FunctionDef of ``module``; returns the new node.

    Runs only in modules with vector cells (``module_uses_vectors``); elsewhere the node is
    returned as it is, so modules without vector cells translate as before."""
    from libcst.metadata import BuiltinAssignment
    if not module_uses_vectors(module) or not isinstance(node.body, cst.IndentedBlock):
        return node
    cells.vector_report = report = {'shape': _shape_json(cells.vector_shape), 'lowered': 0, 'fallbacks': []}
    builtins = {name for name in ('sum', 'len', 'zip', 'range', 'min', 'max', 'list')
                if scope[name] and all(isinstance(b, BuiltinAssignment) for b in scope[name])}
    loops = [name for name in range_loop_locals(node.body, builtins) if name not in cells.params]
    lowering = VectorLowering(cls_info, cells, builtins, counter)
    from .powers import KIND_FLOAT, KIND_INT
    lowering.strict_locals = not module_vector_cells(module)
    lowering.ref_scalars = {name: {KIND_FLOAT: FLOAT, KIND_INT: INT}.get(kind)
                            for name, kind in (ref_kinds or {}).items()}
    for name in cells.params:       # D87: the formula's C-typed parameters are scalars of the typer
        try:
            kind = {'double': FLOAT, 'long long': INT}.get(cells.get_argtype_expr(name, c_style=True))
        except Exception:       # a parameter without type information
            kind = None
        if kind:
            lowering.param_scalars[name] = kind
    lowering.analyse(node.body)
    if not module_vector_cells(module) and not any(k in ('V', 'I') for k in lowering.env.values()):
        # D94: a module without vector cells: only formulas with a typed local vector are lowered
        cells.vector_report = None
        return node
    derived = derived_index_locals(lowering.facts, set(loops), cells.params)
    lowering.range_locals = set(loops)
    lowering.expanding = set()
    lowering.vector_fors = set()
    lowering.leading_sums = set()
    lowering.sum_gens, lowering.sum_calls = _statement_sums(node.body, lowering)
    lowering.sum_slices, lowering.alloc_slices = _vector_slices(node.body, lowering)
    decls = [_stmt(f'{name}: {CY_MOD}.longlong\n') for name in loops + derived]
    touches = (any(lowering.env.values()) or cells.vector_shape or any(
        isinstance(n, cst.Attribute) and n.attr.value in lowering.shapes for n in _walk(node.body))
        or any(isinstance(n, cst.Power) for n in _walk(node.body)))
    report['range_loop_locals'] = loops
    if derived:
        report['derived_index_locals'] = derived
    if not touches:
        return _declare_at_top(node, decls)
    body = node.body.visit(lowering)
    cells.vector_loop_renames = lowering.loop_renames
    vector_locals = sorted(name for name, k in lowering.env.items() if k in ('V', 'I'))
    typed = {name: lowering.env[name] for name in vector_locals}
    for decl in lowering.decls:     # temporaries the lowering declared as vectors
        for small in decl.body:
            if (isinstance(small, cst.AnnAssign) and isinstance(small.target, cst.Name)
                    and _code(small.annotation.annotation) in TYPES.values()):
                typed[small.target.value] = next(k for k, t in TYPES.items() if t == _code(small.annotation.annotation))
    body, converted = _object_uses(body, typed, {n: s for n, s in lowering.shapes.items()
                                                 if s in ('V', 'I') or isinstance(s, tuple)},
                                   cells.vector_shape if cells.vector_shape in ("V", "I") or isinstance(cells.vector_shape, tuple) else None)
    if derived:
        body = body.visit(_CheckedFloatReads(typed, set(derived)))
    for name, items in lowering.facts.assigns.items():
        # a local bound only as the target of loops over vectors holds C elements
        if (name not in cells.params and name not in loops and items
                and all(item[0] == 'for' and lowering.iter_vectors(item[1]) for item in items)
                and len({lowering.elem(lowering.iter_vectors(item[1])[item[2]]) for item in items}) == 1):
            elem = lowering.elem(lowering.iter_vectors(items[0][1])[items[0][2]])
            decls.append(_stmt(f'{name}: {SCALARS[elem]}\n'))
    decls += [_stmt(f'{name}: {TYPES[lowering.env[name]]}\n') for name in vector_locals] + lowering.decls
    for code in converted:      # the report names every Python use of a vector (D154), recorded or not above
        if not any(code in item for item in lowering.fallbacks):
            lowering.fallbacks.append(f'vector used in Python code: a new list: {code}')
    report.update({'lowered': lowering.lowered, 'fallbacks': lowering.fallbacks, 'lists': converted,
                   'bounds_hoisted': lowering.bounds_hoisted, 'powers': lowering.powers,
                   'vector_locals': {name: lowering.env[name] for name in vector_locals}})
    return _declare_at_top(node.with_changes(body=body), decls)


_CHAIN = (cst.Add, cst.Subtract, cst.Multiply, cst.Divide)


def _leading_operand(value):
    """The operand evaluated first in a chain of ``+``/``-``/``*``/``/`` (the leftmost leaf), else None."""
    if not (isinstance(value, cst.BinaryOperation) and isinstance(value.operator, _CHAIN)):
        return None
    while isinstance(value, cst.BinaryOperation) and isinstance(value.operator, _CHAIN):
        value = value.left
    return value


def _replace_leading(value, new):
    """``value`` with its leftmost ``+``/``-``/``*``/``/`` leaf replaced by ``new``."""
    if isinstance(value, cst.BinaryOperation) and isinstance(value.operator, _CHAIN):
        return value.with_changes(left=_replace_leading(value.left, new))
    return new.with_changes(lpar=value.lpar, rpar=value.rpar) if hasattr(new, 'lpar') else new


def _statement_sums(body, lowering):
    """``sum(<generator over vectors>)`` at statement level whose element is provably a float;
    diag-kr: also ``min``/``max`` of such a generator whose element is a float or an integer
    (``lowering.extreme_calls``: id of the call -> its name). diag-us2 (D93): a ``sum`` may also be
    the leftmost operand of a ``+``/``-``/``*``/``/`` chain that is the value of an assignment or return, which
    Python evaluates first, so the sum computed just before the statement keeps the order of
    evaluation (``lowering.leading_sums``)."""
    gens, calls = set(), set()
    lowering.extreme_calls = {}
    for node in _walk(body):
        if isinstance(node, (cst.Assign, cst.Return, cst.AugAssign)) and getattr(node, 'value', None) is not None:
            value = node.value
            leading = _leading_operand(value) if isinstance(node, (cst.Assign, cst.Return)) else None
            if leading is not None:
                if not (isinstance(leading, cst.Call) and _is_name(leading.func, 'sum')):
                    continue
                value = leading
            if (isinstance(value, cst.Call) and isinstance(value.func, cst.Name)
                    and value.func.value in ('sum', 'min', 'max') and value.func.value in lowering.builtins
                    and _plain_args(value, 1, 1) and isinstance(value.args[0].value, cst.GeneratorExp)):
                gen = value.args[0].value
                if lowering.iter_vectors(gen.for_in.iter) and lowering.comp_ok(gen, ifs_ok=True):
                    lowering.comp_scalars.append(lowering.loop_scalars(gen.for_in))
                    elem = lowering.scalar(gen.elt)
                    lowering.comp_scalars.pop()
                    if value.func.value == 'sum' and elem == FLOAT:
                        gens.add(id(gen))
                        calls.add(id(value))
                        if value is not node.value:
                            lowering.leading_sums.add(id(node.value))
                    elif value.func.value != 'sum' and elem in (FLOAT, INT) and not isinstance(node, cst.AugAssign):
                        gens.add(id(gen))
                        lowering.extreme_calls[id(value)] = (value.func.value, elem)
    return gens, calls


def _vector_slices(body, lowering):
    """Slices of vectors summed directly (no copy), and slices stored into a typed vector."""
    sums, allocs = set(), set()
    shape = lowering.cells.vector_shape
    for node in _walk(body):
        if (isinstance(node, cst.Call) and _is_name(node.func, 'sum') and 'sum' in lowering.builtins
                and _plain_args(node, 1, 1) and isinstance(node.args[0].value, cst.Subscript)):
            arg = node.args[0].value
            if _slice(arg) is not None and lowering.elem(arg.value):
                sums.add(id(arg))
        values = []
        if isinstance(node, cst.Assign) and len(node.targets) == 1 and _is_name(node.targets[0].target):
            values.append((node.value, lowering.env.get(node.targets[0].target.value)))
        if isinstance(node, cst.Return) and node.value is not None:
            if shape in ('V', 'I'):
                values.append((node.value, shape))
            elif isinstance(shape, tuple) and isinstance(node.value, cst.Tuple) \
                    and len(node.value.elements) == len(shape[1]):
                values += [(e.value, k) for e, k in zip(node.value.elements, shape[1])]
        for value, kind in values:
            if (kind in ('V', 'I') and isinstance(value, cst.Subscript) and _slice(value) is not None
                    and lowering.elem(value.value) == kind):
                allocs.add(id(value))
    return sums, allocs
