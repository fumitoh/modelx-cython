"""Emit an opt-in scalar reduction through ordinary compiled leaf Cells."""

from dataclasses import dataclass

import libcst as cst
from libcst.metadata import ExpressionContext

from modelx_cython.consts import CY_MOD
from modelx_cython.holds import reset_lines
from .scopes import CstFacts
from .syntax import arguments, code, dotted_name, literal_value, statements, walk


@dataclass(frozen=True)
class Leaf:
    """One output of ``collect_outputs``: ``kind`` is 'scalar' (``self.cell()``, a double), 'dict'
    (``{k: self.cell(k) for k in self.<domain>}``, a double Cell of one C-integer parameter over a
    Reference) or 'python' (anything else, and the whole ``individual`` section), evaluated per item
    and merged as the export merges it."""
    mode: str
    label: str
    kind: str
    cell: str = ''
    domain: str = ''
    expression: str = ''
    reason: str = ''


@dataclass(frozen=True)
class MultiCollect:
    """collect_outputs with several outputs, ``{k: cell(k) for k in D}`` outputs or an ``individual``
    section: the native leaves are summed in C arrays, the others merged by the export's Accumulator."""
    parameter: str
    reuse: bool
    parallel_reuse: bool
    ordered: bool
    leaves: tuple
    extra: tuple = ()       # the collect's other Space parameters: (name, C type or 'object')

    @property
    def native(self):
        return tuple(leaf for leaf in self.leaves if leaf.kind != 'python')

    @property
    def python(self):
        return tuple(leaf for leaf in self.leaves if leaf.kind == 'python')


@dataclass(frozen=True)
class SerialCollect:
    label: str
    cell: str
    parameter: str
    reuse: bool
    parallel_reuse: bool
    ordered: bool = False


def _self_attribute(node):
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name):
        if node.value.value == 'self' and not node.attr.value.startswith('_'):
            return node.attr.value
    return None


def _method(node, name):
    return next((item for item in node.body.body
                 if isinstance(item, cst.FunctionDef) and item.asynchronous is None
                 and item.name.value == name), None)


def _return_expr(method):
    body = statements(method.body) if method else ()
    if len(body) == 1 and isinstance(body[0], cst.Return):
        return body[0].value
    return None


def _single_dict(expr):
    if (isinstance(expr, cst.Dict) and len(expr.elements) == 1
            and isinstance(expr.elements[0], cst.DictElement)):
        key = expr.elements[0].key
        try:
            value = literal_value(key)
        except (ValueError, TypeError):
            return None, None
        if type(value) is str:
            return value, expr.elements[0].value
    return None, None


def collect_contract(node):
    value = _return_expr(_method(node, '_mx_collect_contract_v2'))
    try:
        contract = literal_value(value)
    except (ValueError, TypeError):
        return None
    if (isinstance(contract, tuple) and len(contract) == 4
            and type(contract[0]) is str and isinstance(contract[1], tuple)
            and all(type(name) is str for name in contract[1])
            and type(contract[2]) is str and type(contract[3]) is bool):
        return contract
    return None


def analyze(module, evaluator_args=False):
    """Recognize the numeric sum shape. Policy isolation is a Space contract."""
    options = {name: module.spec.get_spec(cls.fqname)
               for name, cls in module.classes.items()}
    for name, spec in options.items():
        if 'serial_private_lookups' in spec:
            # audit-routes P6 (D126): every compiled root takes its lookup contexts over privately
            raise ValueError(f'{name.removeprefix("_c_")}: the spec key serial_private_lookups was removed; every compiled '
                             'root (collect and direct items) uses private lookups')
        if 'reuse_dense_windows' in spec:
            # D135: the D74 dense window of collect's reused root is removed; both routes keep an over-limit
            # Cell in the hash store
            raise ValueError(f'{name.removeprefix("_c_")}: the spec key reuse_dense_windows was removed (D135): '
                             'collect and the direct route hold a Cell above the dense limit in the same hash '
                             'store; if the Cell is read only on a regular grid, key it on the grid index '
                             '(conversion rule R9)')
    automatic = module.spec.automatic_optimizations
    if not any('serial_collect' in spec or 'parallel_collect_reuse' in spec
               for spec in options.values()) and not automatic:
        return {}
    result = {}
    for node in module.visitor.wrapper.module.body:
        if not isinstance(node, cst.ClassDef) or node.name.value not in module.classes:
            continue
        cls = module.classes[node.name.value]
        spec = options[node.name.value]
        option = spec.get('serial_collect', 'reduce' if automatic else None)
        parallel = spec.get('parallel_collect_reuse', automatic and option == 'reuse')
        if type(parallel) is not bool or (parallel and option != 'reuse'):
            raise ValueError('parallel_collect_reuse requires serial_collect reuse')
        # D18: parallel reuse merges chunk totals in domain order unless the spec asks for completion.
        order = spec.get('parallel_collect_order', 'domain' if parallel is True else 'completion')
        if order not in ('completion', 'domain') or (order == 'domain' and not parallel):
            raise ValueError('parallel_collect_order requires parallel reuse and completion or domain')
        if option is None:
            continue
        if option not in ('reduce', 'reuse'):
            raise ValueError('serial_collect must be reduce or reuse')
        shape = _match(node, cls, option == 'reuse', parallel, order == 'domain', evaluator_args)
        if shape is not None:
            result[node.name.value] = shape
    return result


def _isolated_leaf(node, cls):
    """A slotted leaf ItemSpace without child Spaces, model lock, dynamic References or collect hooks:
    its policy root is exactly ``_mx_new_root_C(base)`` (D116)."""
    return not (len(node.bases) != 1 or dotted_name(node.bases[0].value) != '_mx_sys.BaseSpace'
                or cls.spaces or cls.is_locked or node.name.value in cls.visitor.dynamic_ref_classes
                or any(_method(node, hook) is not None or hook in cls.refs
                       for hook in ('_mx_apply_formula', '_mx_collect_attach', '_mx_collect_release')))


def _match(node, cls, reuse, parallel_reuse, ordered=False, evaluator_args=False):
    if not _isolated_leaf(node, cls):
        return None
    contract = collect_contract(node)
    if contract is None:
        return None
    parameter = contract[0]
    if (not isinstance(parameter, str) or parameter not in cls.refs
            or cls.refs[parameter].get_type_expr(c_style=True) != 'long long'):
        return None
    extra = []
    for name in contract[1]:            # the collect's other Space parameters, typed from the trace
        if name not in cls.refs:
            return None
        kind = cls.refs[name].get_type_expr(c_style=True)
        extra.append((name, kind if kind in ('long long', 'double') else 'object'))
    output = _return_expr(_method(node, '_f_collect_outputs'))
    mode, fields = _single_dict(output)
    label, entry = _single_dict(fields)
    if not extra and mode == 'sum' and label and _scalar_cell(cls, entry):
        _report(cls, (Leaf('sum', label, 'scalar', cell=_self_attribute(entry.func)),))
        return SerialCollect(label, _self_attribute(entry.func), parameter, reuse, parallel_reuse, ordered)
    leaves = _leaves(cls, output, (parameter, *contract[1]))
    if leaves is None:
        return None
    _report(cls, leaves)
    # The export's ordered worker path folds Python sums one policy at a time.
    # Private chunk totals would change the grouping of arbitrary Python addition.
    if any(leaf.mode == 'sum' and leaf.kind == 'python' for leaf in leaves):
        parallel_reuse = False
    shape = MultiCollect(parameter, reuse, parallel_reuse, ordered, leaves, tuple(extra))
    if not shape.native:
        return None
    if extra and not evaluator_args:
        from .collect_adapter import PROTOCOL, _VERSION
        raise ValueError(
            f'{cls.name.removeprefix("_c_")}: its collect takes other Space parameters '
            f'({", ".join(name for name, _ in extra)}), which needs the export protocol {PROTOCOL} extension '
            f'(_mx_collect_evaluator_args in _mx_sys.py, beside {_VERSION} = {PROTOCOL}). '
            'Re-export the model with a modelx that writes it.')
    return shape


def _plain_cell(cls, name):
    cell = cls.cells.get(name)
    return (cell is not None and not cell.body_has_closure and not cell.called_with_kwargs
            and cell.get_rettype_expr(c_style=True) == 'double')


def _scalar_cell(cls, entry):
    """``self.cell()``: a zero-argument Cell with a double result."""
    if arguments(entry, count=0) is None:
        return False
    name = _self_attribute(entry.func)
    return name is not None and _plain_cell(cls, name) and not cls.cells[name].has_args()


def _dict_leaf(cls, entry, parameters):
    """``{k: self.cell(k) for k in self.<domain>}``: (cell, domain) or the reason it is not native."""
    if not isinstance(entry, cst.DictComp):
        return 'not a call of a zero-argument double Cell or {k: cell(k) for k in D}'
    part = entry.for_in
    if (part.inner_for_in is not None or part.ifs or part.asynchronous is not None
            or not isinstance(part.target, cst.Name) or not isinstance(entry.key, cst.Name)
            or entry.key.value != part.target.value):
        return 'the comprehension is not {k: cell(k) for k in D} (one for clause, no condition, the key is k)'
    variable = part.target.value
    value = entry.value
    args = arguments(value, count=1)
    if args is None or not isinstance(args[0], cst.Name) or args[0].value != variable:
        return 'the value is not cell(k) with k its only argument'
    name = _self_attribute(value.func)
    if name is None or not _plain_cell(cls, name):
        return 'the value is not a call of a double Cell'
    cell = cls.cells[name]
    if len(cell.params) != 1 or cell.get_argtype_expr(cell.params[0], c_style=True) != 'long long':
        return f'{name} does not take one C-integer argument'
    domain = _self_attribute(part.iter)
    if domain is None or domain in cls.cells:
        return ('the domain is not a Reference of the Space (a Cell may give a different domain to each '
                'item)')
    if domain not in cls.refs:
        return 'the domain is not a Reference of the Space'
    if domain in parameters:
        return 'a Space parameter domain must be evaluated on the policy item'
    return name, domain


class _Root(cst.CSTTransformer):
    def __init__(self, receiver_reads):
        self.receiver_reads = receiver_reads

    def leave_Name(self, original, updated):
        return updated.with_changes(value='root') if original in self.receiver_reads else updated


def _python_leaf(mode, label, entry, reason):
    facts = CstFacts(entry)
    # Inline only expressions whose lexical names are independent of evaluator
    # locals and which do not rebind or inspect the formula's frame.
    generated = {'root', 'coords', 'args', 'coordinate', 'first', 'need', 'error'}
    introspection = {'dir', 'eval', 'exec', 'globals', 'locals', 'super', 'vars'}
    if (facts.reads & (generated | introspection) or {'root', 'self'} & facts.stores.keys()
            or any(name.startswith('_mx_') for name in facts.reads)
            or any(isinstance(node, cst.NamedExpr) for node in walk(entry))):
        return None
    keywords = {node.keyword for node in walk(entry) if isinstance(node, cst.Arg)}
    receiver_reads = {node for node, context in facts.contexts.items()
                      if isinstance(node, cst.Name) and node.value == 'self'
                      and context is ExpressionContext.LOAD and node not in keywords}
    return Leaf(mode, label, 'python', expression=code(entry.visit(_Root(receiver_reads))), reason=reason)


def _leaves(cls, output, parameters):
    """The outputs of ``collect_outputs`` in source order, or None when it is not a literal
    ``{'individual': ..., 'sum': {name: ...}}``."""
    if not isinstance(output, cst.Dict) or not output.elements:
        return None
    leaves, modes = [], set()
    for element in output.elements:
        if not isinstance(element, cst.DictElement):
            return None
        try:
            mode = literal_value(element.key)
        except (ValueError, TypeError):
            return None
        if mode not in ('individual', 'sum') or mode in modes:
            return None
        modes.add(mode)
        if mode == 'individual':
            leaf = _python_leaf(mode, '', element.value, 'individual outputs are kept per item')
            if leaf is None:
                return None
            leaves.append(leaf)
            continue
        fields = element.value
        if not isinstance(fields, cst.Dict) or not fields.elements:
            return None
        labels = set()
        for item in fields.elements:
            if not isinstance(item, cst.DictElement):
                return None
            try:
                label = literal_value(item.key)
            except (ValueError, TypeError):
                return None
            if type(label) is not str or label in labels:
                return None
            labels.add(label)
            entry = item.value
            if _scalar_cell(cls, entry):
                leaves.append(Leaf(mode, label, 'scalar', cell=_self_attribute(entry.func)))
                continue
            found = _dict_leaf(cls, entry, parameters)
            if isinstance(found, tuple):
                leaves.append(Leaf(mode, label, 'dict', cell=found[0], domain=found[1]))
            else:
                leaf = _python_leaf(mode, label, entry, found)
                if leaf is None:
                    return None
                leaves.append(leaf)
    return tuple(leaves)


def _report(cls, leaves):
    """Build report: each output of collect_outputs, native or not and why (on the collect_outputs Cell)."""
    cell = cls.cells.get('collect_outputs')
    if cell is not None:
        cell.collect_leaves = [{'mode': leaf.mode, 'output': leaf.label, 'native': leaf.kind != 'python',
                                **({'reason': leaf.reason} if leaf.kind == 'python' else {})}
                               for leaf in leaves]


def _check_closure(cls, names):
    """Cells a check Cell reaches through ``self.<cell>`` (transitively), or None when a formula uses
    ``self`` other than as ``self.<attribute>`` (the pre-pass then resets every cache)."""
    defs = cls.visitor.formula_defs.get(cls.name, {})
    seen, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        node = defs.get(name)
        if node is None:
            return None
        parts = list(walk(node.body))
        bare = sum(1 for part in parts if isinstance(part, cst.Name) and part.value == 'self')
        owned = [part.attr.value for part in parts if isinstance(part, cst.Attribute)
                 and isinstance(part.value, cst.Name) and part.value.value == 'self']
        if bare != len(owned):
            return None
        todo.extend(attr for attr in owned if attr in cls.cells)
    return seen


def _reset_lines(cls, only=None):
    # R-4: what the item holds from shared inputs goes with the per-policy caches
    yield from reset_lines(cls)
    for name, cell in cls.cells.items():
        if only is not None and name not in only:
            continue
        if memo := getattr(cell, 'power_memo', None):
            yield f'root.{memo}.ready = False'
        if cell.is_special() or not cell.has_formula_def:
            continue
        if not cell.has_args():
            yield f'root._has_{name} = False'
            if cell.get_rettype_expr(c_style=True) not in ('bint', 'double', 'long long'):
                yield f'root._v_{name} = None'
        elif cell.has_typeinfo() and cell.is_arrayable():
            flags = f'root._has_{name}'
            yield f'_mx_collect_memset({CY_MOD}.address({flags}), 0, {CY_MOD}.sizeof({flags}))'
            if cell.has_dense_spill:
                yield f'root._spill_{name} = None'
        elif cell.uses_native_numeric_cache:
            yield f'root._v_{name}.reset()'
        elif getattr(cls, 'collect_shared_item', False):
            if cell.uses_vector_cache:
                from .vectors import cache_constructor
                yield f'root._v_{name} = {cache_constructor(cell)}'
            else:
                yield f'root._v_{name}.clear()'
        else:
            yield f'root._v_{name} = None'


@dataclass(frozen=True)
class BoundCheck:
    parameter: str
    cell: str
    limit: int          # largest value of the check Cell a policy may have
    bound: int          # size of the parameter on the Cell that limits the check
    margin: int         # traced stop of the parameter minus the traced maximum of the check Cell
    declared: str       # where the bound comes from


def bound_checks(cls):
    """param_bound_checks of a Space: {parameter: Cell}. The check Cell is a zero-argument cached
    Cell whose value is the policy's need for the parameter (a horizon, a maximum age). The sample
    fixes the offset between that value and the arguments the model then passes (``t`` up to
    ``proj_len()``, say): margin = traced stop of the parameter - traced maximum of the Cell. A
    policy is accepted when value + margin <= the parameter's bound, on every cached Cell with
    that parameter."""
    from .config import TransSpec
    spec = cls.module.spec.get_spec(cls.fqname)
    requested = spec.get(TransSpec.PARAM_BOUND_CHECKS, {})
    if not isinstance(requested, dict):
        raise ValueError(f'{cls.fqname}: param_bound_checks must be a dict')
    checks = []
    for parameter, name in requested.items():
        cell = cls.cells.get(name)
        if (cell is None or cell.is_special() or not cell.has_formula_def or cell.has_args()
                or not cell.has_typeinfo()
                or cell.get_rettype_expr(c_style=True) not in ('long long', 'double')):
            raise ValueError(f'{cls.fqname}: param_bound_checks[{parameter!r}] needs a zero-argument '
                             f'cached Cell with a numeric result, got {name!r}')
        check_max = getattr(cell._rt, 'max_ret_value', None)
        if check_max is None:
            raise ValueError(f'{cls.fqname}: the sample gave no numeric value of {name}')
        limit = None
        for signature, sizes in cls.cells_arg_sizes.items():
            if parameter not in signature:
                continue
            stop = sizes[signature.index(parameter)]
            traced = cls._cells_max_args.get(signature, {}).get(parameter)
            if traced is None:
                continue
            margin = traced[0] + 1 - check_max
            allowed = stop - margin
            if limit is None or allowed < limit[0]:
                where = cls.declared_axes.get(signature, {}).get(parameter)
                limit = allowed, stop, margin, (f'cells_param_size at {where}' if where
                                                else 'sampled, not declared')
        if limit is None:
            raise ValueError(f'{cls.fqname}: param_bound_checks[{parameter!r}]: no cached Cell has '
                             f'the integer parameter {parameter!r}')
        checks.append(BoundCheck(parameter, name, int(limit[0]), int(limit[1]), int(limit[2]), limit[3]))
    return checks


def _check_lines(cls, shape, checks):
    """Pre-pass before any projection: every policy's need against the bounds."""
    lines = ['        for coordinate in coords:']
    lines.extend(('            if not first:',))
    # Between policies only the caches the check Cells can reach are reset (all, if unknown).
    closure = _check_closure(cls, [check.cell for check in checks]) if shape.reuse else None
    lines.extend('                ' + line for line in (_restart_lines(cls, shape) if closure is None
                                                         else _reset_lines(cls, closure)))
    if closure is not None:
        lines.append('                pass')
    lines.append(f'            root.{shape.parameter} = coordinate')
    lines.append('            first = False')
    space = cls.name.removeprefix("_c_")
    for check in checks:
        message = (f'{space}: policy {{}} needs {check.parameter} for {check.cell}() = {{}} '
                   f'(sampled margin {check.margin}); the size of {check.parameter} is {check.bound} '
                   f'({check.declared}), which accepts {check.cell}() <= {check.limit}. '
                   f"Raise cells_param_size {{{{{check.parameter!r}: ...}}}} in the spec.")
        failed = (f'{space}: param_bound_checks: {check.cell}() raised for policy {{}} '
                  f'before any projection: {{!r}}')
        lines.extend((
            '            try:',
            f'                need = root.{check.cell}()',
            '            except Exception as error:',
            f'                raise RuntimeError({failed!r}.format(coordinate, error)) from error',
            f'            if need > {check.limit}:',
            f'                raise IndexError({message!r}.format(coordinate, need))',
        ))
    lines.append('        if not first:')
    if shape.reuse:     # what the check and the collected Cell reach (everything when unknown)
        both = _closure(cls, shape, [check.cell for check in checks])
        lines.extend('            ' + line for line in _reset_lines(cls, both))
    else:
        lines.extend('            ' + line for line in _restart_lines(cls, shape))
    lines.append('            pass')
    lines.append('        first = True')
    return lines


def _closure(cls, shape, extra=()):
    """What the collected Cells (and ``extra``) reach; None (everything) when an output is evaluated
    as Python code."""
    if isinstance(shape, MultiCollect):
        if shape.python:
            return None
        return _check_closure(cls, [*dict.fromkeys(leaf.cell for leaf in shape.native), *extra])
    return _check_closure(cls, [shape.cell, *extra])


def _restart_lines(cls, shape):
    if shape.reuse:
        # D107: collect computes the collected Cell only; between policies it resets what that Cell can
        # reach (static closure; everything when unknown). The reused root is private to this entry.
        return list(_reset_lines(cls, _closure(cls, shape)))
    from .item_roots import new_root, release_root
    return [f'{release_root(cls.name)}(root)', f'root = {new_root(cls.name)}(self)', *_bind_extra_lines(shape)]


def _bind_extra_lines(shape):
    if not isinstance(shape, MultiCollect):
        return []
    return [f'root.{name} = args[{position}]' for position, (name, _) in enumerate(shape.extra)]


def add_methods(cls, node, shape, *, workers):
    """Add a private evaluator to the existing generated collect entry."""
    if collect_contract(node) is None:
        return node
    can_reuse = f'type(self) is {cls.name} and not self._mx_roots and not self._mx_spaces'
    callback = 'self._mx_evaluate_serial_collect' if shape is not None else 'None'
    parallel = can_reuse if shape is not None and shape.parallel_reuse else 'False'
    # an individual section is returned in domain order whatever order the parallel chunks finish in
    ordered = bool(shape and (shape.ordered or isinstance(shape, MultiCollect)
                              and any(leaf.mode == 'individual' for leaf in shape.leaves)))
    takes_args = ', True' if isinstance(shape, MultiCollect) and shape.extra else ''
    options = cst.parse_statement(
        f'def _mx_collect_options_v2(self):\n    return {workers}, {callback}, {parallel}, {ordered}{takes_args}\n')
    shared = _shared_root_methods(cls, node)
    if shape is None:
        return node.with_changes(body=node.body.with_changes(body=(*node.body.body, *shared, options)))
    from .item_roots import new_root, release_root     # D116: shared with the direct route
    if isinstance(shape, MultiCollect):
        helper = cst.parse_module('\n'.join(_multi_lines(cls, shape)) + '\n').body[0]
        return node.with_changes(body=node.body.with_changes(body=(*node.body.body, helper, *shared, options)))
    lines = [
        'def _mx_evaluate_serial_collect(self, coords):',
        f'    root: {cls.name}',
        f'    coordinate: {CY_MOD}.longlong',
        f'    total: {CY_MOD}.double = 0.0',
        f'    first: {CY_MOD}.bint = True',
        f'    if type(self) is not {cls.name} or self._mx_roots or self._mx_spaces:',
        '        return None',
        '    if not _mx_sys._mx_collect_native_coords(coords):',
        '        return None',
        f'    root = {new_root(cls.name)}(self)',
        '    try:',
    ]
    checks = bound_checks(cls)
    cls.collect_bound_checks = checks
    if checks:
        lines.extend(_check_lines(cls, shape, checks))
    lines.extend((
        '        for coordinate in coords:',
        '            if not first:',
    ))
    lines.extend('                ' + line for line in _restart_lines(cls, shape))
    lines.extend((
        f'            root.{shape.parameter} = coordinate',
        '            if first:',
        f'                total = root.{shape.cell}()',
        '                first = False',
        '            else:',
        f'                total += root.{shape.cell}()',
        '    finally:',
        f'        {release_root(cls.name)}(root)',
        f'    return _mx_sys.CollectResult({{"sum": _mx_sys.CollectResult({{{shape.label!r}: total}})}})',
    ))
    helper = cst.parse_module('\n'.join(lines) + '\n').body[0]
    return node.with_changes(body=node.body.with_changes(body=(*node.body.body, helper, *shared, options)))


def _shared_root_methods(cls, node):
    """audit-routes P3: the export's generic collect loop (no evaluator, tracing, nested collect,
    non-native coordinates, workers without reuse) builds and releases each policy root through the
    same generated functions as the evaluator and the direct route (D116), when the class offers them:
    ``_mx_collect_new_root()`` returns ``_mx_new_root_C(self)`` under the evaluator's own condition, else
    None (the export then keeps its generic walk); ``_mx_collect_release_root(root)`` releases it."""
    if node.name.value not in cls.visitor.itemspace_classes or not _isolated_leaf(node, cls):
        return ()
    from .item_roots import new_root, release_item
    return tuple(cst.parse_module(
        'def _mx_collect_new_root(self):\n'
        f'    if type(self) is {cls.name} and not self._mx_roots and not self._mx_spaces:\n'
        f'        return {new_root(cls.name)}(self)\n'
        '    return None\n'
        '\n'
        'def _mx_collect_release_root(self, root):\n'
        f'    {release_item(cls.name)}(root)\n').body)


def _multi_lines(cls, shape):
    """The evaluator of a MultiCollect. Outputs are evaluated item by item in source order (the order
    ``collect_outputs`` evaluates them). A native output adds in C: the first item assigns and each
    later one adds, as the export's merge does, so a double sums bit for bit as there. The others are
    collected per item and merged by the export's Accumulator. The domain of a dict output is read once
    from the Reference (every item shares it); a domain that is not a tuple, list or range of ints
    within 64 bits, or no items, leaves the whole collect to the export."""
    from .item_roots import new_root, release_root
    c = CY_MOD
    leaves = shape.leaves
    lines = [
        f"def _mx_evaluate_serial_collect(self, coords{', args' if shape.extra else ''}):",
        f'    root: {cls.name}',
        f'    coordinate: {c}.longlong',
        f'    first: {c}.bint = True',
        f'    _mx_i: {c}.Py_ssize_t',
    ]
    for index, leaf in enumerate(leaves):
        if leaf.kind == 'scalar':
            lines.append(f'    _mx_t{index}: {c}.double = 0.0')
        elif leaf.kind == 'dict':
            lines += [f'    _mx_n{index}: {c}.Py_ssize_t = 0',
                      f'    _mx_k{index}: {c}.p_longlong = {c}.NULL',
                      f'    _mx_a{index}: {c}.p_double = {c}.NULL']
    lines += [
        f'    if type(self) is not {cls.name} or self._mx_roots or self._mx_spaces:',
        '        return None',
        '    if not coords or not _mx_sys._mx_collect_native_coords(coords):',
        '        return None',
    ]
    if shape.extra:     # values of the other parameters the traced type holds, else the export's path
        lines.append(f'    if len(args) != {len(shape.extra)}:')
        lines.append('        return None')
        for position, (name, kind) in enumerate(shape.extra):
            if kind == 'long long':
                lines += [f'    if type(args[{position}]) is not int or not '
                          f'(-9223372036854775808 <= args[{position}] <= 9223372036854775807):',
                          '        return None']
            elif kind == 'double':
                lines += [f'    if type(args[{position}]) is not float:', '        return None']
    for index, leaf in enumerate(leaves):
        if leaf.kind == 'dict':
            lines += [
                f'    _mx_d{index} = self.{leaf.domain}',
                f'    if not (type(_mx_d{index}) is range or type(_mx_d{index}) in (tuple, list)',
                f'            and all(type(_mx_x) is int for _mx_x in _mx_d{index})):',
                '        return None',
                f'    _mx_d{index} = tuple(_mx_d{index})',
                f'    if any(_mx_x < -9223372036854775808 or _mx_x > 9223372036854775807 for _mx_x in _mx_d{index}):',
                '        return None',
                f'    _mx_n{index} = len(_mx_d{index})',
            ]
    if shape.python:
        lines += ['    from . import _mx_collect_semantics as _mx_cs',
                  '    _mx_accumulated = _mx_cs.Accumulator(_mx_sys.CollectResult)']
    lines += [f'    root = {new_root(cls.name)}(self)', '    try:']
    lines.extend('        ' + line for line in _bind_extra_lines(shape))
    for index, leaf in enumerate(leaves):
        if leaf.kind == 'dict':
            lines += [
                f'        _mx_k{index} = {c}.cast({c}.p_longlong, _mx_collect_malloc(({c}.sizeof({c}.longlong)) * (_mx_n{index} + 1)))',
                f'        _mx_a{index} = {c}.cast({c}.p_double, _mx_collect_malloc(({c}.sizeof({c}.double)) * (_mx_n{index} + 1)))',
                f'        if _mx_k{index} == {c}.NULL or _mx_a{index} == {c}.NULL:',
                '            raise MemoryError()',
                f'        for _mx_i in range(_mx_n{index}):',
                f'            _mx_k{index}[_mx_i] = _mx_d{index}[_mx_i]',
            ]
    checks = bound_checks(cls)
    cls.collect_bound_checks = checks
    if checks:
        lines.extend(_check_lines(cls, shape, checks))
    lines += ['        for coordinate in coords:', '            if not first:']
    lines.extend('                ' + line for line in _restart_lines(cls, shape))
    lines.append(f'            root.{shape.parameter} = coordinate')
    if shape.python:
        lines.append('            _mx_sum = {}')
    for index, leaf in enumerate(leaves):
        if leaf.kind == 'scalar':
            lines += ['            if first:',
                      f'                _mx_t{index} = root.{leaf.cell}()',
                      '            else:',
                      f'                _mx_t{index} += root.{leaf.cell}()']
        elif leaf.kind == 'dict':
            lines += ['            if first:',
                      f'                for _mx_i in range(_mx_n{index}):',
                      f'                    _mx_a{index}[_mx_i] = root.{leaf.cell}(_mx_k{index}[_mx_i])',
                      '            else:',
                      f'                for _mx_i in range(_mx_n{index}):',
                      f'                    _mx_a{index}[_mx_i] += root.{leaf.cell}(_mx_k{index}[_mx_i])']
        elif leaf.mode == 'individual':
            lines.append(f'            _mx_individual = ({leaf.expression})')
        else:
            lines.append(f'            _mx_sum[{leaf.label!r}] = ({leaf.expression})')
    if shape.python:
        modes = list(dict.fromkeys(leaf.mode for leaf in shape.python))
        parts = ', '.join(f"'individual': _mx_individual" if mode == 'individual' else "'sum': _mx_sum"
                          for mode in modes)
        lines.append(f'            _mx_accumulated.add(coordinate, _mx_cs.prepare_output({{{parts}}}, '
                     '(_mx_sys.BaseMxObject,)))')
    lines += ['            first = False']
    # the result, in the order collect_outputs writes its sections and outputs
    if shape.python:
        lines.append('        _mx_python = _mx_accumulated.result()._fields')
    sections = []
    for mode in dict.fromkeys(leaf.mode for leaf in leaves):
        if mode == 'individual':
            sections.append("'individual': _mx_python['individual']")
            continue
        fields = []
        for index, leaf in enumerate(leaves):
            if leaf.mode != 'sum':
                continue
            if leaf.kind == 'scalar':
                fields.append(f'{leaf.label!r}: _mx_t{index}')
            elif leaf.kind == 'dict':
                fields.append(f'{leaf.label!r}: {{_mx_d{index}[_mx_i]: _mx_a{index}[_mx_i] '
                              f'for _mx_i in range(_mx_n{index})}}')
            else:
                fields.append(f"{leaf.label!r}: _mx_python['sum']._fields[{leaf.label!r}]")
        sections.append(f"'sum': _mx_sys.CollectResult({{{', '.join(fields)}}})")
    lines.append(f"        return _mx_sys.CollectResult({{{', '.join(sections)}}})")
    lines.append('    finally:')
    for index, leaf in enumerate(leaves):
        if leaf.kind == 'dict':
            lines += [f'        _mx_collect_free(_mx_k{index})', f'        _mx_collect_free(_mx_a{index})']
    lines.append(f'        {release_root(cls.name)}(root)')
    return lines
