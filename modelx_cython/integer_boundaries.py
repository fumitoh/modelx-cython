"""Check integer Cell results and ItemSpace keys at their native boundaries."""

import libcst as cst

from .checked_arithmetic import (
    CheckedArithmetic, divisibility_enabled, divisibility_operands, signed_local_assignments,
)
from .consts import CY_MOD, FORMULA_PREF, MX_SELF, MX_SYS_MOD
from .powers import KIND_FLOAT, OperandKind, kind_of_type_expr
from .scopes import CstFacts, Names, OPAQUE_SCOPES, has_scope_effects, lexical_nodes
from .syntax import dotted_name, local_assignment, statements, walk
from .usage import RESOLVED, _extract_self_chain


_INTEGER_ANNOTATION = cst.Attribute(cst.Name(CY_MOD), cst.Name('longlong'))
_SIGNED_OPERATIONS = {
    # integer guards: the operators that keep a helper (Python semantics, the checked power);
    # + - * are plain C operators, checked by Cython's overflowcheck in the checked mode
    cst.FloorDivide: '_mx_signed_floordiv',
    cst.Modulo: '_mx_signed_remainder',
    cst.LeftShift: '_mx_signed_left_shift',
    cst.RightShift: '_mx_signed_right_shift',
    cst.Power: '_mx_signed_pow',
}
_PLAIN_OPERATIONS = (cst.Add, cst.Subtract, cst.Multiply)
_INTEGER_OPERATIONS = (*_PLAIN_OPERATIONS, *_SIGNED_OPERATIONS)
_PYTHON_SEMANTICS = (cst.LeftShift, cst.RightShift, cst.FloorDivide, cst.Modulo)
_DIVISIONS = {cst.FloorDivide: '_mx_signed_floordiv', cst.Modulo: '_mx_signed_remainder',
              cst.FloorDivideAssign: '_mx_signed_floordiv', cst.ModuloAssign: '_mx_signed_remainder'}
_HELPER_OPERATIONS = (*_PYTHON_SEMANTICS, cst.Power)
_INTEGER_CONVERSION = f'{MX_SYS_MOD}._mx_integer_result'
# the floor division raises Cython's unnamed OverflowError (LLONG_MIN // -1): the handler names it
_FLOORDIV = f'{MX_SYS_MOD}._mx_signed_floordiv'
_SIGNED_HELPERS = frozenset((*_SIGNED_OPERATIONS.values(), '_mx_integer_result', '_mx_exact_int64'))


_CALL_GATE = """\
@{cy}.cclass
class {gate}:
    # D110 P0: a Python-bound Cell with integer arguments, checked as before, then the compiled method;
    # built without functools.wraps (its attributes are read from the method on demand)
    method: object
    params: tuple
    path: str

    def __call__(self, *args, **kwargs):
        values = list(args)
        for position, name in self.params:
            if position < len(values):
                values[position] = {runtime}._mx_exact_int64(values[position], self.path)
            if name in kwargs:
                kwargs[name] = {runtime}._mx_exact_int64(kwargs[name], self.path)
        return self.method(*values, **kwargs)

    @property
    def __wrapped__(self):
        return self.method

    def __getattribute__(self, name):
        # introspection reads the method's own attributes (``__doc__`` included), as functools.wraps copied them
        if name == '__doc__':
            return self.method.__doc__
        try:
            return {runtime}._mx_generic_getattr(self, name)
        except AttributeError:
            return getattr(self.method, name)

    def __repr__(self):
        return repr(self.method)


@{cy}.cfunc
def {name}(method: object, params: tuple, path: str) -> object:
    gate: {gate} = {gate}.__new__({gate})
    gate.method, gate.params, gate.path = method, params, path
    return gate
"""


def integer_parameters(cell):
    if cell.is_special() or cell.body_has_closure or not cell.has_typeinfo():
        return ()
    return tuple((index, name) for index, name in enumerate(cell.params)
                 if cell.get_argtype_expr(name, c_style=True) == 'long long')


def _boundary_call(function, path, *values):
    target = cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(function))
    return cst.Call(target, tuple(cst.Arg(value) for value in values) +
                    (cst.Arg(cst.SimpleString(repr(path))),))


def _checked_integer(value, path):
    return _boundary_call('_mx_integer_result', path, value)


def _overflow_handler(body, path, error, leading_lines=()):
    """``try: <body> except OverflowError: raise <named>``: the error names the Cell; an error a
    Cell already named (a call to another Cell, a checked conversion) passes unchanged."""
    handler = cst.parse_statement(
        f'try:\n    pass\nexcept OverflowError as {error}:\n'
        f'    raise {MX_SYS_MOD}._mx_named_overflow({error}, {path!r})\n')
    return handler.with_changes(body=cst.IndentedBlock(body=body), leading_lines=leading_lines)


_ARITHMETIC_NODES = (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo, cst.Power,
                     cst.LeftShift, cst.AddAssign, cst.SubtractAssign, cst.MultiplyAssign,
                     cst.FloorDivideAssign, cst.ModuloAssign, cst.PowerAssign, cst.LeftShiftAssign)


def _name_overflows(node, path):
    """Checked mode: one handler around the body of a formula function that does arithmetic, so
    an OverflowError from Cython's overflowcheck names the Cell. Declarations stay before it."""
    if not isinstance(node.body, cst.IndentedBlock) or not any(
            isinstance(part, _ARITHMETIC_NODES)
            or isinstance(part, cst.UnaryOperation) and isinstance(part.operator, cst.Minus)
            or isinstance(part, cst.Call) and dotted_name(part.func) == _FLOORDIV
            for part in walk(node.body)):
        return node
    lines = list(node.body.body)
    prefix = 1 if node.get_docstring() is not None else 0
    while prefix < len(lines) and isinstance(lines[prefix], cst.SimpleStatementLine) and all(
            isinstance(small, cst.AnnAssign) and small.value is None for small in lines[prefix].body):
        prefix += 1
    if prefix == len(lines):
        return node
    error = Names(node).reserve('_mx_integer_overflow')
    handler = _overflow_handler(tuple(lines[prefix:]), path, error)
    return node.with_changes(body=node.body.with_changes(body=(*lines[:prefix], handler)))


def _python_call_gate(cls, entries, names, call_gate):
    """Check Python-bound Cells while preserving their Cython vtable types."""
    routes = {cell.name: (entries[cell.fqname], cell.model_path)
              for cell in cls.cells.values() if cell.fqname in entries}
    if not routes:
        return (), ()
    field = names.reserve(f'_mx_integer_arguments_{cls.name}')
    declaration = cst.parse_statement(f'{field} = {routes!r}\n')
    method = cst.parse_statement(f"""\
def __getattribute__(self, name):
    method = {MX_SYS_MOD}._mx_generic_getattr(self, name)
    entry = {field}.get(name)
    if entry is not None and (
        type(self) is {cls.name}
        or getattr(type(self), name) is getattr({CY_MOD}.cast(object, {cls.name}), name)
    ):
        return {call_gate}(method, entry[0], entry[1])
    return method
""")
    return (declaration,), (method,)


class _GuardCalls(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider,)

    def __init__(self, receiver, integer_params, counters, owner, resolver, entries, cells, private_entries,
                 item_cells=None):
        self.item_cells = item_cells or {}     # D110 P1: {typed item local: {cell name: cell}}
        self.receiver = receiver
        self.integer_params = integer_params
        self.counters = counters
        self.loop_targets = []
        self.owner = owner
        self.resolver = resolver
        self.entries = entries
        self.cells = cells
        self.private_entries = private_entries

    def visit_For_body(self, node):
        self.loop_targets.append(node.target.value if isinstance(node.target, cst.Name) else None)

    def leave_For_body(self, node):
        self.loop_targets.pop()

    def _native_argument(self, value):
        if not isinstance(value, cst.Name):
            return False
        scope = self.get_metadata(cst.metadata.ScopeProvider, value, None)
        # Range analysis already proved the only store and that its reads
        # stay in the loop body. Check the receiver scope after entry lowering.
        if value.value in self.counters and value.value in self.loop_targets:
            receiver_scope = self.get_metadata(cst.metadata.ScopeProvider, self.receiver, None)
            if scope is receiver_scope:
                return True
        if value.value not in self.integer_params:
            return False
        bindings = tuple(scope[value.value]) if scope is not None else ()
        return len(bindings) == 1 and bindings[0].node is self.integer_params[value.value]

    def leave_Call(self, before, after):
        if isinstance(before.func, cst.Name) and (entry := self.private_entries.get(before.func.value)):
            path, params = entry
            return self._arguments(before, after, path, params, start=1)
        if (isinstance(before.func, cst.Attribute) and isinstance(before.func.value, cst.Name)
                and (owned := self.item_cells.get(before.func.value.value)) is not None):
            # D110 P1: a call on a typed item local checks its integer arguments as a call on self does
            cell = owned.get(before.func.attr.value)
            if (cell is None or any(arg.star for arg in before.args) or cell.called_with_kwargs
                    or not (params := self.entries.get(cell.fqname))):
                return after
            return self._arguments(before, after, cell.model_path, params)
        chain = _extract_self_chain(before.func)
        if not isinstance(chain, list) or any(arg.star for arg in before.args):
            return after
        base = before.func
        while isinstance(base, cst.Attribute):
            base = base.value
        scope = self.get_metadata(cst.metadata.ScopeProvider, base, None)
        assignments = tuple(scope[MX_SELF]) if scope is not None else ()
        if len(assignments) != 1 or assignments[0].node is not self.receiver:
            return after
        status, target = self.resolver.resolve_chain(self.owner, chain)
        if status != RESOLVED or not (params := self.entries.get(target)):
            return after
        cell = self.cells[target]
        if cell.called_with_kwargs:
            # A Python-dispatched method goes through the normal attribute gate.
            return after
        return self._arguments(before, after, cell.model_path, params)

    def _arguments(self, before, after, path, params, start=0):
        guarded = {index + start for index, _ in params}
        arguments = tuple(arg.with_changes(value=_checked_integer(arg.value, path))
                          if index in guarded and arg.keyword is None and not self._native_argument(old.value)
                          else arg for index, (old, arg) in enumerate(zip(before.args, after.args)))
        return after.with_changes(args=arguments)


def _guard_calls(node, cls, cell, resolver, entries, cells, private_entries, classes=None):
    params = (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)
    if not params or params[0].name.value != MX_SELF:
        return node
    integer_params = {param.name.value: param for param in params if param.annotation is not None
                      and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    counters = getattr(cell, 'signed_range_counters', ())
    wrapper = cst.MetadataWrapper(cst.Module(body=(node,)), unsafe_skip_copy=True)
    item_cells = {local: classes[owner].cells for local, owner in getattr(cell, 'typed_item_locals', {}).items()
                  if classes and owner in classes}
    return wrapper.visit(_GuardCalls(params[0], integer_params, counters, cls.fqname, resolver, entries, cells,
                                    private_entries, item_cells)).body[0]


def _object_parameters(params, checked):
    return tuple(param.with_changes(annotation=cst.Annotation(cst.Name('object')))
                 if param.name.value in checked else param for param in params)


def _guard_space_parameters(node, cell):
    checked = tuple(name for name in cell.params
                    if cell.get_argtype_expr(name, c_style=True) == 'long long')
    if not checked:
        return node
    assignments = tuple(cst.parse_statement(
        f'{name} = {MX_SYS_MOD}._mx_exact_int64({name}, {cell.model_path!r})\n')
        for name in checked)
    params = _object_parameters(node.params.params, checked)
    statements = node.body.body
    doc_end = 1 if node.get_docstring() is not None else 0
    return node.with_changes(params=node.params.with_changes(params=params),
                             body=node.body.with_changes(
                                 body=(*statements[:doc_end], *assignments, *statements[doc_end:])))


class _GuardIntegers(cst.CSTTransformer):
    def __init__(self, path, kinds, native_helpers, arithmetic, signed_fields,
                 signed_functions, signed_methods, signed_names, signed_return, error, checked=True):
        self.path = path
        # integer guards: False in the unchecked mode, where C integer arithmetic stays plain C
        self.checked = checked
        self.kinds = kinds
        self.native_helpers = native_helpers
        self.arithmetic = arithmetic
        self.signed_fields = signed_fields
        self.signed_functions = signed_functions
        self.signed_methods = signed_methods
        self.signed_names = signed_names
        self.signed_return = signed_return
        self.error = error
        self.arithmetic.signed_atom = self._signed_atom

    def on_visit(self, node):
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)


    def leave_Return(self, before, after):
        if not self.signed_return or after.value is None:
            return after
        return after.with_changes(value=self._checked(after.value))

    def leave_Assign(self, before, after):
        if not self.signed_return:
            return after
        value, _ = self._lower(after.value, self.path)
        return after.with_changes(value=value)

    def leave_AnnAssign(self, before, after):
        if not self.signed_return or after.value is None:
            return after
        value, _ = self._lower(after.value, self.path)
        return after.with_changes(value=value)

    def leave_Call(self, before, after):
        if (dotted_name(after.func) != _INTEGER_CONVERSION or len(after.args) != 2
                or not isinstance(after.args[1].value, cst.SimpleString)):
            return after
        operand, path = after.args
        value, signed = self._lower(operand.value, path.value.evaluated_value)
        if signed:
            return value.with_changes(lpar=(*after.lpar, *value.lpar), rpar=(*value.rpar, *after.rpar))
        return after.with_changes(args=(operand.with_changes(value=value), path))

    def _signed_expression(self, value, path):
        """Lower signed C expressions, leaving object operators on Cython's object path."""
        if isinstance(value, cst.BinaryOperation) and isinstance(value.operator, _INTEGER_OPERATIONS):
            left, left_signed = self._signed_expression(value.left, path)
            right, right_signed = self._signed_expression(value.right, path)
            # Shifts and the remainder keep their helpers in both modes, for Python's semantics:
            # a right shift rounds a negative value down (C leaves it to the implementation), a
            # left shift of a negative value or by 64 or more is exact in Python (Cython's checked
            # shift raises), LLONG_MIN % -1 is 0 and LLONG_MIN // -1 raises (C traps, and Cython
            # checks the division only where long long is as wide as long). An integer power
            # stored as a C integer takes the checked power in both modes: Cython's own C integer
            # power wraps, and with a variable exponent Cython makes it a double, which a C integer
            # result cannot take (at all, not only beyond 64 bits). Every other
            # operator is plain C: Cython's overflowcheck checks it in the checked mode.
            if (isinstance(value.operator, _HELPER_OPERATIONS) and left_signed and right_signed
                    and not (isinstance(value.operator, cst.FloorDivide) and _positive_literal(value.right))):
                operation = _SIGNED_OPERATIONS[type(value.operator)]
                checked = _boundary_call(operation, path, left, right)
                return checked.with_changes(lpar=value.lpar, rpar=value.rpar), True
            return value.with_changes(left=left, right=right), left_signed and right_signed
        if isinstance(value, cst.IfExp):
            body, body_signed = self._signed_expression(value.body, path)
            orelse, else_signed = self._signed_expression(value.orelse, path)
            return value.with_changes(body=body, orelse=orelse), body_signed and else_signed
        if isinstance(value, cst.UnaryOperation) and isinstance(value.operator, (cst.Plus, cst.Minus)):
            operand, signed = self._signed_expression(value.expression, path)
            if signed and isinstance(value.operator, cst.Minus) and self.checked:
                # Cython's overflowcheck does not check a unary minus (-LLONG_MIN): as 0 - x it does
                return cst.BinaryOperation(cst.Integer('0'), cst.Subtract(), operand,
                                           lpar=(cst.LeftParen(), *value.lpar),
                                           rpar=(*value.rpar, cst.RightParen())), True
            return value.with_changes(expression=operand), signed
        if isinstance(value, cst.Integer):
            return value, value.evaluated_value <= (1 << 63) - 1
        return value, self._signed_atom(value)

    def _signed_atom(self, value):
        if isinstance(value, cst.Name):
            return value.value in self.signed_names
        if (isinstance(value, cst.Call) and dotted_name(value.func) == f'{CY_MOD}.cast' and len(value.args) == 2
                and dotted_name(value.args[0].value) == f'{CY_MOD}.longlong'):
            return True         # the compiler's own C cast to long long (the virtual branch of a dispatch)
        if isinstance(value, cst.Call):
            target = value.func
            if isinstance(target, cst.Name):
                return target.value in self.signed_functions
            if isinstance(target, cst.Attribute) and isinstance(target.value, cst.Name):
                return (target.value.value == MX_SELF and target.attr.value in self.signed_methods
                        or target.value.value == MX_SYS_MOD and target.attr.value in _SIGNED_HELPERS)
        if isinstance(value, cst.Subscript) and any(not isinstance(s.slice, cst.Index) for s in value.slice):
            return False
        base = value.value if isinstance(value, cst.Subscript) else value
        if isinstance(base, cst.Attribute) and isinstance(base.value, cst.Name) and base.value.value == MX_SELF:
            dimensions = len(value.slice) if isinstance(value, cst.Subscript) else 0
            return self.signed_fields.get(base.attr.value) == dimensions
        return False

    def _checked(self, value, in_conditional=False):
        if isinstance(value, cst.IfExp):
            # Keep the conditional's evaluation order. Leave a statically
            # floating branch for Cython's existing narrow-return diagnostic.
            return value.with_changes(body=self._checked(value.body, in_conditional=True),
                                      orelse=self._checked(value.orelse, in_conditional=True))
        if ((isinstance(value, cst.Integer) and value.evaluated_value < (1 << 63))
                or (isinstance(value, cst.Call) and isinstance(value.func, cst.Name)
                    and value.func.value in self.native_helpers)
                or (in_conditional and self.kinds.kind(value) == KIND_FLOAT)):
            return value
        value, signed = self._lower(value, self.path)
        return value if signed else _checked_integer(value, self.path)

    def _lower(self, value, path):
        return self._signed_expression(value, path)


def _signed_operand_names(value):
    """Find the name uses that signed expression lowering treats as C operands."""
    if isinstance(value, cst.Name):
        yield value
    elif isinstance(value, cst.BinaryOperation) and isinstance(value.operator, _INTEGER_OPERATIONS):
        yield from _signed_operand_names(value.left)
        yield from _signed_operand_names(value.right)
    elif isinstance(value, cst.UnaryOperation) and isinstance(value.operator, (cst.Plus, cst.Minus)):
        yield from _signed_operand_names(value.expression)
    elif isinstance(value, cst.IfExp):
        yield from _signed_operand_names(value.body)
        yield from _signed_operand_names(value.orelse)


def _signed_local_assignments(body, name, operands, guard, facts):
    """Prove a native local across simple assignments and conditional joins."""
    def native_value(value):
        expression, signed = guard._signed_expression(value, guard.path)
        return expression if signed else None

    assignments, reads = signed_local_assignments(body, name, facts, native_value)
    if not assignments:
        return {}
    # A Python-context use could depend on the local's object identity or
    # Python operator semantics. Keep the whole local boxed in that case.
    safe_reads = set()
    for value in (*operands, *(assignment.value for assignment in assignments)):
        local_reads = reads.intersection(_signed_operand_names(value))
        if local_reads and guard._signed_expression(value, guard.path)[1]:
            safe_reads.update(local_reads)
    return assignments if reads <= safe_reads else {}


class _SignedLocalValues(cst.CSTTransformer):
    def __init__(self, assignments):
        self.assignments = assignments

    def leave_Assign(self, before, after):
        value = self.assignments.get(before)
        return after if value is None else after.with_changes(value=value)


def _signed_divisibility(body, facts, guard, parameters):
    """Admit locals whose only uses are signed zero-remainder comparisons."""
    if (has_scope_effects(body.body)
            or facts.reads & {'eval', 'exec', 'locals', 'vars', 'globals', '_getframe', 'currentframe'}):
        return body, ()
    probes = {node: operands for node in lexical_nodes(body.body)
              if (operands := divisibility_operands(node)) is not None
              and all(isinstance(value, cst.Name) for value in operands[:2])}
    uses = {value for operands in probes.values() for value in operands[:2]}
    native = guard.signed_names - {name for name in parameters if facts.stores[name]}
    declarations = []
    for name in sorted({value.value for value in uses} - set(parameters) - native):
        if not facts.stores[name]:
            continue
        assignments, reads = signed_local_assignments(
            body, name, facts,
            lambda value: value if guard._signed_expression(value, guard.path)[1] else None,
        )
        if not assignments or not reads <= uses:
            continue
        # Each comparison must be able to preserve the local's signed value.
        if not all(operand.value in native or operand.value == name
                   for operands in probes.values() if any(value in reads for value in operands[:2])
                   for operand in operands[:2]):
            continue
        native.add(name)
        declarations.append(cst.SimpleStatementLine((
            cst.AnnAssign(cst.Name(name), cst.Annotation(_INTEGER_ANNOTATION)),)))

    replacements = {}
    for probe, (dividend, divisor, operation, negate) in probes.items():
        if not {dividend.value, divisor.value} <= native:
            continue
        call = cst.Call(cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(operation)),
                        (cst.Arg(dividend), cst.Arg(divisor)))
        if negate:
            call = cst.UnaryOperation(cst.Not(), call)
        replacements[probe] = call.with_changes(lpar=probe.lpar, rpar=probe.rpar)

    class LowerDivisibility(cst.CSTTransformer):
        def leave_Comparison(self, before, after):
            return replacements.get(before, after)

    return body.visit(LowerDivisibility()) if replacements else body, tuple(declarations)


def _type_signed_locals(body, guard, facts, parameters):
    """Keep provably signed locals native and preserve existing prefix admission."""
    if (not isinstance(body, cst.IndentedBlock) or has_scope_effects(body.body)
            or facts.reads & {'eval', 'exec', 'locals', 'vars', 'globals', '_getframe', 'currentframe'}):
        return body, ()
    operands = []
    identity_names = set()
    for part in lexical_nodes(body.body):
        if guard.signed_return and isinstance(part, cst.Return) and part.value is not None:
            operands.append(part.value)
        if (isinstance(part, cst.Call) and dotted_name(part.func) == _INTEGER_CONVERSION
                and len(part.args) == 2):
            operands.append(part.args[0].value)
    for part in walk(body):
        if isinstance(part, cst.Comparison) and any(
                isinstance(item.operator, (cst.Is, cst.IsNot)) for item in part.comparisons):
            identity_names.update(item.value for item in walk(part) if isinstance(item, cst.Name))
    needed = {part.value for value in operands for part in walk(value) if isinstance(part, cst.Name)}
    needed.difference_update(identity_names)
    if not needed:
        return body, ()
    statements = list(body.body)
    parameters = set(parameters)
    seen = parameters.copy()
    for index, line in enumerate(statements):
        if not isinstance(line, cst.SimpleStatementLine) or len(line.body) != 1:
            break
        assignment = line.body[0]
        if ((target := local_assignment(assignment)) is not None
                and target.value in needed and target.value not in seen
                and facts.stores[target.value] == 1):
            value, signed = guard._signed_expression(assignment.value, guard.path)
            if signed:
                statements[index] = line.with_changes(body=(
                    cst.AnnAssign(target, cst.Annotation(_INTEGER_ANNOTATION), value),))
                guard.signed_names.add(target.value)
        seen.update(part.value for part in walk(line) if isinstance(part, cst.Name))
    declarations, assignments = [], {}
    for name in sorted(needed - parameters - guard.signed_names):
        if not facts.stores[name]:
            continue
        guard.signed_names.add(name)
        values = _signed_local_assignments(body, name, operands, guard, facts)
        if not values:
            guard.signed_names.remove(name)
            guard.python_locals.add(name)       # D72: examined and left to Python
            continue
        declarations.append(cst.SimpleStatementLine((cst.AnnAssign(
            cst.Name(name), cst.Annotation(_INTEGER_ANNOTATION)),)))
        assignments.update(values)
    body = body.with_changes(body=statements)
    if assignments:
        body = body.visit(_SignedLocalValues(assignments))
    return body, tuple(declarations)


def _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds, signatures, native_helpers=(),
                  argument_axes=(), checked=True):
    signed_return = node.returns is not None and node.returns.annotation.deep_equals(_INTEGER_ANNOTATION)
    divisibility = (node.returns is not None and dotted_name(node.returns.annotation) == f'{CY_MOD}.bint'
                    and not cell.is_locked and not cell.body_has_closure and divisibility_enabled(cell)
                    and any(divisibility_operands(part) is not None for part in lexical_nodes(node.body.body)))
    params = {name: kind_of_type_expr(cell.get_argtype_expr(name, c_style=True, formula=True))
              for name in cell.params}
    kinds = OperandKind(resolver, cells_kinds, ref_kinds, cls.fqname, params)
    arithmetic = CheckedArithmetic(cls, cell)
    arithmetic.parameters = {
        param.name.value for param in node.params.params
        if param.annotation is not None and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)
    }
    facts = CstFacts(node.body.body)
    bindings = facts.bindings(lexical_nodes(node.body.body))
    if not signed_return and not divisibility and not any(
            isinstance(part, cst.Call) and dotted_name(part.func) == _INTEGER_CONVERSION
            for part in lexical_nodes(node.body.body)):
        return node
    fields, functions, methods = signatures
    signed_names = arithmetic.parameters | {
        statement.target.value for statement in lexical_nodes(node.body.body)
        if isinstance(statement, cst.AnnAssign) and isinstance(statement.target, cst.Name)
        and statement.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)
    }
    receiver_fields = fields.get(cls.name, {}) if MX_SELF not in bindings else {}
    receiver_methods = methods.get(cls.name, set()) if MX_SELF not in bindings else set()
    error = Names(node).reserve('_mx_integer_overflow')
    guard = _GuardIntegers(cell.model_path, kinds, native_helpers, arithmetic, receiver_fields,
                           functions.difference(bindings), receiver_methods, signed_names, signed_return,
                           error, checked)
    parameters = tuple(parameter.name.value for parameter in
                       (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params))
    guard.python_locals = set()
    if divisibility:
        body, declarations = _signed_divisibility(node.body, facts, guard, parameters)
    else:
        body, declarations = _type_signed_locals(node.body, guard, facts, parameters)
    cell.python_int_locals = set(getattr(cell, 'python_int_locals', ())) | guard.python_locals
    body = body.visit(guard)
    if declarations:
        doc_end = 1 if node.get_docstring() is not None else 0
        body = body.with_changes(body=(*body.body[:doc_end], *declarations, *body.body[doc_end:]))
    return node.with_changes(body=body)


def _positive_literal(value):
    """A divisor that cannot be -1: the division keeps Cython's C operator (no trap)."""
    return isinstance(value, cst.Integer) and value.evaluated_value > 0


class _GuardDivisions(cst.CSTTransformer):
    """Every ``//`` and ``%`` of two C integers takes the checked helper, in both modes and
    wherever it is (a float formula's ``12 // m`` as well as a C integer result): C traps on
    LLONG_MIN // -1 and LLONG_MIN % -1 (see _mx_signed_floordiv). A divisor that is a positive
    literal cannot be -1 and stays a C operator."""

    def __init__(self, guard):
        self.guard = guard

    def on_visit(self, node):
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)

    def _signed(self, value):
        return self.guard._signed_expression(value, self.guard.path)[1]

    def leave_BinaryOperation(self, before, after):
        operation = _DIVISIONS.get(type(after.operator))
        if (operation is None or _positive_literal(after.right)
                or not (self._signed(before.left) and self._signed(before.right))):
            return after
        return _boundary_call(operation, self.guard.path, after.left, after.right).with_changes(
            lpar=after.lpar, rpar=after.rpar)

    def leave_AugAssign(self, before, after):
        operation = _DIVISIONS.get(type(after.operator))
        if (operation is None or not isinstance(after.target, cst.Name)
                or _positive_literal(after.value)
                or not (self._signed(before.target) and self._signed(before.value))):
            return after
        return cst.Assign((cst.AssignTarget(after.target),),
                          _boundary_call(operation, self.guard.path, after.target, after.value))


def _guard_divisions(node, cell, cls, signatures, native_helpers=()):
    """Route the C integer ``//`` and ``%`` of one formula function through the helpers. The
    operands are recognised as the result lowering recognises them: parameters, declared and
    assigned locals, fields, casts and calls of C integer type."""
    if not any(isinstance(part, (cst.BinaryOperation, cst.AugAssign)) and type(part.operator) in _DIVISIONS
               for part in lexical_nodes(node.body.body)):
        return node
    fields, functions, methods = signatures
    facts = CstFacts(node.body.body)
    bindings = facts.bindings(lexical_nodes(node.body.body))
    parameters = {param.name.value for param in (*node.params.posonly_params, *node.params.params,
                                                 *node.params.kwonly_params)
                  if param.annotation is not None and param.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    declared = {statement.target.value for statement in lexical_nodes(node.body.body)
                if isinstance(statement, cst.AnnAssign) and isinstance(statement.target, cst.Name)
                and statement.annotation.annotation.deep_equals(_INTEGER_ANNOTATION)}
    arithmetic = CheckedArithmetic(cls, cell)
    arithmetic.parameters = parameters
    guard = _GuardIntegers(cell.model_path, None, native_helpers, arithmetic,
                           fields.get(cls.name, {}) if MX_SELF not in bindings else {},
                           functions.difference(bindings),
                           methods.get(cls.name, set()) if MX_SELF not in bindings else set(),
                           parameters | declared, False, None, False)
    # locals Cython infers as C integers: every store an assignment (plain or augmented) of a
    # C integer expression; any other store (a loop target, a float value) leaves the local out
    values = {}
    for part in lexical_nodes(node.body.body):
        if isinstance(part, cst.AugAssign) and isinstance(part.target, cst.Name):
            values.setdefault(part.target.value, []).append(part.value)
        elif (target := local_assignment(part)) is not None:
            values.setdefault(target.value, []).append(part.value)
    candidates = {name for name, count in facts.stores.items()
                  if count and name not in guard.signed_names and len(values.get(name, ())) == count}
    changed = True
    while changed:              # greatest fixpoint: assume all, drop the ones an assignment refutes
        changed = False
        guard.signed_names |= candidates
        for name in sorted(candidates):
            if not all(guard._signed_expression(value, guard.path)[1] for value in values[name]):
                candidates.discard(name)
                guard.signed_names.discard(name)
                changed = True
    return node.with_changes(body=node.body.visit(_GuardDivisions(guard)))


def _signed_signatures(state):
    def signed_result(node):
        return (node.returns is not None and node.returns.annotation.deep_equals(_INTEGER_ANNOTATION)
                and any(dotted_name(item.decorator) in {f'{CY_MOD}.cfunc', f'{CY_MOD}.ccall'}
                        for item in node.decorators))

    functions = {name for name, node in state.functions.items() if signed_result(node)}
    methods = {}
    for (owner, name), node in state.methods.items():
        if signed_result(node):
            methods.setdefault(owner, set()).add(name)
    fields = {}
    for owner in state.syntax.body:
        if not isinstance(owner, cst.ClassDef) or not isinstance(owner.body, cst.IndentedBlock):
            continue
        declared = fields[owner.name.value] = {}
        for line in owner.body.body:
            if not isinstance(line, cst.SimpleStatementLine):
                continue
            for declaration in line.body:
                if not isinstance(declaration, cst.AnnAssign) or not isinstance(declaration.target, cst.Name):
                    continue
                annotation = declaration.annotation.annotation
                dimensions = 0
                if isinstance(annotation, cst.Subscript) and all(isinstance(s.slice, cst.Index)
                                                                  for s in annotation.slice):
                    dimensions = len(annotation.slice)
                    annotation = annotation.value
                if annotation.deep_equals(_INTEGER_ANNOTATION):
                    declared[declaration.target.value] = dimensions
    return fields, functions, methods


def _python_locals(node, cell):
    """D72/D76: declare ``object`` the locals compared by identity.

    With ``infer_types`` (W1.5 0005) Cython infers a C type for every untyped local; a local
    compared with ``is``/``is not`` would lose its identity (``x is saved[0]`` on a re-boxed
    int gives a wrong answer), so it is declared ``object``. Integer locals and ``range``
    targets otherwise stay C integers (their arithmetic: plain C, checked by overflowcheck in the checked mode).
    Locals with an annotation of their own are left alone. The admission report lists each
    declared local (``object_locals``).
    """
    body = node.body
    # the overflow handler the guard itself wraps around the body is an except clause; any
    # other binding construct leaves the formula as it is
    effects = (cst.Global, cst.Nonlocal, cst.Import, cst.ImportFrom, cst.Del, cst.NamedExpr,
               cst.FunctionDef, cst.ClassDef, cst.Lambda)
    if not isinstance(body, cst.IndentedBlock) or any(
            isinstance(part, effects) or isinstance(part, getattr(cst, 'Match', ()))
            for part in lexical_nodes(body.body)):
        return node
    facts = CstFacts(body.body)
    parameters = {parameter.name.value for parameter in
                  (*node.params.posonly_params, *node.params.params, *node.params.kwonly_params)}
    annotated = {statement.target.value for statement in lexical_nodes(body.body)
                 if isinstance(statement, cst.AnnAssign) and isinstance(statement.target, cst.Name)}
    # D76: integer locals and range targets stay C integers with guarded arithmetic
    # (integer guards); only identity keeps the Python object
    names = set()
    for part in lexical_nodes(body.body):
        if isinstance(part, cst.Comparison) and any(
                isinstance(item.operator, (cst.Is, cst.IsNot)) for item in part.comparisons):
            names.update(item.value for item in walk(part) if isinstance(item, cst.Name))
    names = sorted(name for name in names
                   if facts.stores[name] and name not in parameters and name not in annotated)
    if not names:
        return node
    declarations = tuple(cst.SimpleStatementLine((cst.AnnAssign(
        cst.Name(name), cst.Annotation(cst.Name('object'))),)) for name in names)
    doc_end = 1 if node.get_docstring() is not None else 0
    cell.python_object_locals = tuple(names)
    return node.with_changes(body=body.with_changes(body=(*body.body[:doc_end], *declarations,
                                                          *body.body[doc_end:])))


def guard_integer_boundaries(state, module_info, resolver, cells_kinds, ref_kinds, cells, infer_types=False):
    """Check formula results, arguments and ItemSpace keys after entry lowering."""
    # integer guards: arguments, keys and results are checked in both modes; arithmetic only in
    # the checked mode
    checked = module_info.spec.policy.integer_checks != 'unchecked'
    entries = {name: params for name, cell in cells.items() if (params := integer_parameters(cell))}
    private_entries = {
        name: (cell.model_path, params)
        for cls in module_info.classes.values() for cell in cls.cells.values()
        if (name := getattr(cell, 'inline_entry', None)) and (params := entries.get(cell.fqname))
    }
    private_entries.update({name: private_entries[checked] for name, checked in state.inbounds_entries.items()
                            if checked in private_entries})
    classes = []
    declarations = []
    call_gate = state.names.reserve('_mx_checked_cell_callable')
    gate_class = state.names.reserve('_mx_checked_cell_call')
    changed = False
    signatures = _signed_signatures(state)
    native_helpers = {}
    for node in state.syntax.body:
        if (isinstance(node, cst.FunctionDef) and node.returns is not None
                and node.returns.annotation.deep_equals(_INTEGER_ANNOTATION)
                and (origin := state.function_origins.get(node.name.value)) is not None):
            native_helpers.setdefault(origin, set()).add(node.name.value)
    for node in state.syntax.body:
        cls = module_info.classes.get(node.name.value) if isinstance(node, cst.ClassDef) else None
        if cls is None:
            origin = state.function_origins.get(node.name.value) if isinstance(node, cst.FunctionDef) else None
            if origin:
                owner, name = origin
                cls = module_info.classes[owner]
                # D139: the private entry of an uncached Cell has the Cell's own name as its origin
                cell = cls.cells[name[len(FORMULA_PREF):] if name.startswith(FORMULA_PREF) else name]
                node = _guard_calls(node, cls, cell, resolver, entries, cells, private_entries, module_info.classes)
                # Physical cache misses and compact spill entries both check
                # the logical domain before reaching the private formula.
                axes = (cell.dense_logical_axes if cell.has_formula_def and cell.has_args() and cell.has_typeinfo()
                        and cell.is_arrayable() else ())
                node = _guard_result(node, cell, cls, resolver, cells_kinds, ref_kinds, signatures,
                                     argument_axes=axes, checked=checked)
                if checked:
                    node = _name_overflows(node, cell.model_path)
                node = _guard_divisions(node, cell, cls, signatures)
                if infer_types:
                    node = _python_locals(node, cell)
                changed = True
            classes.append(node)
            continue
        methods = []
        for method in node.body.body:
            if not isinstance(method, cst.FunctionDef):
                methods.append(method)
                continue
            name = method.name.value
            if name.startswith(FORMULA_PREF) and (formula := cls.cells.get(name[len(FORMULA_PREF):])):
                method = _guard_calls(method, cls, formula, resolver, entries, cells, private_entries, module_info.classes)
                method = _guard_result(method, formula, cls, resolver, cells_kinds, ref_kinds, signatures,
                                       native_helpers.get((cls.name, name), ()), checked=checked)
                if checked:
                    method = _name_overflows(method, formula.model_path)
                method = _guard_divisions(method, formula, cls, signatures,
                                          native_helpers.get((cls.name, name), ()))
                if infer_types:
                    method = _python_locals(method, formula)
                methods.append(method)
                changed = True
                continue
            cell = cls.cells.get(name)
            if name == '__call__' and cell is not None and cell.has_typeinfo():
                methods.append(_guard_space_parameters(method, cell))
                changed = True
                continue
            if cell is not None and not cell.has_formula_def:
                method = _guard_calls(method, cls, cell, resolver, entries, cells, private_entries, module_info.classes)
                method = _guard_result(method, cell, cls, resolver, cells_kinds, ref_kinds, signatures,
                                       checked=checked)
                if checked:
                    method = _name_overflows(method, cell.model_path)
                method = _guard_divisions(method, cell, cls, signatures)
                if infer_types:
                    method = _python_locals(method, cell)
                changed = True
            methods.append(method)
        added, gates = _python_call_gate(cls, entries, state.names, call_gate)
        declarations.extend(added)
        changed |= bool(gates)
        classes.append(node.with_changes(body=node.body.with_changes(body=(*methods, *gates))))
    if declarations:
        classes.extend(declarations)
        classes.extend(cst.parse_module(_CALL_GATE.format(
            cy=CY_MOD, name=call_gate, gate=gate_class, runtime=MX_SYS_MOD)).body)
    return state.replace(state.syntax.with_changes(body=classes)) if changed else state
