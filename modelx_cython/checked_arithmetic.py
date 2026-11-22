"""Native divisibility of signed C integers (``x % k == 0``) and the signed-local proof it uses."""
import libcst as cst
from libcst.metadata import ExpressionContext

from .scopes import CstFacts, has_scope_effects
from .syntax import arguments, dotted_name, local_assignment, statements, walk

_BOUND = 1 << 25


def entry_names(cls):
    """The private entries :func:`inline_entries.add_inline_entries` emitted for the Cells of ``cls``
    (recorded on each Cell as ``inline_entry``)."""
    return {name for cell in cls.cells.values() if (name := getattr(cell, 'inline_entry', None))}


def virtual_call(expr, entries):
    """D139: the Cell call of a private-entry expression ``(entry(self, ...) if <exact receiver> else
    [cast(T, ]self.cell(...)[)])``, where ``entry`` is one of ``entries`` (see :func:`entry_names`);
    the two branches have the same value, so the arithmetic proofs read it as that call. Any other
    expression is returned unchanged."""
    if not (isinstance(expr, cst.IfExp) and isinstance(expr.body, cst.Call)
            and isinstance(expr.body.func, cst.Name) and expr.body.func.value in entries):
        return expr
    test = expr.test
    exact = (isinstance(test, cst.Name) and test.value == '_mx_exact_receiver'
             or isinstance(test, cst.Comparison) and len(test.comparisons) == 1
             and isinstance(test.comparisons[0].operator, cst.Is)
             and isinstance(test.left, cst.Call) and dotted_name(test.left.func) == 'type'
             and len(test.left.args) == 1 and isinstance(test.left.args[0].value, cst.Name)
             and test.left.args[0].value.value == 'self')
    if not exact:
        return expr
    virtual = expr.orelse
    if (isinstance(virtual, cst.Call) and dotted_name(virtual.func) == '_mx_cy.cast'
            and len(virtual.args) == 2):
        virtual = virtual.args[1].value
    return virtual if isinstance(virtual, cst.Call) and isinstance(virtual.func, cst.Attribute) else expr


def divisibility_operands(node):
    if not isinstance(node, cst.Comparison) or len(node.comparisons) != 1:
        return None
    comparison, = node.comparisons
    remainder = node.left
    if (not isinstance(comparison.operator, (cst.Equal, cst.NotEqual))
            or not isinstance(comparison.comparator, cst.Integer)
            or comparison.comparator.evaluated_value != 0
            or not isinstance(remainder, cst.BinaryOperation)
            or not isinstance(remainder.operator, cst.Modulo)):
        return None
    dividend = remainder.left
    operation = '_mx_divisible'
    if (isinstance(dividend, cst.BinaryOperation)
            and isinstance(dividend.right, cst.Integer) and dividend.right.evaluated_value == 1):
        if isinstance(dividend.operator, cst.Subtract):
            operation, dividend = '_mx_divisible_after_decrement', dividend.left
        elif isinstance(dividend.operator, cst.Add):
            operation, dividend = '_mx_divisible_after_increment', dividend.left
    return dividend, remainder.right, operation, isinstance(comparison.operator, cst.NotEqual)


def divisibility_enabled(cell):
    from . import automatic
    enabled = getattr(cell, '_spec', {}).get('native_divisibility', automatic.enabled(cell))
    if type(enabled) is not bool:
        raise ValueError('native_divisibility must be a boolean')
    return enabled


def signed_local_assignments(body, name, facts, native_value):
    """Find all signed definitions and the reads they definitely initialize."""
    reads = {node for node, context in facts.contexts.items()
             if isinstance(node, cst.Name) and node.value == name and context is ExpressionContext.LOAD}
    assignments = {}

    def flow(lines, assigned):
        for line in lines:
            evaluated = line.test if isinstance(line, cst.If) else line
            if not assigned and any(part in reads for part in walk(evaluated)):
                return None
            if isinstance(line, cst.If):
                taken = flow(statements(line.body), assigned)
                other = line.orelse
                if isinstance(other, cst.If):
                    skipped = flow((other,), assigned)
                elif other is not None:
                    skipped = flow(statements(other.body), assigned)
                else:
                    skipped = assigned
                if taken is None or skipped is None:
                    return None
                assigned = taken and skipped
            elif (target := local_assignment(line)) is not None and target.value == name:
                value = native_value(line.value)
                if value is None:
                    return None
                assignments[line] = value
                assigned = True
        return assigned

    if flow(statements(body), False) is None or len(assignments) != facts.stores[name]:
        return {}, reads
    return assignments, reads


class CheckedArithmetic:
    def __init__(self, cls, cell, builtin_keys=()):
        self.cls, self.cell = cls, cell
        self.builtin_keys = builtin_keys
        self.parameter_intervals = {}
        # diag-us2 (D94): intervals of locals bound once (range loop targets, single assignments)
        self.local_intervals = {}
        self.signed_atom = None

    def native_call(self, expr):
        expr = virtual_call(expr, entry_names(self.cls))
        args = arguments(expr)
        if (args is None or not isinstance(expr.func, cst.Attribute)
                or not isinstance(expr.func.value, cst.Name) or expr.func.value.value != 'self'
                or not all(isinstance(arg, cst.Name) and arg.value in self.parameters
                           or isinstance(arg, cst.Integer) and -_BOUND <= arg.evaluated_value <= _BOUND
                           for arg in args)):
            return False
        cell = self.cls.cells.get(expr.func.attr.value)
        return (cell is not None and not cell.is_special() and not cell.called_with_kwargs
                and not cell.body_has_closure and cell.get_rettype_expr(c_style=True) == 'long long')


    def _native_local(self, body, name, facts):
        if name in self.cell.params or name in facts.deleted or not facts.stores[name] or facts.stores['self']:
            return {}, set()
        return signed_local_assignments(
            body, name, facts, lambda value: value if self.native_call(value) else None)


    def divisibility(self, body, facts):
        local_reads = {}

        def signed(value):
            if not isinstance(value, cst.Name):
                return False
            if value.value in self.parameters:
                return True
            if value.value not in local_reads:
                assignments, reads = self._native_local(body, value.value, facts)
                local_reads[value.value] = reads if assignments else ()
            return value in local_reads[value.value]

        class Divisibility(cst.CSTTransformer):
            def visit_Lambda(inner, node):
                return False

            def visit_FunctionDef(inner, node):
                return False

            visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_Lambda

            def leave_Comparison(inner, before, after):
                operands = divisibility_operands(before)
                if operands is None:
                    return after
                dividend, divisor, operation, negate = operands
                if not (signed(dividend) and signed(divisor)):
                    return after
                prefix = 'not ' if negate else ''
                return cst.parse_expression(f'({prefix}_mx_sys.{operation}({dividend.value}, {divisor.value}))')

        return cst.Module(body).visit(Divisibility()).body

    def rewrite(self, function):
        """Lower zero-remainder comparisons of signed C integers to the native divisibility helpers.
        (integer guards: the range proofs and exact fallbacks that rewrote other arithmetic are
        gone; Cython's overflowcheck checks it in the checked mode.)"""
        self.cell.signed_range_counters = frozenset()
        if (not divisibility_enabled(self.cell) or self.cell.is_locked or self.cell.body_has_closure
                or self.cell.cls in self.cls.visitor.dynamic_ref_classes
                or not isinstance(function.body, cst.IndentedBlock)):
            return function
        body = function.body.body
        if has_scope_effects(body):
            return function
        facts = CstFacts(body)
        self.parameters = {name for name in self.cell.params if name not in facts.stores
                           and self.cell.get_argtype_expr(name, c_style=True, formula=True) == 'long long'}
        body = self.divisibility(body, facts)
        return function.with_changes(body=function.body.with_changes(body=body))
