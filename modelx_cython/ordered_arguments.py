"""Preserve object power semantics outside the proved native domain.

Numeric Cells keep c8's power and extrema mechanics unchanged. Object-result
Cells and child scopes use ordinary object operators when Cython's native
promotion would change their scalar-class contract.
"""
import libcst as cst
from libcst.metadata import ScopeProvider, ParentNodeProvider, PositionProvider
from libcst.metadata.scope_provider import BuiltinAssignment


class _OrderedArguments(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider, PositionProvider)

    def __init__(self, native_result=False, operand_kind=None, providers=None):
        super().__init__()
        self.opaque_depth = 0
        self.native_result = native_result
        self.operand_kind = operand_kind
        self.providers = providers       # W1.4: typed min/max need the math provider
        self.lowered = 0
        self.untyped = []
        self.objects = []                # W1.8: compiled comparisons on Python objects
        self.first_line = None
        self.kinds = None                # W1.8: ExpressionKind of the formula (locals, builtins)
        self.formula_scope = None

    def on_visit(self, node):
        from .scopes import OPAQUE_SCOPES
        if isinstance(node, OPAQUE_SCOPES):
            self.opaque_depth += 1
        return super().on_visit(node)

    def on_leave(self, before, after):
        result = super().on_leave(before, after)
        from .scopes import OPAQUE_SCOPES
        if isinstance(before, OPAQUE_SCOPES):
            self.opaque_depth -= 1
        return result

    def leave_BinaryOperation(self, before, after):
        if not isinstance(before.operator, cst.Power):
            return after
        if self.native_result and self.opaque_depth == 1:
            # Proven floating powers were lowered earlier. An integral exponent
            # retains c8's arithmetic; other unproved powers need the existing
            # object operator so Cython cannot infer an unordered complex value.
            from .powers import KIND_INT
            if self.operand_kind is None or self.operand_kind.kind(before.right) == KIND_INT:
                return after
            # A native return already supplies Cython's scalar conversion.
            # Keep that existing arithmetic path; intermediate values and
            # comparisons lack this conversion boundary and must stay objects.
            parent = self.get_metadata(ParentNodeProvider, before)
            while isinstance(parent, (cst.BinaryOperation, cst.UnaryOperation)):
                parent = self.get_metadata(ParentNodeProvider, parent)
            if isinstance(parent, cst.Return):
                return after
        # The qualified real-power pass has already handled its proved C
        # domain. Preserve Python's remaining integer/mixed/complex power
        # result and error semantics through Cython's ordinary object operator.
        scope = self.get_metadata(ScopeProvider, before)
        bindings = tuple(scope['object'])
        if not bindings or not all(isinstance(binding, BuiltinAssignment) for binding in bindings):
            return after
        if tuple(scope['_mx_cy']):
            return after
        def boxed(value):
            return cst.Call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                            (cst.Arg(cst.Name('object')), cst.Arg(value)))
        return after.with_changes(left=boxed(after.left), right=boxed(after.right))

    def integer_identity(self, before, after):
        """``int(x)`` of a proven C integer is ``x`` (diag-us).

        Builtin ``int`` of an int returns that int, but Cython types the call as a Python
        object, so ``n = int((last - start) // step)`` made ``n`` an object and with it every
        local derived from it (``for k in range(1, n + 1)``: boxed ``k``, ``k * step``, ...),
        although the local-kind analysis proves ``n`` an integer (D59, ded875f). Only the
        builtin, one positional argument, at the formula's own level (not in a comprehension,
        lambda or nested def, where names may be bound otherwise)."""
        from .powers import KIND_INT
        if (self.operand_kind is None or self.opaque_depth != 1 or not isinstance(before.func, cst.Name)
                or before.func.value != 'int' or len(before.args) != 1 or before.args[0].keyword is not None
                or before.args[0].star):
            return None
        scope = self.get_metadata(ScopeProvider, before.func)
        bindings = tuple(scope['int'])
        if not bindings or not all(isinstance(binding, BuiltinAssignment) for binding in bindings):
            return None
        if self.operand_kind.kind(before.args[0].value) != KIND_INT:
            return None
        value = after.args[0].value
        return value.with_changes(lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),)) if not value.lpar else value

    def operand_kinds(self, before):
        """Kinds of the positional operands of the original call (W1.8 classifier)."""
        classify = self.kinds.kind if self.kinds is not None else self.operand_kind.kind
        return [classify(argument.value) for argument in before.args if argument.keyword is None]

    def typed_extrema(self, before, after, positional):
        """W1.4: min/max whose operands are all double or all C integers, as C comparisons.

        The result has the operands' class, so no object is made; arguments are
        evaluated left to right. Mixed or unknown operands keep the boxed call.
        W1.8: the operands are classified on the original call with the formula's
        locals, cached cell calls and numeric built-ins (``expression_kinds``).
        """
        from .powers import KIND_FLOAT, KIND_INT
        if self.operand_kind is None or self.providers is None:
            return None
        order = self.operand_kinds(before)
        kinds = set(order)
        from .helper_members import MATH_PROVIDER
        from .helpers import module_name
        name = lambda suffix: cst.Attribute(cst.Name(module_name(MATH_PROVIDER)),
                                            cst.Name(before.func.value + suffix))
        if kinds == {KIND_FLOAT}:
            function = name('_f64')
        elif kinds == {KIND_INT}:
            function = name('_i64')
        elif kinds == {KIND_FLOAT, KIND_INT} and len(order) == 2:
            # one double and one C integer: exact comparison, the selected operand's class
            function = name('_f64_i64' if order[0] == KIND_FLOAT else '_i64_f64')
        else:
            return None
        result = positional[0].value
        for argument in positional[1:]:
            result = cst.Call(function, (cst.Arg(result), cst.Arg(argument.value)))
        self.providers.add(MATH_PROVIDER)
        self.lowered += 1
        return result.with_changes(lpar=after.lpar, rpar=after.rpar)

    def visit_FunctionDef(self, node):
        if self.first_line is None:
            position = self.get_metadata(PositionProvider, node.name, None)
            self.first_line = position.start.line if position is not None else None
            if self.operand_kind is not None and self.providers is not None:
                # W1.8: the formula's own scope, locals and built-ins for the operand kinds
                from .expression_kinds import ExpressionKind, local_kinds
                from .helper_members import MATH_PROVIDER
                from .helpers import module_name
                self.formula_scope = self.get_metadata(ScopeProvider, node.body, None)
                self.kinds = ExpressionKind(
                    self.operand_kind, lambda item: self.get_metadata(ScopeProvider, item, None),
                    self.formula_scope, math_alias=module_name(MATH_PROVIDER))
                local_kinds(node, self.kinds)

    def leave_Call(self, before, after):
        integer = self.integer_identity(before, after)
        if integer is not None:
            return integer
        # A numeric Cell already normalizes its result through its native ABI.
        # Object results/child scopes must keep the selected operand's Python
        # scalar class: Cython promotes mixed native min/max inputs to double.
        # This fallback is about the result contract, not first-error order.
        if self.native_result and self.opaque_depth == 1:
            return after
        if not isinstance(before.func, cst.Name) or before.func.value not in ('min', 'max'):
            return after
        if any(argument.star for argument in after.args):
            return after
        positional = tuple(argument for argument in after.args if argument.keyword is None)
        if len(positional) < 2:
            return after
        scope = self.get_metadata(ScopeProvider, before.func)
        bindings = tuple(scope[before.func.value])
        if not bindings or not all(isinstance(binding, BuiltinAssignment) for binding in bindings):
            return after
        keywords = tuple(argument for argument in after.args if argument.keyword is not None)
        if not keywords:
            typed = self.typed_extrema(before, after, positional)
            if typed is not None:
                return typed
            objects = self.object_extrema(before, after, positional)
            if objects is not None:
                return objects
        if self.providers is not None:
            from .reductions import untyped_entry
            self.untyped.append(untyped_entry(self.line_of, before, before.func.value,
                'operands are not all typed (double or C integer; mixed only for two): called as '
                'the built-in to keep the result class' if not keywords else 'keyword arguments: built-in call'))
        values = cst.Tuple(tuple(cst.Element(argument.value) for argument in positional))
        return after.with_changes(args=(cst.Arg(values, star='*'), *keywords))

    def object_extrema(self, before, after, positional):
        """W1.8: min/max of operands not all of one C kind, compiled on Python objects.

        Every operand is cast to ``object``, so Cython's own min/max lowering (all
        operands evaluated left to right into temporaries, then ``PyObject_RichCompare``
        with ``<`` / ``>`` and a truth test, the first extreme kept) applies with no
        promotion between C types: the result is the selected operand itself, as the
        built-in returns it, and nan, -0.0, ties and comparison errors follow from the
        same comparisons the built-in makes. No argument tuple and no built-in call.
        """
        if self.providers is None:
            return None
        scope = self.get_metadata(ScopeProvider, before.func)
        if tuple(scope['_mx_cy']) or tuple(scope['object']) and not all(
                isinstance(binding, BuiltinAssignment) for binding in scope['object']):
            return None
        from .powers import KIND_FLOAT, KIND_INT, KIND_UNKNOWN
        from .reductions import untyped_entry
        kinds = self.operand_kinds(before)
        names = {KIND_FLOAT: 'float', KIND_INT: 'int', KIND_UNKNOWN: 'object'}
        self.objects.append(untyped_entry(self.line_of, before, before.func.value,
            'operands ' + ', '.join(names.get(kind, 'object') for kind in kinds)
            + ': compiled, operands are Python objects (object comparisons; the selected operand is the result)'))

        def boxed(value):
            return cst.Call(cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                            (cst.Arg(cst.Name('object')), cst.Arg(value)))
        return after.with_changes(args=tuple(cst.Arg(boxed(argument.value)) for argument in positional))

    def line_of(self, node):
        position = self.get_metadata(PositionProvider, node, None)
        if position is None or self.first_line is None:
            return 0
        return position.start.line - self.first_line + 1


def preserve_argument_order(formula, native_result=False, operand_kind=None, cells=None, providers=None):
    """Keep unproved powers on Cython's ordinary Python-object operator.

    With ``providers`` (W1.4), min/max of typed operands in object-result scopes become
    C comparisons (``_native_math.min_f64`` ...) instead of the boxed built-in call; the
    counts and the calls left boxed go to ``cells`` for the admission report.
    """
    wrapper = cst.MetadataWrapper(cst.Module(body=(formula,)))
    ordered = _OrderedArguments(native_result, operand_kind, providers)
    result = wrapper.visit(ordered).body[0]
    if cells is not None:
        cells.minmax_native, cells.minmax_untyped = ordered.lowered, ordered.untyped
        cells.minmax_object = ordered.objects
    return result
