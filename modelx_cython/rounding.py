"""Lower lexically resolved calls to the ordinary guarded round helper, and ``round(x, n)``
returns of double formulas to the repeated-round provider."""
import libcst as cst

from .consts import MX_SYS_MOD
from .helpers import module_name
from .scopes import OPAQUE_SCOPES, CstFacts, Names, has_scope_effects
from .syntax import arguments, literal_value, single_statement, walk
from .typedefs import primitive_kind


NATIVE_ROUNDING_PROVIDER = 'modelx_cython.runtime._native_rounding'


ROUND_BINDING = """\
import builtins as _mx_round_builtins
from types import BuiltinFunctionType as _mx_round_builtin_type
_mx_canonical_round = _mx_round_builtins.round
if (type(_mx_canonical_round) is not _mx_round_builtin_type
        or _mx_canonical_round.__self__ is not _mx_round_builtins
        or _mx_canonical_round.__name__ != 'round'):
    _mx_canonical_round = None
"""


class NativeRounding(cst.CSTTransformer):
    def __init__(self):
        self.used = False

    def on_visit(self, node):
        return False if isinstance(node, OPAQUE_SCOPES) else super().on_visit(node)

    def leave_Call(self, original, updated):
        args = arguments(original, 'round')
        if args is None or len(args) not in (1, 2):
            return updated
        self.used = True
        operation = '_mx_round_one' if len(args) == 1 else '_mx_round_digits'
        return updated.with_changes(
            func=cst.Attribute(cst.Name(MX_SYS_MOD), cst.Name(operation)),
            args=(cst.Arg(updated.func), *updated.args, cst.Arg(cst.Name('_mx_canonical_round'))))


class RoundedInputs(cst.CSTTransformer):
    """``return round(x, n)`` at a double formula's return with a float ``x`` and a C integer
    ``n``: the repeated-round provider call, which keeps the private lookup context of the cell.

    ``operations`` lists the provider operations used, ``contexts`` the context fields added
    (name -> provider)."""

    def __init__(self, cls, cell, providers, builtin_keys=(), resolver=None):
        self.cls, self.cell, self.providers = cls, cell, providers
        self.contexts = {}
        self.operations = set()
        self.builtin_keys = builtin_keys
        self.resolver = resolver
        self.function_depth = 0
        self.scope_safe = True

    def visit_FunctionDef(self, node):
        if not self.function_depth:
            self.scope_safe = not has_scope_effects(node.body.body)
        self.function_depth += 1

    def leave_FunctionDef(self, before, after):
        self.function_depth -= 1
        if (self.function_depth or not self.scope_safe or 'min' not in self.builtin_keys
                or self.cell.is_locked or self.cell.cls in self.cls.visitor.dynamic_ref_classes
                or self.cell.get_rettype_expr(c_style=True) != 'double'
                or not isinstance(after.body, cst.IndentedBlock) or len(after.body.body) < 4):
            return after
        rounded = self.rounded_inputs(after.body.body)
        return after if rounded is None else after.with_changes(
            body=after.body.with_changes(body=rounded))

    def locked_result_type(self, node):
        if self.resolver is None or not isinstance(node, cst.Call) or node.args:
            return None
        from .usage import RESOLVED, _extract_self_chain
        chain = _extract_self_chain(node.func)
        if not isinstance(chain, list):
            return None
        status, name = self.resolver.resolve_chain(self.cls.fqname, chain)
        return self.resolver.locked_scalar_results.get(name) if status == RESOLVED else None

    def native_key(self, key):
        integer_types = ('long long',)
        if isinstance(key, cst.Integer):
            return primitive_kind(literal_value(key)) == 'longlong'
        if isinstance(key, cst.Name) and key.value in self.cell.params:
            return self.cell.get_argtype_expr(key.value, c_style=True, formula=True) in integer_types
        if (isinstance(key, cst.Attribute) and isinstance(key.value, cst.Name)
                and key.value.value == 'self'):
            ref = self.cls.refs.get(key.attr.value)
            return ref is not None and ref.get_type_expr(c_style=True) in integer_types
        if (isinstance(key, cst.Call) and isinstance(key.func, cst.Name)
                and key.func.value in self.builtin_keys and len(key.args) >= 2
                and not any(a.keyword or a.star for a in key.args)):
            return all(self.native_key(a.value) for a in key.args)
        if (isinstance(key, cst.Call) and isinstance(key.func, cst.Attribute)
                and isinstance(key.func.value, cst.Name) and key.func.value.value == 'self'):
            cell = self.cls.cells.get(key.func.attr.value)
            return (cell is not None and not cell.is_special() and not cell.called_with_kwargs
                    and not cell.body_has_closure
                    and cell.get_rettype_expr(c_style=True) in integer_types)
        return self.locked_result_type(key) in integer_types

    def lookup_context(self, provider):
        for name, owner in self.contexts.items():
            if owner == provider:
                return cst.parse_expression('self.' + name)
        context = f'_v_mx_context_{self.cell.name}'
        # a second provider owns a distinct extension type and never receives the first's context
        if context in self.contexts:
            context += module_name(provider)
        cache_fields = (prefix + name for name in self.cls.cells for prefix in ('_v_', '_has_'))
        previous = getattr(self.cls, 'native_contexts', {})
        context = Names(used=(*previous, *self.contexts, *self.cls.refs, *cache_fields)).reserve(context)
        self.contexts[context] = provider
        return cst.parse_expression('self.' + context)

    def rounded_inputs(self, body):
        if 'round' not in self.builtin_keys or self.resolver is None:
            return None
        result = single_statement(body[-1])
        if not isinstance(result, cst.Return):
            return None
        expression = result.value
        choices = (expression.body, expression.orelse) if isinstance(expression, cst.IfExp) else (expression,)
        calls = [node for node in choices if arguments(node, 'round', 2) is not None]
        if len(calls) != 1:
            return None
        call, = calls
        from .powers import OperandKind, KIND_FLOAT, KIND_INT, KIND_UNKNOWN, kind_of_type_expr
        facts = CstFacts(body)
        cells = {cell.fqname: kind_of_type_expr(cell.get_rettype_expr(c_style=True))
                 for cls in self.cls.module.classes.values() for cell in cls.cells.values()
                 if cell.has_typeinfo()}
        refs = {self.cls.fqname: {
            name: kind_of_type_expr(ref.get_type_expr(c_style=True))
            for name, ref in self.cls.refs.items()}}
        kinds = {name: kind_of_type_expr(self.cell.get_argtype_expr(name, c_style=True, formula=True))
                 for name in self.cell.params}
        classifier = OperandKind(self.resolver, cells, refs, self.cls.fqname, kinds)
        integer_locals = set()
        for line in body[:-1]:
            statement = single_statement(line)
            if (isinstance(statement, cst.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0].target, cst.Name)):
                name = statement.targets[0].target.value
                kind = classifier.kind(statement.value)
                if (facts.stores[name] == 1 and kind != KIND_UNKNOWN
                        and not any(isinstance(node, cst.Power) for node in walk(statement.value))):
                    kinds[name] = kind
                    if kind == KIND_INT and self.native_key(statement.value):
                        integer_locals.add(name)
        number, digits = (arg.value for arg in call.args)
        if (classifier.kind(number) != KIND_FLOAT
                or any(isinstance(node, cst.Power) for node in walk(number))
                or not (self.native_key(digits)
                        or isinstance(digits, cst.Name) and digits.value in integer_locals)):
            return None
        operation = 'repeated_round_float'
        provider = NATIVE_ROUNDING_PROVIDER
        replacement = call.with_changes(
            func=cst.Attribute(cst.Name(module_name(provider)), cst.Name(operation)),
            args=(cst.Arg(self.lookup_context(provider)), cst.Arg(call.func), *call.args))
        self.providers.add(provider)
        self.operations.add(f'{provider}.{operation}')
        return (*body[:-1], body[-1].visit(_ReplaceExpression(call, replacement)))


class _ReplaceExpression(cst.CSTTransformer):
    def __init__(self, source, replacement):
        self.source, self.replacement = source, replacement

    def on_leave(self, before, after):
        return self.replacement if before is self.source else after
