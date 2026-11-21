"""W1.8: Python classes of the operands of built-in calls in one formula.

The kinds are those of :mod:`modelx_cython.powers` (``KIND_FLOAT``: the value is a
Python ``float`` in the interpreter and a C ``double`` compiled; ``KIND_INT``: an ``int``
held in a C ``long long``; ``KIND_UNKNOWN``: anything else). This classifier adds, on top
of :class:`powers.OperandKind` (literals, typed parameters, typed cells and References):

- formula locals whose every binding is a plain assignment (``x = v``, ``x: T = v``,
  ``x op= v``) of one kind; solved as a greatest fixpoint, so ``worst = max(worst, ...)``
  keeps its kind;
- built-ins: ``float(x)`` (always a float), ``abs(x)`` (the kind of ``x``), ``len(x)``
  (an integer), ``round(x, n)`` with a float ``x`` and an integer ``n`` (a float), and
  ``min``/``max`` of operands of one kind (that kind);
- calls of the math provider written by the W1.4 binding (``m_exp`` ... float,
  ``m_floor_ll``/``m_ceil_ll`` integer).

Names count only where they resolve to the formula's own scope (a comprehension or lambda
variable of the same name is not the parameter or local). Mixed int/float operands give
``KIND_UNKNOWN``: the class of such a result depends on the values.
"""
import libcst as cst
from libcst.metadata import ScopeProvider
from libcst.metadata.scope_provider import BuiltinAssignment

from .powers import KIND_FLOAT, KIND_INT, KIND_UNKNOWN
from .scopes import OPAQUE_SCOPES, lexical_nodes
from .syntax import walk

_ARITHMETIC = (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo)
_AUGMENTED = {cst.AddAssign: cst.Add, cst.SubtractAssign: cst.Subtract,
              cst.MultiplyAssign: cst.Multiply, cst.DivideAssign: cst.Divide,
              cst.FloorDivideAssign: cst.FloorDivide, cst.ModuloAssign: cst.Modulo}
_MATH_FLOAT = ('m_erf', 'm_exp', 'm_log', 'm_sqrt')
_MATH_INT = ('m_floor_ll', 'm_ceil_ll')


def join(kinds):
    kinds = set(kinds)
    return kinds.pop() if len(kinds) == 1 else KIND_UNKNOWN


class ExpressionKind:
    """Kind of an expression of one formula (see the module docstring).

    ``base`` is the formula's :class:`powers.OperandKind`; ``scope_of(node)`` returns the
    libcst scope of a node or None (then a name is trusted only where it is no local of a
    nested scope, which the caller guarantees); ``formula_scope`` is the function scope;
    ``locals_`` maps formula locals to kinds (see :func:`local_kinds`).
    """

    def __init__(self, base, scope_of=None, formula_scope=None, locals_=None, math_alias=None):
        self.base = base
        self.scope_of = scope_of
        self.formula_scope = formula_scope
        self.locals = dict(locals_ or {})
        self.math_alias = math_alias

    # ------------------------------------------------------------ names and builtins
    def _own(self, node, name):
        """True when ``name`` read at ``node`` is the formula's own binding (or no binding)."""
        if self.scope_of is None:
            return True
        scope = self.scope_of(node)
        if scope is None:
            return False
        if scope is self.formula_scope:
            return True
        # a nested scope: the name must not be bound there (or in any scope in between)
        current = scope
        while current is not None and current is not self.formula_scope:
            if any(assignment.scope is current for assignment in current.assignments[name]):
                return False
            current = current.parent
        return current is self.formula_scope

    def builtin(self, node):
        """``node`` is a Name bound to the built-in of that name (no local rebinding)."""
        if not isinstance(node, cst.Name):
            return False
        if node.value in self.locals or node.value in getattr(self.base, '_param_kinds', {}):
            return False
        if self.scope_of is None:
            return False
        scope = self.scope_of(node)
        if scope is None:
            return False
        assignments = tuple(scope[node.value])
        return bool(assignments) and all(isinstance(a, BuiltinAssignment) for a in assignments)

    # ------------------------------------------------------------ classification
    def kind(self, expr):
        if isinstance(expr, cst.Name):
            if expr.value in self.locals:
                return self.locals[expr.value] if self._own(expr, expr.value) else KIND_UNKNOWN
            if expr.value in getattr(self.base, '_param_kinds', {}) and not self._own(expr, expr.value):
                return KIND_UNKNOWN
            return self.base.kind(expr)
        if isinstance(expr, cst.UnaryOperation):
            if isinstance(expr.operator, (cst.Plus, cst.Minus)):
                return self.kind(expr.expression)
            return KIND_UNKNOWN
        if isinstance(expr, cst.BinaryOperation):
            return self._binop(expr)
        if isinstance(expr, cst.IfExp):
            return self._ifexp(expr)
        if isinstance(expr, cst.Call):
            kind = self._builtin_call(expr)
            if kind is not None:
                return kind
            kind = self._math_call(expr)
            if kind is not None:
                return kind
            kind = self._rounding_call(expr)
            if kind is not None:
                return kind
        return self.base.kind(expr)

    def _rounding_call(self, expr):
        """``round(x, n)`` as the rounding pass writes it: ``_mx_sys._mx_round_digits(round,
        x, n, _mx_canonical_round)``; a float for a float ``x`` and an integer ``n`` when
        ``round`` is the built-in (the fallback calls that same built-in)."""
        func = expr.func
        if not (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                and func.value.value == '_mx_sys' and func.attr.value == '_mx_round_digits'
                and len(expr.args) == 4 and not any(a.star or a.keyword for a in expr.args)):
            return None
        function, value, digits = (a.value for a in expr.args[:3])
        if (isinstance(function, cst.Name) and function.value == 'round' and self.builtin(function)
                and self.kind(value) == KIND_FLOAT and self.kind(digits) == KIND_INT):
            return KIND_FLOAT
        return KIND_UNKNOWN

    def _binop(self, expr):
        left, right = self.kind(expr.left), self.kind(expr.right)
        if KIND_UNKNOWN in (left, right):
            return KIND_UNKNOWN
        if isinstance(expr.operator, cst.Divide):
            return KIND_FLOAT
        if isinstance(expr.operator, _ARITHMETIC):
            return KIND_FLOAT if KIND_FLOAT in (left, right) else KIND_INT
        # ``**`` and the rest keep the conservative classifier's answer
        return self.base._binop_kind(expr) if hasattr(self.base, '_binop_kind') else KIND_UNKNOWN

    def _ifexp(self, expr):
        return join((self.kind(expr.body), self.kind(expr.orelse)))

    def _builtin_call(self, expr):
        if not isinstance(expr.func, cst.Name) or any(a.star or a.keyword for a in expr.args):
            return None
        name, args = expr.func.value, [a.value for a in expr.args]
        if name not in ('float', 'abs', 'len', 'round', 'min', 'max') or not self.builtin(expr.func):
            return None
        if name == 'float' and len(args) == 1:
            return KIND_FLOAT
        if name == 'abs' and len(args) == 1:
            kind = self.kind(args[0])
            return kind if kind in (KIND_FLOAT, KIND_INT) else KIND_UNKNOWN
        if name == 'len' and len(args) == 1:
            return KIND_INT
        if name == 'round' and len(args) == 2:
            return KIND_FLOAT if (self.kind(args[0]) == KIND_FLOAT
                                  and self.kind(args[1]) == KIND_INT) else KIND_UNKNOWN
        if name in ('min', 'max') and len(args) >= 2:
            return join(self.kind(arg) for arg in args)
        return KIND_UNKNOWN

    def _math_call(self, expr):
        func = expr.func
        if (self.math_alias is None or not isinstance(func, cst.Attribute)
                or not isinstance(func.value, cst.Name) or func.value.value != self.math_alias):
            return None
        name = func.attr.value
        if name in _MATH_FLOAT:
            return KIND_FLOAT
        if name in _MATH_INT:
            return KIND_INT
        if name in ('min_f64', 'max_f64'):
            return KIND_FLOAT
        if name in ('min_i64', 'max_i64'):
            return KIND_INT
        return None


def local_kinds(formula, classifier):
    """Kinds of the formula's locals bound only by plain assignments (greatest fixpoint).

    ``classifier`` is an :class:`ExpressionKind`; its ``locals`` are replaced. Locals with
    any other binding (``for`` targets, ``with ... as``, unpacking, ``del``, ``global``,
    imports, walrus anywhere in the formula), and parameters, are left out.
    """
    if not isinstance(formula, cst.FunctionDef) or not isinstance(formula.body, cst.IndentedBlock):
        return {}
    if any(isinstance(node, cst.NamedExpr) for node in walk(formula.body)):
        return {}
    # a nested function may rebind a formula local through ``nonlocal``
    rebound = {item.name.value for node in walk(formula.body) if isinstance(node, cst.Nonlocal)
               for item in node.names}
    params = formula.params
    parameters = {p.name.value for p in (*params.posonly_params, *params.params, *params.kwonly_params)}
    parameters |= {p.name.value for p in (params.star_arg, params.star_kwarg) if isinstance(p, cst.Param)}
    values, plain, stores = {}, set(), set()
    for node in lexical_nodes(formula.body.body):
        if isinstance(node, cst.Assign):
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    values.setdefault(target.target.value, []).append(node.value)
                    plain.add(id(target.target))
        elif isinstance(node, cst.AnnAssign) and isinstance(node.target, cst.Name):
            plain.add(id(node.target))
            if node.value is not None:
                values.setdefault(node.target.value, []).append(node.value)
            else:
                values.setdefault(node.target.value, [])
        elif isinstance(node, cst.AugAssign) and isinstance(node.target, cst.Name):
            plain.add(id(node.target))
            operator = _AUGMENTED.get(type(node.operator))
            values.setdefault(node.target.value, []).append(
                cst.BinaryOperation(cst.Name(node.target.value), operator(), node.value)
                if operator is not None else None)
    from .scopes import CstFacts
    facts = CstFacts(formula.body.body)
    for node in lexical_nodes(formula.body.body):
        if isinstance(node, cst.Name) and id(node) not in plain and facts.contexts.get(node) is not None \
                and facts.contexts[node].name in ('STORE', 'DEL'):
            stores.add(node.value)
        elif isinstance(node, (cst.Global, cst.Nonlocal)):
            stores.update(item.name.value for item in node.names)
        elif isinstance(node, (cst.FunctionDef, cst.ClassDef)):
            stores.add(node.name.value)
        elif isinstance(node, cst.ImportAlias):
            stores.add('*')
        elif isinstance(node, cst.ExceptHandler) and node.name is not None:
            stores.update(n.value for n in walk(node.name.name) if isinstance(n, cst.Name))
        elif isinstance(node, (cst.Import, cst.ImportFrom)):
            return {}
    candidates = {name: exprs for name, exprs in values.items()
                  if name not in stores and name not in parameters and name not in rebound
                  and exprs and None not in exprs}
    # seed: a kind from any value computable with the locals seeded so far
    kinds = {}
    changed = True
    while changed:
        changed = False
        for name, exprs in candidates.items():
            if name in kinds:
                continue
            classifier.locals = kinds
            for expr in exprs:
                kind = classifier.kind(expr)
                if kind != KIND_UNKNOWN:
                    kinds[name] = kind
                    changed = True
                    break
    # verify: every value of a local has its kind, given the others (demotion only)
    changed = True
    while changed:
        changed = False
        for name in list(kinds):
            classifier.locals = kinds
            if any(classifier.kind(expr) != kinds[name] for expr in candidates[name]):
                del kinds[name]
                changed = True
    classifier.locals = kinds
    return dict(kinds)
