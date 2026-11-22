# Copyright (c) 2023-2026 Fumito Hamamura <fumito.ham@gmail.com>

# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation version 3.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library.  If not, see <http://www.gnu.org/licenses/>.

"""Rewriting of ``**`` on C-typed operands into a real-valued call.

Cython gives ``**`` the semantics of Python's, so a C ``double`` base
raised to a fractional exponent has to be able to return a complex
number.  Unless the power is coerced to a C floating type directly,
Cython evaluates it on ``double complex`` and narrows the result back.
That is slower than a ``pow`` call, and it does not compile at all once
the result is compared, as in
``max((1 + rate(t)) ** (1 / 12) - 1, floor())``, which Cython rejects
with "complex types are unordered".

Setting Cython's ``cpow`` directive would fix that, but it would also
make ``**`` between two C integers stay integral, so a formula such as
``2 ** -t`` would silently return ``0`` rather than a fraction.  Cython
takes the complex path if and only if the exponent is a C floating
type, and the integral path if and only if the exponent is an integer,
so rewriting only those powers whose exponent is *provably* floating
removes the failure without touching integer arithmetic at all.

The proof is the conservative classifier below.  Every operand is
:data:`KIND_FLOAT`, :data:`KIND_INT` or :data:`KIND_UNKNOWN`, and only
numeric literals, the declaring cells' own parameters, calls to cells
the resolver reaches, scalar references, single-assignment locals
annotated with the builtin ``float``, and arithmetic over those are
ever known. Everything else -- other locals, subscripts, module
attributes, and anything inside a construct that binds names of its own,
where a name can shadow a parameter of the cells -- is unknown. This pass
leaves powers with unknown operands for ordinary Cell lowering. The object
operator in :mod:`~modelx_cython.ordered_arguments` preserves Python power
semantics where Cython would otherwise infer a complex intermediate.

A native double return can also use a uniquely assigned local from a
resolved double Cell or Reference. That local proof applies only to arithmetic directly
feeding the return, where Cython already requires a real result.

The same classifier decides, in :func:`demote_rebound_params`, whether a
parameter that a formula binds again keeps its C type there: only if
every value bound to it is provably of that type.

Attributes
----------
KIND_FLOAT : str
    The expression is a C floating-point value.
KIND_INT : str
    The expression is a C integer value.
KIND_UNKNOWN : str
    Nothing is known about the expression; never rewrite through it.
"""
import logging
from typing import Dict, List, Mapping, Optional

from collections import Counter
import libcst as cst
import libcst.matchers as m
from libcst.metadata import ExpressionContext

from modelx_cython.consts import CY_MOD, MX_POW, MX_SELF, MX_SYS_MOD
from modelx_cython.typedefs import CY_FLOAT_T, CY_INT_T
from modelx_cython.usage import RESOLVED, CellsResolver, _extract_self_chain
from modelx_cython.scopes import CstFacts, OPAQUE_SCOPES as _OPAQUE_SCOPES, has_scope_effects, lexical_nodes
from modelx_cython.syntax import local_assignment, single_statement, walk

_logger = logging.getLogger(__name__)

KIND_FLOAT = "float"
KIND_INT = "int"
KIND_UNKNOWN = "unknown"

# Expressions whose value carries the kind of their operands.  True
# division is excluded: it always produces a float in Python 3, which
# is what makes ``1 / 12`` a provably floating exponent.
_NUMERIC_OPS = (
    cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo,
    cst.Power,
)

# Constructs that introduce their own bindings.  A name bound by one of
# these can shadow a parameter of the cells, and the parameter's declared
# C type would not apply to it; rather than track the shadowing, no power
# inside one is ever rewritten.  A nested ``def`` or ``class`` belongs
# here as much as a comprehension does: modelx exports such a formula
# verbatim, and its body reaches Cython inside the ``@cfunc`` method.


def kind_of_type_expr(type_expr: str) -> str:
    """Classify a C-style type expression from :mod:`~modelx_cython.typedefs`.

    Only the exact scalar names count: ``bint`` is left unknown rather
    than treated as an integer, and a memoryview expression such as
    ``const double[:, :]`` does not match ``double``.
    """
    if type_expr == CY_FLOAT_T:
        return KIND_FLOAT
    elif type_expr == CY_INT_T:
        return KIND_INT
    else:
        return KIND_UNKNOWN


def build_kind_maps(module_infos: Mapping[str, object]):
    """Collect the kind of every cells return value and every ref.

    Parameters
    ----------
    module_infos : mapping of str to ModuleInfo
        The parsed modules, keyed by module fqname.

    Returns
    -------
    tuple of (dict, dict)
        ``cells_kinds`` maps a cells fqname to its return kind;
        ``ref_kinds`` maps a class fqname to a mapping of ref name to
        kind.
    """
    cells_kinds: Dict[str, str] = {}
    ref_kinds: Dict[str, Dict[str, str]] = {}
    for info in module_infos.values():
        for cls_info in info.classes.values():
            for cells in cls_info.cells.values():
                cells_kinds[cells.fqname] = kind_of_type_expr(
                    cells.get_rettype_expr(c_style=True)
                )
            ref_kinds[cls_info.fqname] = {
                name: kind_of_type_expr(ref.get_type_expr(c_style=True))
                for name, ref in cls_info.refs.items()
            }
    return cells_kinds, ref_kinds


class OperandKind:
    """Conservative kind classifier for the operands of a ``**``.

    Parameters
    ----------
    resolver : CellsResolver
        Resolves self-rooted attribute chains to cells fqnames.
    cells_kinds : mapping of str to str
        Return kind of every cells in the model, by cells fqname.
    ref_kinds : mapping of str to mapping
        Kind of each ref, by class fqname then ref name.
    cls_fqname : str
        Fqname of the class the formula being classified belongs to.
    param_kinds : mapping of str to str
        Kind of each parameter of the declaring cells, by name.
    """

    def __init__(
        self,
        resolver: CellsResolver,
        cells_kinds: Mapping[str, str],
        ref_kinds: Mapping[str, Mapping[str, str]],
        cls_fqname: str,
        param_kinds: Mapping[str, str],
    ) -> None:
        self._resolver = resolver
        self._cells_kinds = cells_kinds
        self._ref_kinds = ref_kinds
        self._cls_fqname = cls_fqname
        self._param_kinds = param_kinds

    def kind(self, expr: cst.BaseExpression) -> str:
        """Return the kind of ``expr``, or :data:`KIND_UNKNOWN`.

        Parentheses need no handling of their own: libcst attaches them
        to the expression node rather than wrapping it.
        """
        if isinstance(expr, cst.Float):
            return KIND_FLOAT
        elif isinstance(expr, cst.Integer):
            return KIND_INT
        elif isinstance(expr, cst.UnaryOperation):
            if isinstance(expr.operator, (cst.Plus, cst.Minus)):
                return self.kind(expr.expression)
            return KIND_UNKNOWN
        elif isinstance(expr, cst.BinaryOperation):
            return self._binop_kind(expr)
        elif isinstance(expr, cst.IfExp):
            # both branches of one kind give that kind (D59); mixed or unknown stay unknown
            body, orelse = self.kind(expr.body), self.kind(expr.orelse)
            return body if body == orelse else KIND_UNKNOWN
        elif isinstance(expr, cst.Name):
            return self._param_kinds.get(expr.value, KIND_UNKNOWN)
        elif isinstance(expr, cst.Call):
            return self._call_kind(expr)
        elif isinstance(expr, cst.Attribute):
            return self._ref_kind(expr)
        else:
            return KIND_UNKNOWN

    def _binop_kind(self, expr: cst.BinaryOperation) -> str:
        left = self.kind(expr.left)
        right = self.kind(expr.right)
        if KIND_UNKNOWN in (left, right):
            return KIND_UNKNOWN
        if isinstance(expr.operator, cst.Divide):
            return KIND_FLOAT       # true division, whatever the operands
        if isinstance(expr.operator, cst.Power) and left == right == KIND_INT:
            # An integral base raised to a negative integer is fractional.
            # Without a sign proof, its result cannot prove an outer exponent.
            exponent = expr.right
            sign = 1
            if isinstance(exponent, cst.UnaryOperation) and isinstance(exponent.operator, (cst.Plus, cst.Minus)):
                sign = -1 if isinstance(exponent.operator, cst.Minus) else 1
                exponent = exponent.expression
            if not isinstance(exponent, cst.Integer) or sign * exponent.evaluated_value < 0:
                return KIND_UNKNOWN
        if isinstance(expr.operator, _NUMERIC_OPS):
            return KIND_FLOAT if KIND_FLOAT in (left, right) else KIND_INT
        return KIND_UNKNOWN

    def _call_kind(self, expr: cst.Call) -> str:
        chain = _extract_self_chain(expr.func)
        if not isinstance(chain, list):
            return KIND_UNKNOWN
        status, fqname = self._resolver.resolve_chain(self._cls_fqname, chain)
        if status != RESOLVED:
            return KIND_UNKNOWN
        return self._cells_kinds.get(fqname, KIND_UNKNOWN)

    def _ref_kind(self, expr: cst.Attribute) -> str:
        chain = _extract_self_chain(expr)
        # only ``self.<ref>``: a longer chain would have to be followed
        # through child spaces, which this pass deliberately does not do
        if not isinstance(chain, list) or len(chain) != 1:
            return KIND_UNKNOWN
        return self._ref_kinds.get(self._cls_fqname, {}).get(
            chain[0], KIND_UNKNOWN
        )


MX_SIGNED_POW = "_mx_signed_pow"


def _nonnegative_integer(expr):
    """``expr`` is an integer literal >= 0, as ``+2`` or ``20`` (not ``-1``)."""
    if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, cst.Plus):
        expr = expr.expression
    return isinstance(expr, cst.Integer)


class _PowerRewriter(cst.CSTTransformer):
    """Replace provably real ``**`` operations with ``_mx_sys._mx_pow``."""

    def __init__(self, operand_kind: OperandKind, memo=None, return_operands=None, local_kind=None,
                 path=None) -> None:
        super().__init__()
        self._path = path                 # the Cell's "Space.cell", named by a checked integer power
        self.integer_rewritten = 0
        self._kind = operand_kind
        self._local_kind = local_kind     # report only: which declined powers are integer powers
        self._return_operands = return_operands or {}
        self._opaque_depth = 0
        self.rewritten = 0
        self.declined: List[str] = []
        self.declined_in_loops: List[str] = []
        self._loop_depth = 0
        self.memo = memo

    def on_visit(self, node: cst.CSTNode) -> bool:
        if isinstance(node, _OPAQUE_SCOPES):
            self._opaque_depth += 1
        if isinstance(node, (cst.For, cst.While, cst.CompFor)):
            self._loop_depth += 1
        return super().on_visit(node)

    def on_leave(self, original_node, updated_node):
        if isinstance(original_node, _OPAQUE_SCOPES):
            self._opaque_depth -= 1
        if isinstance(original_node, (cst.For, cst.While, cst.CompFor)):
            self._loop_depth -= 1
        return super().on_leave(original_node, updated_node)

    def _decline(self, node):
        text = cst.Module([]).code_for_node(node).strip()
        self.declined.append(text)
        if self._loop_depth:
            self.declined_in_loops.append(text)

    def leave_BinaryOperation(self, original_node, updated_node):
        if not isinstance(original_node.operator, cst.Power):
            return updated_node
        if self._opaque_depth:
            # a name here may be bound by the enclosing comprehension,
            # lambda or nested def rather than by the cells, so nothing
            # can be typed from the parameters
            self._decline(original_node)
            return updated_node
        # classify the original operands: a nested power has already
        # been rewritten into a call in ``updated_node``
        if original_node in self._return_operands:
            base, exponent = self._return_operands[original_node]
        else:
            base = self._kind.kind(original_node.left)
            exponent = self._kind.kind(original_node.right)
        integer_base = base
        if base == KIND_UNKNOWN and self._local_kind is not None:
            # a local Cython infers from a C integer (infer_types): formula_local_kinds
            integer_base = self._local_kind.kind(original_node.left)
        if (integer_base == KIND_INT and self._path is not None
                and not isinstance(original_node.left, cst.Integer)
                and _nonnegative_integer(original_node.right)):
            # Cython types a C integer raised to a non-negative integer literal as a C integer
            # and computes it unchecked (__Pyx_pow_*): the checked power raises instead of wrapping
            self.integer_rewritten += 1
            return cst.Call(
                func=cst.Attribute(value=cst.Name(MX_SYS_MOD), attr=cst.Name(MX_SIGNED_POW)),
                args=[cst.Arg(updated_node.left.with_changes(lpar=(), rpar=())),
                      cst.Arg(updated_node.right.with_changes(lpar=(), rpar=())),
                      cst.Arg(cst.SimpleString(repr(self._path)))],
                lpar=updated_node.lpar, rpar=updated_node.rpar)
        if exponent == KIND_FLOAT and base != KIND_UNKNOWN:
            self.rewritten += 1
            operation = MX_POW
            args = [
                cst.Arg(value=self._as_double(updated_node.left, base)),
                cst.Arg(value=updated_node.right),
            ]
            if self.memo:
                operation = "_mx_memo_pow"
                args.insert(0, cst.Arg(cst.parse_expression(f"{CY_MOD}.address(self.{self.memo})")))
            return cst.Call(
                func=cst.Attribute(
                    value=cst.Name(MX_SYS_MOD), attr=cst.Name(operation)
                ),
                args=args,
                lpar=updated_node.lpar,
                rpar=updated_node.rpar,
            )
        # Only a floating exponent puts a power on Cython's complex
        # path, so an integer one is left alone without comment; an exponent
        # that the formula's local kinds prove an integer is one too (diag-jp:
        # argument ordering, which knows those kinds, keeps its C power)
        if exponent in (KIND_FLOAT, KIND_UNKNOWN) and not (
                self._local_kind is not None and self._local_kind.kind(original_node.right) == KIND_INT):
            self._decline(original_node)
        return updated_node

    @staticmethod
    def _as_double(expr, kind):
        """Cast an integer base explicitly.

        ``_mx_pow`` takes two doubles, and letting the C compiler narrow
        the base implicitly costs an MSVC C4244 warning on a build that
        was warning-free before.
        """
        if kind != KIND_INT:
            return expr
        return cst.Call(
            func=cst.Attribute(value=cst.Name(CY_MOD), attr=cst.Name("cast")),
            args=[
                cst.Arg(value=cst.Attribute(
                    value=cst.Name(CY_MOD), attr=cst.Name(CY_FLOAT_T))),
                cst.Arg(value=expr.with_changes(lpar=(), rpar=())),
            ],
        )


def _native_return_locals(body, facts, operand_kind, parameters):
    """Prove a local from one dominating double Cell or Reference read."""
    if (facts.stores[MX_SELF] or has_scope_effects(body)
            or facts.reads & {'locals', 'vars', 'eval', 'exec', '_getframe', 'currentframe'}):
        return {}
    seen, result = set(parameters), {}
    for line in body:
        assignment = single_statement(line)
        if ((target := local_assignment(assignment)) is not None
                and target.value not in seen and facts.stores[target.value] == 1
                and target.value not in facts.deleted
                and isinstance(assignment.value, (cst.Call, cst.Attribute))
                and operand_kind.kind(assignment.value) == KIND_FLOAT):
            result[target.value] = KIND_FLOAT
        seen.update(part.value for part in walk(line) if isinstance(part, cst.Name)
                    and facts.contexts.get(part) is ExpressionContext.LOAD)
    return result


def _native_return_arithmetic(body, operand_kind):
    """Identify typed arithmetic through the native return boundary only."""
    def arithmetic(expr):
        if operand_kind.kind(expr) == KIND_UNKNOWN:
            return
        if isinstance(expr, cst.BinaryOperation):
            if isinstance(expr.operator, cst.Power):
                yield expr, (operand_kind.kind(expr.left), operand_kind.kind(expr.right))
            yield from arithmetic(expr.left)
            yield from arithmetic(expr.right)
        elif isinstance(expr, cst.UnaryOperation):
            yield from arithmetic(expr.expression)

    return {expr: kinds for stmt in lexical_nodes(body) if isinstance(stmt, cst.Return) and stmt.value is not None
            for expr, kinds in arithmetic(stmt.value)}


def rewrite_powers(
    node: cst.FunctionDef,
    resolver: Optional[CellsResolver],
    cells_kinds: Mapping[str, str],
    ref_kinds: Mapping[str, Mapping[str, str]],
    cls_fqname: str,
    cells,
    *,
    builtin_float=False,
    checked_integer_powers=True,
) -> cst.FunctionDef:
    """Rewrite the provably real powers in one method body.

    Parameters
    ----------
    node : libcst.FunctionDef
        The method whose body holds the exported formula.
    resolver : CellsResolver or None
        Resolver over the whole model; when None the body is returned
        unchanged, so a transformer built without one behaves as before.
    cells_kinds, ref_kinds : mapping
        As returned by :func:`build_kind_maps`.
    cls_fqname : str
        Fqname of the class the method belongs to.
    cells : CombinedCellsInfo
        The cells the method implements, for its parameter types.
    builtin_float : bool
        Whether the formula's lexical float binding is the builtin.

    Returns
    -------
    libcst.FunctionDef
        The method, with each rewritable ``**`` replaced by a call to
        ``_mx_sys._mx_pow``.  Every power left alone that could still be
        on Cython's complex path is logged at INFO.
    """
    if resolver is None:
        return node
    from modelx_cython import automatic
    memo = getattr(cells, '_spec', {}).get('memoize_power', automatic.enabled(cells))
    if type(memo) is not bool:
        raise ValueError('memoize_power must be a boolean')
    if memo:
        from modelx_cython.numeric_cache import exclusively_owned
        memo = f'_mx_power_{cells.name}' if exclusively_owned(cells) else None
    param_kinds = _param_kinds(cells)
    # A cells object that does not report a return type (any object that
    # implements only the parameter-type interface) has an unknown return
    # type, so the native-return arithmetic path does not apply to it.
    rettype = getattr(cells, 'get_rettype_expr', None)
    native_return = (isinstance(node.body, cst.IndentedBlock) and rettype is not None
                     and rettype(c_style=True) == CY_FLOAT_T
                     and any(isinstance(part, cst.Power) for part in lexical_nodes(node.body.body)))
    facts = CstFacts(node.body.body) if native_return else None
    if builtin_float and isinstance(node.body, cst.IndentedBlock):
        declarations = [
            decl.target.value for decl in lexical_nodes(node.body.body)
            if isinstance(decl, cst.AnnAssign) and isinstance(decl.target, cst.Name)
            and isinstance(decl.annotation.annotation, cst.Name)
            and decl.annotation.annotation.value == 'float' and decl.value is not None
        ]
        if declarations:
            facts = facts if facts is not None else CstFacts(node.body.body)
            # Cython gives an unshadowed float local a double declaration.
            param_kinds.update((name, KIND_FLOAT) for name in declarations
                               if facts.stores[name] == 1 and name not in cells.params
                               and name not in facts.deleted)
    kind = OperandKind(resolver, cells_kinds, ref_kinds, cls_fqname, param_kinds)
    return_operands = None
    if native_return:
        local_kinds = _native_return_locals(node.body.body, facts, kind, cells.params)
        if local_kinds:
            return_kind = OperandKind(resolver, cells_kinds, ref_kinds, cls_fqname, {**param_kinds, **local_kinds})
            return_operands = _native_return_arithmetic(node.body.body, return_kind)
    if isinstance(node.body, cst.IndentedBlock):
        accumulated = _double_accumulator_powers(
            node.body.body, cells.params, resolver, cells_kinds, ref_kinds, cls_fqname, param_kinds)
        if accumulated:
            return_operands = {**(return_operands or {}), **accumulated}
    local_kinds = formula_local_kinds(node, cells.params, resolver, cells_kinds, ref_kinds, cls_fqname, param_kinds)
    rewriter = _PowerRewriter(kind, memo, return_operands, OperandKind(
        resolver, cells_kinds, ref_kinds, cls_fqname, {**local_kinds, **param_kinds}) if local_kinds else None,
        path=getattr(cells, 'model_path', None) if checked_integer_powers else None)
    updated = node.body.visit(rewriter)
    # D62: powers left on Python's semantics inside a loop, for the admission report
    cells.powers_in_loops = list(rewriter.declined_in_loops)
    if rewriter.declined:
        _logger.info(
            f"'**' left as exported in {cells.formula_fqname}, so a "
            "fractional exponent there still compiles to complex "
            "arithmetic: " + "; ".join(rewriter.declined[:3])
        )
    if not rewriter.rewritten and not rewriter.integer_rewritten:
        return node
    if memo and rewriter.rewritten:
        cells.power_memo = memo
    return node.with_changes(body=updated)


class _LocalKind(OperandKind):
    """OperandKind for the local and sink analysis (D59, D62): also ``min``/``max``/``abs``
    of operands of one known kind and ``int()`` of a known number (the builtins; exported formulas read References through
    ``self``, so a bare ``min`` is the builtin unless the formula binds it, which disqualifies
    the formula through its scope facts)."""

    _CASTS = {'double': KIND_FLOAT, 'longlong': KIND_INT}

    def _call_kind(self, expr):
        func = expr.func
        # the compiler's own C cast (``_mx_cy.cast(_mx_cy.double, ...)``, emitted by the integer
        # overflow guard and the typed dispatch): its value has the C type it names
        if (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                and func.value.value == '_mx_cy' and func.attr.value == 'cast' and len(expr.args) == 2
                and not any(arg.keyword or arg.star for arg in expr.args)
                and isinstance(expr.args[0].value, cst.Attribute)
                and isinstance(expr.args[0].value.value, cst.Name) and expr.args[0].value.value.value == '_mx_cy'
                and expr.args[0].value.attr.value in self._CASTS):
            return self._CASTS[expr.args[0].value.attr.value]
        if (isinstance(func, cst.Name) and func.value in ('min', 'max', 'abs') and expr.args
                and not any(arg.keyword or arg.star for arg in expr.args)
                and (func.value != 'abs' or len(expr.args) == 1)
                and (func.value == 'abs' or len(expr.args) >= 2)):
            kinds = {self.kind(arg.value) for arg in expr.args}
            return kinds.pop() if len(kinds) == 1 and KIND_UNKNOWN not in kinds else KIND_UNKNOWN
        if (isinstance(func, cst.Name) and func.value == 'int' and len(expr.args) == 1
                and not expr.args[0].keyword and not expr.args[0].star):
            # int() of a known number is an integer (``n = int((last - start) // step)``)
            return KIND_INT if self.kind(expr.args[0].value) != KIND_UNKNOWN else KIND_UNKNOWN
        return super()._call_kind(expr)


_AUGMENTED = {
    cst.AddAssign: cst.Add, cst.SubtractAssign: cst.Subtract, cst.MultiplyAssign: cst.Multiply,
    cst.DivideAssign: cst.Divide, cst.FloorDivideAssign: cst.FloorDivide, cst.ModuloAssign: cst.Modulo,
}


_C_KINDS = {'longlong': KIND_INT, 'double': KIND_FLOAT}


def _declared_c_kinds(body, parameters):
    """Locals declared with one C scalar type (``x: _mx_cy.longlong``, with or without a value)
    and no other annotation: a C-typed local can only hold that type, whatever else the formula
    does (diag-jp: the loop of a lowered reduction, whose ``except ... as`` is a scope effect)."""
    found = {}
    for node in lexical_nodes(body):
        if isinstance(node, cst.AnnAssign) and isinstance(node.target, cst.Name):
            annotation = node.annotation.annotation
            ctype = (annotation.attr.value if isinstance(annotation, cst.Attribute)
                     and isinstance(annotation.value, cst.Name) and annotation.value.value == '_mx_cy'
                     else None)
            found.setdefault(node.target.value, set()).add(_C_KINDS.get(ctype))
    return {name: kinds.pop() for name, kinds in found.items()
            if name not in parameters and len(kinds) == 1 and None not in kinds}


def _assigned_local_kinds(body, parameters, resolver, cells_kinds, ref_kinds, cls_fqname, param_kinds):
    """Kinds of locals proven from every assignment to them (D59).

    A local qualifies when each of its bindings in the formula is a plain or annotated
    assignment, or an augmented ``+= -= *= /= //= %=``, at the formula's own level or in
    its ``if``/``while``/``for`` blocks (never in a comprehension, lambda or nested def),
    it is not a parameter and never deleted, and every assigned value has one known kind.
    An augmented assignment keeps the kind only when ``local op value`` has that same
    kind, so ``total = 0.0; total += x`` is a double and ``k = 1; k += 1`` an integer,
    while ``x = 0; x += 0.5`` is unknown. Formulas with scope effects (global, nonlocal,
    del, walrus, nested defs, ...) get no local kinds.
    """
    declared_kinds = _declared_c_kinds(body, parameters)
    if has_scope_effects(body):
        return declared_kinds   # only what C declarations fix (Cython enforces them)
    facts = CstFacts(body)
    if any(facts.stores[name] for name in ('min', 'max', 'abs', 'range', 'int')):
        return declared_kinds   # a formula that binds a builtin name: no inferred local kinds
    plain, augmented, declared = {}, {}, Counter()
    c_typed = {}
    for node in lexical_nodes(body):
        if isinstance(node, cst.AnnAssign) and isinstance(node.target, cst.Name):
            # diag-jp: a local declared with a C scalar type (``s: _mx_cy.longlong``,
            # ``b: _mx_cy.longlong = ...``, emitted by earlier passes) can only hold that type
            annotation = node.annotation.annotation
            ctype = (annotation.attr.value if isinstance(annotation, cst.Attribute)
                     and isinstance(annotation.value, cst.Name) and annotation.value.value == '_mx_cy'
                     else None)
            c_typed.setdefault(node.target.value, set()).add(_C_KINDS.get(ctype))
        if isinstance(node, cst.AnnAssign) and isinstance(node.target, cst.Name) and node.value is None:
            # a bare declaration (``s: _mx_cy.longlong``, added before this pass for a loop
            # variable or typed local) binds nothing: it neither gives nor removes a kind
            declared[node.target.value] += 1
        elif isinstance(node, cst.Assign) and len(node.targets) == 1 and isinstance(node.targets[0].target, cst.Name):
            plain.setdefault(node.targets[0].target.value, []).append(node.value)
        elif isinstance(node, cst.AnnAssign) and isinstance(node.target, cst.Name) and node.value is not None:
            plain.setdefault(node.target.value, []).append(node.value)
        elif (isinstance(node, cst.For) and isinstance(node.target, cst.Name) and not node.asynchronous
              and isinstance(node.iter, cst.Call) and isinstance(node.iter.func, cst.Name)
              and node.iter.func.value == 'range' and 1 <= len(node.iter.args) <= 3
              and not any(arg.keyword or arg.star for arg in node.iter.args)):
            # a loop over range(): an integer when every bound is (checked below)
            plain.setdefault(node.target.value, []).append(node.iter)
        elif isinstance(node, cst.AugAssign) and isinstance(node.target, cst.Name):
            operator = _AUGMENTED.get(type(node.operator))
            augmented.setdefault(node.target.value, []).append(
                None if operator is None else (operator(), node.value))
    candidates = {
        name for name, values in plain.items()
        if name not in parameters and name not in facts.deleted
        and facts.stores[name] == len(values) + len(augmented.get(name, ())) + declared[name]
        and None not in augmented.get(name, ())
    }
    kinds: Dict[str, str] = {name: found.pop() for name, found in c_typed.items()
                             if name not in parameters and name not in facts.deleted
                             and len(found) == 1 and None not in found}
    changed = True
    while changed:
        changed = False
        classify = _LocalKind(resolver, cells_kinds, ref_kinds, cls_fqname, {**param_kinds, **kinds})
        for name in sorted(candidates - set(kinds)):
            # a value may read the local itself (``total = total + x``): take the kind of the
            # values that are known without it, then require every value to keep that kind
            found = {_range_kind(classify, value) if _is_range(value) else classify.kind(value)
                     for value in plain[name]} - {KIND_UNKNOWN}
            if len(found) != 1:
                continue
            kind, = found
            trial = _LocalKind(resolver, cells_kinds, ref_kinds, cls_fqname,
                                {**param_kinds, **kinds, name: kind})
            if all((_range_kind(trial, value) if _is_range(value) else trial.kind(value)) == kind
                   for value in plain[name]) and all(
                    trial.kind(cst.BinaryOperation(cst.Name(name), operator, value)) == kind
                    for operator, value in augmented.get(name, ())):
                kinds[name] = kind
                changed = True
    return kinds


def formula_local_kinds(node, parameters, resolver, cells_kinds, ref_kinds, cls_fqname, param_kinds):
    """Kinds of the locals of a formula FunctionDef that every assignment proves (D59),
    for later passes that classify operands (argument order, typed min/max)."""
    if resolver is None or not isinstance(getattr(node, 'body', None), cst.IndentedBlock):
        return {}
    return _assigned_local_kinds(node.body.body, parameters, resolver, cells_kinds, ref_kinds,
                                 cls_fqname, param_kinds)


def _is_range(value):
    return isinstance(value, cst.Call) and isinstance(value.func, cst.Name) and value.func.value == 'range'


def _range_kind(classify, call):
    kinds = {classify.kind(arg.value) for arg in call.args}
    return KIND_INT if kinds == {KIND_INT} else KIND_UNKNOWN


def _double_accumulator_powers(body, parameters, resolver, cells_kinds, ref_kinds, cls_fqname, param_kinds):
    """Powers whose value flows into a C double local (D59, D62).

    A sink is a local whose every assignment is a double expression (``_assigned_local_kinds``)
    and which has at least one assignment that is a double without any power in it
    (``total = 0.0``): its C type does not depend on the powers, so Cython declares it a
    double either way and a complex power result could not be stored in it. Each power
    in the arithmetic of a value assigned to a sink (``total += c * v ** ((s - t) / 12.0)``)
    with a floating exponent and a known base is native, raising as ``_mx_pow`` does where
    Python would give a complex number. A local whose only double assignments are powers
    (``root = base ** 0.5``) is not a sink, so its power keeps Python's semantics.
    """
    kinds = _assigned_local_kinds(body, parameters, resolver, cells_kinds, ref_kinds, cls_fqname, param_kinds)
    if not any(kind == KIND_FLOAT for kind in kinds.values()):
        return {}
    classify = _LocalKind(resolver, cells_kinds, ref_kinds, cls_fqname, {**param_kinds, **kinds})

    def has_power(expr):
        return any(isinstance(part, cst.BinaryOperation) and isinstance(part.operator, cst.Power)
                   for part in walk(expr))

    assignments = []
    for statement in lexical_nodes(body):
        if isinstance(statement, cst.Assign) and len(statement.targets) == 1:
            target = statement.targets[0].target
        elif isinstance(statement, (cst.AnnAssign, cst.AugAssign)):
            target = statement.target
        else:
            continue
        if isinstance(target, cst.Name) and statement.value is not None:
            assignments.append((target.value, statement))
    sinks = {name for name, statement in assignments
             if kinds.get(name) == KIND_FLOAT and not isinstance(statement, cst.AugAssign)
             and classify.kind(statement.value) == KIND_FLOAT and not has_power(statement.value)}

    def arithmetic(expr):
        if isinstance(expr, cst.BinaryOperation):
            if isinstance(expr.operator, cst.Power):
                yield expr, (classify.kind(expr.left), classify.kind(expr.right))
            yield from arithmetic(expr.left)
            yield from arithmetic(expr.right)
        elif isinstance(expr, cst.UnaryOperation):
            yield from arithmetic(expr.expression)

    found = {}
    for name, statement in assignments:
        if name in sinks:
            for power, operands in arithmetic(statement.value):
                if operands[1] == KIND_FLOAT and operands[0] != KIND_UNKNOWN:
                    found[power] = operands
    return found


def _param_kinds(cells) -> Dict[str, str]:
    """The kind of each parameter of ``cells`` in its formula."""
    param_kinds = {}
    if cells.has_args() and cells.has_typeinfo():
        for param in cells.params:
            param_kinds[param] = kind_of_type_expr(
                cells.get_argtype_expr(param, c_style=True, formula=True)
            )
    return param_kinds


# ---------------------------------------------------------------------------
# Parameters that a formula binds again
#
# A parameter keeps its C type in the formula only if every value the
# formula binds to it is provably of that type, by the classifier above.
# Otherwise Cython rejects the formula, as for ``rate = rate / 100`` with
# an integer ``rate`` ("Cannot assign type 'double' to 'long long'"), or,
# for ``t /= 2``, it compiles and truncates the quotient.  The bindings
# are collected by modelx_cython.parser.collect_param_rebinds.

def _rebind_fits(value, ctype: str, kind: OperandKind) -> bool:
    """Whether ``value``, bound to a parameter of C type ``ctype``, is
    provably of that type.

    A value of kind :data:`KIND_INT` fits a ``long long`` unless it has
    a ``**``, whose type Cython derives from the sign of the exponent
    (``2 ** -t`` is a double); one of kind :data:`KIND_INT` or
    :data:`KIND_FLOAT` fits a ``double``.  Nothing fits another C type,
    and a binding whose value is unknown (a for loop, unpacking, ...)
    fits none.
    """
    if not isinstance(value, cst.BaseExpression):
        return False
    if ctype == CY_INT_T:
        return (kind.kind(value) == KIND_INT
                and not m.findall(value, m.Power()))
    elif ctype == CY_FLOAT_T:
        return kind.kind(value) in (KIND_INT, KIND_FLOAT)
    return False


def demote_rebound_params(
    module_infos: Mapping[str, object],
    resolver: CellsResolver,
    cells_kinds: Mapping[str, str],
    ref_kinds: Mapping[str, Mapping[str, str]],
) -> None:
    """Type ``object`` the parameters a formula binds to other types.

    For every cells with type information whose formula binds one of
    its C-typed parameters again (see
    :attr:`~modelx_cython.builder.CombinedCellsInfo.param_rebinds`),
    each bound value must be provably of the parameter's C type (see
    :func:`_rebind_fits`), classified with the parameter types of the
    formula.  If one is not, the parameter is typed ``object`` in the
    formula with
    :meth:`~modelx_cython.builder.CombinedCellsInfo.demote_rebound_param`,
    which logs it.  A parameter typed ``object`` is unknown to the
    classifier, so the check is repeated until no parameter changes.

    Run once the return types of all cells are final, as they are for
    :func:`build_kind_maps`, and before any parameter type is read.

    Parameters
    ----------
    module_infos : mapping of str to ModuleInfo
        The parsed modules, keyed by module fqname.
    resolver : CellsResolver
        As for :func:`rewrite_powers`.
    cells_kinds, ref_kinds : mapping
        As returned by :func:`build_kind_maps`.
    """
    for info in module_infos.values():
        for cls_info in info.classes.values():
            for cells in cls_info.cells.values():
                if not (cells.param_rebinds and cells.has_typeinfo()):
                    continue
                changed = True
                while changed:
                    changed = False
                    kind = OperandKind(resolver, cells_kinds, ref_kinds,
                                       cls_info.fqname, _param_kinds(cells))
                    for param, values in cells.param_rebinds.items():
                        ctype = cells.get_argtype_expr(
                            param, c_style=True, formula=True)
                        if ctype == "object":
                            continue
                        for value in values:
                            if not _rebind_fits(value, ctype, kind):
                                cells.demote_rebound_param(
                                    param, _describe_rebind(value, ctype))
                                changed = True
                                break


def _describe_rebind(value, ctype: str) -> str:
    if isinstance(value, cst.BaseExpression):
        code = cst.Module([]).code_for_node(value).strip()
        return (f"the formula binds it to {code}, which is not provably "
                f"a {ctype}")
    return f"the formula binds it in {value}"
