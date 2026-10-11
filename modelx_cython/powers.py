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
the resolver reaches, scalar references and arithmetic over those are
ever known.  Everything else -- local variables, subscripts, module
attributes, and anything inside a construct that binds names of its own,
where a name can shadow a parameter of the cells -- is unknown, and an
unknown operand leaves the power exactly as modelx exported it.  A power
that is left alone therefore behaves as it does without this pass:
usually it compiles, and where it is compared it still fails, which is a
loud error rather than a wrong number.

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

import libcst as cst
import libcst.matchers as m

from modelx_cython.consts import CY_MOD, MX_POW, MX_SYS_MOD
from modelx_cython.typedefs import CY_FLOAT_T, CY_INT_T
from modelx_cython.usage import RESOLVED, CellsResolver, _extract_self_chain

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
_OPAQUE_SCOPES = (
    cst.ListComp, cst.SetComp, cst.DictComp, cst.GeneratorExp, cst.Lambda,
    cst.FunctionDef, cst.ClassDef,
)

# The functions of _mx_sys.pxd that the rewrites below call, all of which
# return a C double
_MX_SYS_FLOAT_FUNCS = frozenset([MX_POW, "_mx_exp", "_mx_log", "_mx_math_pow"])


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
        if isinstance(expr.operator, _NUMERIC_OPS):
            return KIND_FLOAT if KIND_FLOAT in (left, right) else KIND_INT
        return KIND_UNKNOWN

    def _call_kind(self, expr: cst.Call) -> str:
        func = expr.func
        if (isinstance(func, cst.Attribute)
                and isinstance(func.value, cst.Name)
                and func.value.value == MX_SYS_MOD
                and func.attr.value in _MX_SYS_FLOAT_FUNCS):
            # a call this module generated, such as ``_mx_sys._mx_pow(...)``
            # from a rewritten ``**``, which the math pass then sees
            return KIND_FLOAT
        chain = _extract_self_chain(func)
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


class _PowerRewriter(cst.CSTTransformer):
    """Replace provably real ``**`` operations with ``_mx_sys._mx_pow``."""

    def __init__(self, operand_kind: OperandKind) -> None:
        super().__init__()
        self._kind = operand_kind
        self._opaque_depth = 0
        self.rewritten = 0
        self.declined: List[str] = []

    def on_visit(self, node: cst.CSTNode) -> bool:
        if isinstance(node, _OPAQUE_SCOPES):
            self._opaque_depth += 1
        return super().on_visit(node)

    def on_leave(self, original_node, updated_node):
        if isinstance(original_node, _OPAQUE_SCOPES):
            self._opaque_depth -= 1
        return super().on_leave(original_node, updated_node)

    def leave_BinaryOperation(self, original_node, updated_node):
        if not isinstance(original_node.operator, cst.Power):
            return updated_node
        if self._opaque_depth:
            # a name here may be bound by the enclosing comprehension,
            # lambda or nested def rather than by the cells, so nothing
            # can be typed from the parameters
            self.declined.append(
                cst.Module([]).code_for_node(original_node).strip()
            )
            return updated_node
        # classify the original operands: a nested power has already
        # been rewritten into a call in ``updated_node``
        base = self._kind.kind(original_node.left)
        exponent = self._kind.kind(original_node.right)
        if exponent == KIND_FLOAT and base != KIND_UNKNOWN:
            self.rewritten += 1
            return cst.Call(
                func=cst.Attribute(
                    value=cst.Name(MX_SYS_MOD), attr=cst.Name(MX_POW)
                ),
                args=[
                    cst.Arg(value=self._as_double(updated_node.left, base)),
                    cst.Arg(value=updated_node.right),
                ],
                lpar=updated_node.lpar,
                rpar=updated_node.rpar,
            )
        # Only a floating exponent puts a power on Cython's complex
        # path, so an integer one is left alone without comment.
        if exponent in (KIND_FLOAT, KIND_UNKNOWN):
            self.declined.append(
                cst.Module([]).code_for_node(original_node).strip()
            )
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


def rewrite_powers(
    node: cst.FunctionDef,
    resolver: Optional[CellsResolver],
    cells_kinds: Mapping[str, str],
    ref_kinds: Mapping[str, Mapping[str, str]],
    cls_fqname: str,
    cells,
    math_refs: Optional[Mapping[str, set]] = None,
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
    math_refs : mapping, optional
        As for :func:`rewrite_math_calls`, given when the spec sets
        ``"use_libm"``: a ``math`` call that :func:`rewrite_math_calls`
        rewrites afterwards is a C double, so a ``**`` on it is
        rewritten too, as it must be.  Left alone, it would go on
        Cython's complex path once the call is a C double, and fail to
        compile where it is compared.

    Returns
    -------
    libcst.FunctionDef
        The method, with each rewritable ``**`` replaced by a call to
        ``_mx_sys._mx_pow``.  Every power left alone that could still be
        on Cython's complex path is logged at INFO.
    """
    if resolver is None:
        return node
    param_kinds = _param_kinds(cells)
    if math_refs and cls_fqname in math_refs:
        kind = _MathOperandKind(resolver, cells_kinds, ref_kinds, cls_fqname,
                                param_kinds, math_refs=math_refs[cls_fqname])
    else:
        kind = OperandKind(resolver, cells_kinds, ref_kinds, cls_fqname,
                           param_kinds)
    rewriter = _PowerRewriter(kind)
    updated = node.body.visit(rewriter)
    if rewriter.declined:
        _logger.info(
            f"'**' left as exported in {cells.formula_fqname}, so a "
            "fractional exponent there still compiles to complex "
            "arithmetic: " + "; ".join(rewriter.declined[:3])
        )
    if not rewriter.rewritten:
        return node
    return node.with_changes(body=updated)


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


# ---------------------------------------------------------------------------
# math.exp, math.log and math.pow on C numbers (spec "use_libm")
#
# A formula reaches the math module through a reference, as
# ``self.math.exp(x)``, which Cython compiles to a Python attribute lookup
# and a Python call even when ``x`` is a C double.  Where the spec sets
# "use_libm", such a call whose arguments are all provably C numbers --
# the same proof as for ``**`` above -- is rewritten to the wrapper of the
# C library function in ``_mx_sys.pxd``.  The wrapper returns the C result
# where the inputs and the result are finite, and calls the Python
# function otherwise, so overflow, poles and domain errors still raise
# what Python raises.

MATH_FUNCS = {
    # math function: (_mx_sys wrapper, number of arguments)
    "exp": ("_mx_exp", 1),
    "log": ("_mx_log", 1),
    "pow": ("_mx_math_pow", 2),
}
"""dict: The math functions rewritten under the spec's ``"use_libm"``."""


def build_math_refs(module_infos: Mapping[str, object]) -> Dict[str, set]:
    """Collect the references that hold the ``math`` module.

    Parameters
    ----------
    module_infos : mapping of str to ModuleInfo
        The parsed modules, keyed by module fqname.

    Returns
    -------
    dict
        Class fqname to the set of names of its references whose
        sampled value is the ``math`` module.
    """
    result: Dict[str, set] = {}
    for info in module_infos.values():
        for cls_info in info.classes.values():
            names = {name for name, ref in cls_info.refs.items()
                     if getattr(ref, "module_name", "") == "math"}
            if names:
                result[cls_info.fqname] = names
    return result


class _MathOperandKind(OperandKind):
    """:class:`OperandKind` that also knows a math call it rewrites: one
    is a C double, so ``exp(log(x))`` is rewritten inside out."""

    def __init__(self, *args, math_refs=(), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._math_refs = math_refs

    def math_call_kinds(self, expr: cst.Call):
        """The kinds of the arguments of ``expr`` if it is a math call
        that can be rewritten, else None."""
        chain = _extract_self_chain(expr.func)
        if not (isinstance(chain, list) and len(chain) == 2
                and chain[0] in self._math_refs and chain[1] in MATH_FUNCS):
            return None
        if (len(expr.args) != MATH_FUNCS[chain[1]][1]
                or any(a.keyword is not None or a.star for a in expr.args)):
            return None
        kinds = [self.kind(a.value) for a in expr.args]
        return None if KIND_UNKNOWN in kinds else kinds

    def _call_kind(self, expr: cst.Call) -> str:
        if self.math_call_kinds(expr) is not None:
            return KIND_FLOAT
        return super()._call_kind(expr)


class _MathCallRewriter(cst.CSTTransformer):
    """Replace ``self.<math ref>.<exp|log|pow>(...)`` on C numbers."""

    def __init__(self, operand_kind: _MathOperandKind, math_refs) -> None:
        super().__init__()
        self._kind = operand_kind
        self._math_refs = math_refs
        self._opaque_depth = 0
        self.rewritten = 0
        self.declined: List[str] = []

    def on_visit(self, node: cst.CSTNode) -> bool:
        if isinstance(node, _OPAQUE_SCOPES):
            self._opaque_depth += 1
        return super().on_visit(node)

    def on_leave(self, original_node, updated_node):
        if isinstance(original_node, _OPAQUE_SCOPES):
            self._opaque_depth -= 1
        return super().on_leave(original_node, updated_node)

    def leave_Call(self, original_node, updated_node):
        chain = _extract_self_chain(original_node.func)
        if not (isinstance(chain, list) and len(chain) == 2
                and chain[0] in self._math_refs and chain[1] in MATH_FUNCS):
            return updated_node
        wrapper = MATH_FUNCS[chain[1]][0]
        # None for math.log(x, base), keyword or starred arguments, or an
        # argument that is not provably a C number
        kinds = self._kind.math_call_kinds(original_node)
        if self._opaque_depth or kinds is None:
            # (inside a comprehension, lambda or nested def, a name may
            # be rebound)
            self.declined.append(
                cst.Module([]).code_for_node(original_node).strip())
            return updated_node
        self.rewritten += 1
        return cst.Call(
            func=cst.Attribute(
                value=cst.Name(MX_SYS_MOD), attr=cst.Name(wrapper)),
            args=[
                cst.Arg(value=_PowerRewriter._as_double(a.value, kind))
                for a, kind in zip(updated_node.args, kinds)
            ],
            lpar=updated_node.lpar,
            rpar=updated_node.rpar,
        )


def rewrite_math_calls(
    node: cst.FunctionDef,
    resolver: Optional[CellsResolver],
    cells_kinds: Mapping[str, str],
    ref_kinds: Mapping[str, Mapping[str, str]],
    math_refs: Optional[Mapping[str, set]],
    cls_fqname: str,
    cells,
) -> cst.FunctionDef:
    """Rewrite the ``math.exp``/``log``/``pow`` calls on C numbers.

    Parameters are as for :func:`rewrite_powers`, plus ``math_refs``
    from :func:`build_math_refs`; when it is None (the spec does not set
    ``"use_libm"``) or the class has no reference to the ``math``
    module, the method is returned unchanged.  A call is rewritten when
    it is ``self.<ref>.<name>(...)`` with ``<ref>`` holding the ``math``
    module, ``<name>`` in :data:`MATH_FUNCS`, the function's number of
    positional arguments, and every argument provably a C number; the
    calls left alone are logged at INFO.
    """
    if resolver is None or not math_refs or cls_fqname not in math_refs:
        return node
    rewriter = _MathCallRewriter(
        _MathOperandKind(resolver, cells_kinds, ref_kinds, cls_fqname,
                         _param_kinds(cells), math_refs=math_refs[cls_fqname]),
        math_refs[cls_fqname],
    )
    updated = node.body.visit(rewriter)
    if rewriter.declined:
        _logger.info(
            f"math call left as exported in {cells.formula_fqname}, as its "
            "arguments are not all provably C numbers: "
            + "; ".join(rewriter.declined[:3])
        )
    if not rewriter.rewritten:
        return node
    return node.with_changes(body=updated)
