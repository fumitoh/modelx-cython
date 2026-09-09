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

    Returns
    -------
    libcst.FunctionDef
        The method, with each rewritable ``**`` replaced by a call to
        ``_mx_sys._mx_pow``.  Every power left alone that could still be
        on Cython's complex path is logged at INFO.
    """
    if resolver is None:
        return node
    param_kinds = {}
    if cells.has_args() and cells.has_typeinfo():
        for param in cells.params:
            param_kinds[param] = kind_of_type_expr(
                cells.get_argtype_expr(param, c_style=True)
            )
    rewriter = _PowerRewriter(
        OperandKind(resolver, cells_kinds, ref_kinds, cls_fqname, param_kinds)
    )
    updated = node.body.visit(rewriter)
    if rewriter.declined:
        _logger.info(
            f"'**' left as exported in {cells.fqname}, so a fractional "
            "exponent there still compiles to complex arithmetic: "
            + "; ".join(rewriter.declined[:3])
        )
    if not rewriter.rewritten:
        return node
    return node.with_changes(body=updated)
