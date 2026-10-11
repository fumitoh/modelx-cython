# Copyright (c) 2023-2025 Fumito Hamamura <fumito.ham@gmail.com>

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

"""Lexical analysis of exported modelx modules using libcst.

This module parses the source code of the modules that make up an
exported modelx model and records their lexical (static) structure:
which classes are space classes (names prefixed ``_c_``), which methods
of those classes are cells, which names are refs assigned in
``_mx_assign_refs``, and which child spaces each space creates in its
``__init__``.

:class:`ModuleVisitor` performs the traversal and stores the results as
:class:`LexicalCellsInfo` and :class:`LexicalRefInfo` objects, which the
builder later combines with runtime trace information to produce the
typing information used by the Cython transformer.
"""

import sys
from abc import ABC, abstractmethod
from typing import Union, Sequence, Mapping

try:
    from types import NoneType
except ImportError:  # Python -3.9
    NoneType = type(None)

from functools import cached_property
import libcst as cst
import libcst.matchers as m
from libcst.metadata import ParentNodeProvider, ScopeProvider, GlobalScope, ClassScope

from modelx_cython.consts import (
    FORMULA_PREF,
    GLOBAL_PREF,
    MODULE_PREF,
    SPACE_PREF,
    MX_SELF,
    MX_SPACE_MOD,
    MX_ASSIGN_REFS,
    MX_LOCK,
    is_user_defined,
)


class LexicalBaseMemberInfo:
    """Base class for lexical information on a space class member.

    Identifies a member (a cells method or a ref) of a space class by
    the module, class, and member name found in the source code.

    Attributes
    ----------
    module : str
        Name of the module the member is defined in.
    cls : str
        Name of the space class the member belongs to.
    name : str
        Name of the member itself.
    """

    module: str
    cls: str
    name: str

    def __init__(self, module, cls, name):
        """Store the module, class, and member names."""
        self.module: str = module
        self.cls: str = cls
        self.name: str = name

    @cached_property
    @abstractmethod
    def fqname(self):
        """Fully-qualified name of the member.

        Abstract cached property; subclasses build the dotted name from
        ``module``, ``cls``, and ``name``.
        """
        pass


class LexicalCellsInfo(LexicalBaseMemberInfo):
    """Lexical information on a cells method of a space class.

    Attributes
    ----------
    params : Sequence[str]
        Parameter names as collected from the formula signature
        (positional parameters, excluding ``self``).
    params_with_defaults : Sequence[str]
        Names of the parameters in ``params`` that have default
        values in the formula signature.
    param_defaults : Mapping[str, cst.BaseExpression]
        The default value expression of each parameter in
        ``params_with_defaults``, as parsed.
    """

    params: Sequence[str]
    params_with_defaults: Sequence[str]
    param_defaults: Mapping[str, cst.BaseExpression]

    def __init__(self, module, cls, name, params, params_with_defaults=(),
                 param_defaults=None) -> None:
        """Store identifying names and the parameter name lists."""
        super().__init__(module, cls, name)
        self.params: Sequence[str] = params
        self.params_with_defaults: Sequence[str] = params_with_defaults
        self.param_defaults: Mapping[str, cst.BaseExpression] = (
            param_defaults or {})

    def is_special(self):
        """Return whether the cells name is a special (dunder) name.

        Returns
        -------
        bool
            ``True`` if ``name`` both starts and ends with ``"__"``
            (e.g. ``__call__``), ``False`` otherwise.
        """
        return self.name[:2] == self.name[-2:] == "__"

    @cached_property
    def fqname(self):
        """Fully-qualified name of the cells' formula method.

        Returns ``"<module>.<cls>._f_<name>"``, where the ``_f_``
        (:data:`FORMULA_PREF`) prefix is omitted for special (dunder)
        names such as ``__call__``.
        """
        pref = "" if self.is_special() else FORMULA_PREF
        result = self.module + "." + self.cls + "." + pref + self.name
        return result

    @cached_property
    def method_fqname(self):
        """Fully-qualified name of the cells' public method.

        Returns ``"<module>.<cls>.<name>"``.  For an uncached cells,
        which has no ``_f_`` method, this is the method that holds the
        formula and that the tracer records.
        """
        return self.module + "." + self.cls + "." + self.name


class LexicalRefInfo(LexicalBaseMemberInfo):
    """Lexical information on a ref assigned in ``_mx_assign_refs``."""

    @cached_property
    def fqname(self):
        """Fully-qualified name: ``"<module>.<cls>.<name>"``."""
        return self.module + "." + self.cls + "." + self.name


class ParentScopeAddin:
    """Mixin adding parent-node and scope queries to a libcst visitor.

    Relies on the host visitor's ``get_metadata`` with
    :class:`ParentNodeProvider` and :class:`ScopeProvider` resolved.
    """

    def get_parent(self, node, level=0):
        """Return the ancestor of ``node`` that is ``level`` parents up.

        ``level=0`` returns ``node`` itself; each increment walks one
        step up via :class:`ParentNodeProvider`.
        """
        while level:
            node = self.get_metadata(ParentNodeProvider, node)
            level -= 1
        return node

    def _get_scope(self, node, level=0):
        """Return the scope of the ancestor ``level`` parents above
        ``node``."""
        return self.get_metadata(ScopeProvider, self.get_parent(node, level=level))

    def is_space_scope(self, node, level=0):
        """Return whether the node's scope is a space class body.

        Checks the scope of the ancestor ``level`` parents above
        ``node``.

        Returns
        -------
        bool
            ``True`` if that scope is a :class:`ClassScope` whose name
            starts with ``_c_`` (:data:`SPACE_PREF`) and whose parent
            scope is the module's :class:`GlobalScope`, i.e. the body
            of a top-level space class.
        """
        scope = self._get_scope(node, level)
        return bool(
            isinstance(scope, ClassScope)
            and scope.name[: len(SPACE_PREF)] == SPACE_PREF
            and isinstance(scope.parent, GlobalScope)
        )


class ModuleVisitor(m.MatcherDecoratableVisitor, ParentScopeAddin):
    """Collect the lexical structure of one exported modelx module.

    Parses ``source`` with libcst and visits it immediately on
    construction, so all attributes below are fully populated when
    ``__init__`` returns.  Cells, refs, and child spaces are collected
    only from top-level space classes (names prefixed ``_c_``), while
    the ``classes`` list records every ``_c_``-prefixed class found
    anywhere in the module.

    Attributes
    ----------
    module : str
        Name of the module being parsed.
    source : str
        Source code of the module.
    cells_info : dict
        Maps each space class name to a dict mapping cells names to
        :class:`LexicalCellsInfo` objects.
    ref_info : dict
        Maps each space class name to a dict mapping ref names to
        :class:`LexicalRefInfo` objects.
    classes : list of str
        Names of all classes prefixed ``_c_`` found in the module.
    spaces : dict
        Maps each space class name to the list of child space names
        assigned to ``self`` in its ``__init__``.
    cimports : list
        Always left empty by this visitor.
    formula_defs : dict
        Maps each space class name to the set of cells names for
        which a ``_f_<name>`` formula method exists in the source.
        Cells that modelx exports as uncached have no ``_f_`` method.
    closure_funcs : dict
        Maps each space class name to the set of cells method names
        whose bodies contain a construct Cython compiles as a closure
        (nested function, lambda, generator expression or yield).
    generator_funcs : dict
        Maps each space class name to the set of ``_f_`` method names
        containing a ``yield``.
    param_rebinds : dict
        Maps each space class name to a dict mapping the name of each
        cells method (``_f_<name>`` or the public ``<name>``) whose body
        binds one of its parameters again to a dict mapping that
        parameter to what the body binds to it, as collected by
        :func:`collect_param_rebinds`.
    locked_classes : set of str
        Names of the classes whose ``__init__`` assigns ``self._mx_lock``:
        the spaces that modelx exported with ``locked_spaces``, whose
        cells run their formulas under the model lock.  The generated
        model class assigns the lock too, so it is listed when its module
        is parsed.
    kwarg_called_names : set of str
        Names of functions/methods called with keyword arguments
        anywhere in the module, collected after the visit by
        :meth:`_collect_kwarg_called_names`.
    attr_called_names : set of str
        Names of the methods called anywhere in the module as
        ``<expr>.<name>(...)``, collected with ``kwarg_called_names``.
    wrapper : cst.metadata.MetadataWrapper
        Metadata wrapper around the parsed module.
    """

    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(self, module, source):
        """Parse ``source`` and visit it to collect member info."""
        super().__init__()
        self.module = module
        self.source = source
        self.cells_info = {}
        self.ref_info = {}  # {class_name: {name: CombinedRefInfo}}
        self.classes = []
        self.spaces = {}  # Parent class name to list of child space names
        self.cimports = []
        self.formula_defs = {}  # {class_name: set of cells names with _f_ defs}
        self.closure_funcs = {}  # {class_name: set of method names containing closures}
        self.generator_funcs = {}  # {class_name: set of method names containing yield}
        self.param_rebinds = {}  # {class_name: {method name: {param: [binding]}}}
        self.locked_classes = set()  # classes whose __init__ assigns self._mx_lock
        self.wrapper = cst.metadata.MetadataWrapper(cst.parse_module(source))
        self.wrapper.visit(self)
        self.attr_called_names = set()
        self.kwarg_called_names = self._collect_kwarg_called_names()

    def _collect_kwarg_called_names(self):
        """Names of functions/methods called with keyword arguments anywhere
        in the module. Cython compiles C-level calls as positional-only, so
        cells called with keyword arguments must stay plain Python methods.

        Also fills ``attr_called_names`` with the names of all methods
        called as attributes."""
        names = set()
        for call in m.findall(self.wrapper.module, m.Call()):
            func = call.func
            if isinstance(func, cst.Attribute):
                self.attr_called_names.add(func.attr.value)
            if any(a.keyword is not None or a.star == "**" for a in call.args):
                if isinstance(func, cst.Attribute):
                    names.add(func.attr.value)
                elif isinstance(func, cst.Name):
                    names.add(func.value)
        return names

    _closure_matcher = (
        m.FunctionDef() | m.Lambda() | m.GeneratorExp() | m.Yield())

    def _has_closure(self, funcdef: cst.FunctionDef) -> bool:
        """True if the function body contains a construct that Cython
        compiles as a closure (nested function, lambda, generator
        expression or yield), which is not supported inside cpdef
        (ccall) functions. cdef (cfunc) functions support closures."""
        return bool(m.findall(funcdef.body, self._closure_matcher))

    def _is_generator(self, funcdef: cst.FunctionDef) -> bool:
        """True if the function body contains a yield, which is not
        supported inside either cdef or cpdef functions. A yield inside
        a nested def is a rare false positive; the resulting fallback to
        a plain Python method is safe."""
        return bool(m.findall(funcdef.body, m.Yield()))

    @m.leave(m.ClassDef())
    def collect_classes(self, original_node):
        """Record every class whose name starts with ``_c_`` in
        ``self.classes``."""
        name = original_node.name.value
        if name[:len(SPACE_PREF)] == SPACE_PREF:
            self.classes.append(name)

    @m.call_if_inside(m.ClassDef())
    @m.call_if_inside(m.FunctionDef(name=cst.Name("__init__")))
    @m.leave(m.SimpleStatementLine())
    def collect_space_info(self, original_node):
        """Record child spaces from ``self.<name>`` assignments in a
        space class's ``__init__``, for names without a leading
        underscore, into ``self.spaces``; and the class into
        ``self.locked_classes`` when it assigns ``self._mx_lock``."""
        if self.is_space_scope(original_node, level=2):
            # SimpleStatement in IndentedBlock in FunctionDef in IndentedBlock in ClassDef
            node = original_node

            # Retrieve class node
            cls_node = original_node
            for _ in range(4):
                cls_node = self.get_metadata(ParentNodeProvider, cls_node)
            cls_name = cst.ensure_type(cls_node, cst.ClassDef).name.value

            try:
                target = cst.ensure_type(node.body[0], cst.Assign).targets[0].target
            except Exception:
                return

            if target.value.value == MX_SELF and target.attr.value == MX_LOCK:
                self.locked_classes.add(cls_name)
                return

            # Assuming all member assignments in __init__ to names without prefix "_" are child spaces
            # TODO: Rewrite to a robuster condition. Make modelx.export output child space list
            if (target.value.value == MX_SELF and is_user_defined(target.attr.value)):
                self.spaces.setdefault(cls_name, []).append(target.attr.value)

    @m.call_if_inside(m.ClassDef())
    @m.call_if_inside(m.FunctionDef(name=cst.Name(MX_ASSIGN_REFS)))
    @m.leave(m.SimpleStatementLine())
    def collect_refs_info(self, original_node):
        """Record a :class:`LexicalRefInfo` in ``self.ref_info`` for
        each attribute assignment in a space class's
        ``_mx_assign_refs`` method; non-assignments are ignored."""

        if self.is_space_scope(original_node, level=2):
            # SimpleStatement in IndentedBlock in FunctionDef in IndentedBlock in ClassDef

            # Retrieve class node
            cls_node = original_node
            for _ in range(4):
                cls_node = self.get_metadata(ParentNodeProvider, cls_node)
            cls_name = cst.ensure_type(cls_node, cst.ClassDef).name.value

            try:
                name = cst.ensure_type(
                    cst.ensure_type(original_node.body[0], cst.Assign).targets[0],
                    cst.AssignTarget,
                ).target.attr.value
            except Exception:  # igonore other than assignments, such as 'pass'
                return

            self.ref_info.setdefault(cls_name, {})[name] = LexicalRefInfo(
                self.module,
                cls_name,
                name
            )

    @m.call_if_inside(m.ClassDef())
    @m.visit(m.FunctionDef())
    def collect_methods(self, original_node):
        """Record a :class:`LexicalCellsInfo` in ``self.cells_info``
        for each cells method of a space class, skipping ``_f_`` and
        ``_mx_`` prefixed methods and any method whose name starts
        with ``__`` other than ``__call__``.  Also records, for each
        ``_f_`` method, the corresponding cells name in
        ``formula_defs`` (and the ``_f_``-prefixed method name in
        ``generator_funcs`` when it contains ``yield``), cells whose
        bodies contain closures in ``closure_funcs``, parameters
        with default values in the cells' ``params_with_defaults``
        and ``param_defaults``, and, for ``_f_`` and cells methods, the
        parameters their bodies bind again in ``param_rebinds``."""

        if self.is_space_scope(original_node):
            cls_name = cst.ensure_type(
                self.get_parent(original_node, level=2),
                cst.ClassDef,
            ).name.value

            if original_node.name.value[: len(FORMULA_PREF)] == FORMULA_PREF:
                # _f_ methods
                name = original_node.name.value
                self.formula_defs.setdefault(cls_name, set()).add(
                    name[len(FORMULA_PREF):])
                if self._is_generator(original_node):
                    self.generator_funcs.setdefault(cls_name, set()).add(name)
                self._collect_rebinds(cls_name, original_node)
            elif original_node.name.value[: len(GLOBAL_PREF)] == GLOBAL_PREF:
                # _mx_ methods
                pass
            elif (
                    original_node.name.value[:2] == "__"
                    and original_node.name.value != "__call__"
            ):
                # Special methods
                pass
            else:
                # cells
                name = original_node.name.value
                param_nodes = [
                    p
                    for p in original_node.params.params
                             + original_node.params.posonly_params
                    if p.name.value != MX_SELF
                ]
                params = [p.name.value for p in param_nodes]
                params_with_defaults = [
                    p.name.value for p in param_nodes if p.default is not None
                ]
                if self._has_closure(original_node):
                    self.closure_funcs.setdefault(cls_name, set()).add(name)
                self._collect_rebinds(cls_name, original_node)

                ci = LexicalCellsInfo(
                    module=self.module,
                    cls=cls_name,
                    name=name,
                    params=params,
                    params_with_defaults=params_with_defaults,
                    param_defaults={p.name.value: p.default
                                    for p in param_nodes
                                    if p.default is not None},
                )
                self.cells_info.setdefault(cls_name, {})[name]  = ci

        return False

    def _collect_rebinds(self, cls_name, funcdef: cst.FunctionDef):
        """Record in ``param_rebinds`` what the body of ``funcdef``
        binds to its parameters, if anything."""
        params = [p.name.value
                  for p in funcdef.params.params + funcdef.params.posonly_params
                  if p.name.value != MX_SELF]
        rebinds = collect_param_rebinds(funcdef, params)
        if rebinds:
            self.param_rebinds.setdefault(cls_name, {})[
                funcdef.name.value] = rebinds


# ---------------------------------------------------------------------------
# Parameters a formula binds again
#
# A cells parameter is declared with the C type of the values the sample
# passed to it.  If the formula then binds the parameter to a value of
# another type -- ``rate = rate / 100`` for an integer ``rate`` --
# Cython rejects the formula ("Cannot assign type 'double' to 'long
# long'"), or, for ``t /= 2``, silently truncates the result.  The
# builder therefore types such a parameter ``object`` in the formula
# unless every value bound to it is provably of its C type (see
# :func:`modelx_cython.powers.demote_rebound_params`); this collects the
# bindings.

_AUG_OPS = {
    cst.AddAssign: cst.Add,
    cst.SubtractAssign: cst.Subtract,
    cst.MultiplyAssign: cst.Multiply,
    cst.DivideAssign: cst.Divide,
    cst.FloorDivideAssign: cst.FloorDivide,
    cst.ModuloAssign: cst.Modulo,
    cst.PowerAssign: cst.Power,
}


class _ParamRebinds(cst.CSTVisitor):
    """Collect the bindings of a function body to the names in
    ``params``.  See :func:`collect_param_rebinds`."""

    def __init__(self, params):
        super().__init__()
        self.params = frozenset(params)
        self.rebinds = {}
        self._nested = 0    # inside a nested def, class or lambda
        self._comp = 0      # inside a comprehension

    def _bind(self, name, value):
        if name in self.params:
            self.rebinds.setdefault(name, []).append(value)

    def _bind_target(self, target, value, what):
        """Record what ``target`` binds: ``value`` for a name bound
        to a value the formula's own scope evaluates, else ``what``."""
        if self._nested:
            return  # the name is local to the nested scope
        if isinstance(target, cst.Name):
            if value is None or self._comp:
                value = what
            self._bind(target.value, value)
        elif isinstance(target, (cst.Tuple, cst.List)):
            for element in target.elements:
                self._bind_target(element.value, None, "unpacking")
        elif isinstance(target, cst.StarredElement):
            self._bind_target(target.value, None, "unpacking")
        # an attribute or a subscript binds no name

    # nested scopes: their own bindings do not rebind the parameters

    def visit_FunctionDef(self, node):
        if not self._nested:
            self._bind(node.name.value, "a nested def")
        self._nested += 1

    def leave_FunctionDef(self, original_node):
        self._nested -= 1

    def visit_ClassDef(self, node):
        if not self._nested:
            self._bind(node.name.value, "a nested class")
        self._nested += 1

    def leave_ClassDef(self, original_node):
        self._nested -= 1

    def visit_Lambda(self, node):
        self._nested += 1

    def leave_Lambda(self, original_node):
        self._nested -= 1

    def _enter_comp(self, node):
        self._comp += 1

    def _leave_comp(self, original_node):
        self._comp -= 1

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = (
        _enter_comp)
    leave_ListComp = leave_SetComp = leave_DictComp = leave_GeneratorExp = (
        _leave_comp)

    # bindings

    def visit_Assign(self, node):
        for target in node.targets:
            self._bind_target(target.target, node.value, "an assignment")

    def visit_AnnAssign(self, node):
        self._bind_target(node.target, None, "an annotated assignment")

    def visit_AugAssign(self, node):
        op = _AUG_OPS.get(type(node.operator))
        value = None
        if op is not None and isinstance(node.target, cst.Name):
            value = cst.BinaryOperation(
                left=cst.Name(node.target.value), operator=op(),
                right=node.value)
        self._bind_target(node.target, value, "an augmented assignment")

    def visit_NamedExpr(self, node):
        self._bind_target(node.target, node.value, "an assignment expression")

    def visit_For(self, node):
        self._bind_target(node.target, None, "a for loop")

    def visit_CompFor(self, node):
        self._bind_target(node.target, None, "a comprehension")

    def visit_WithItem(self, node):
        if node.asname is not None:
            self._bind_target(node.asname.name, None, "a with statement")

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self._bind_target(node.name.name, None, "an except clause")

    def visit_ExceptStarHandler(self, node):
        if node.name is not None:
            self._bind_target(node.name.name, None, "an except clause")

    def visit_ImportAlias(self, node):
        if node.asname is not None:
            target = node.asname.name
        else:
            target = node.name
            while isinstance(target, cst.Attribute):
                target = target.value
        self._bind_target(target, None, "an import")

    def visit_Del(self, node):
        self._bind_target(node.target, None, "a del statement")

    def visit_Global(self, node):
        for item in node.names:
            self._bind(item.name.value, "a global statement")

    def visit_Nonlocal(self, node):
        for item in node.names:
            self._bind(item.name.value, "a nonlocal statement")

    def visit_MatchAs(self, node):
        if node.name is not None:
            self._bind_target(node.name, None, "a match pattern")

    def visit_MatchStar(self, node):
        if node.name is not None:
            self._bind_target(node.name, None, "a match pattern")

    def visit_MatchMapping(self, node):
        if node.rest is not None:
            self._bind_target(node.rest, None, "a match pattern")


def collect_param_rebinds(funcdef: cst.FunctionDef, params):
    """What the body of ``funcdef`` binds to the names in ``params``.

    Parameters
    ----------
    funcdef : libcst.FunctionDef
        A formula method.
    params : iterable of str
        Its parameters, without ``self``.

    Returns
    -------
    dict
        Each parameter the body binds again, mapped to a list with one
        item per binding: the value expression for a plain or augmented
        assignment, or an assignment expression, in the formula's own
        scope (``t += 1`` gives ``t + 1``), or else a phrase naming the
        kind of binding (``"a for loop"``, ``"unpacking"``, ...), whose
        value is not known.  A binding inside a comprehension is never
        known; one inside a nested def, class or lambda does not bind
        the parameter, unless it is declared ``nonlocal`` there.
    """
    visitor = _ParamRebinds(params)
    funcdef.body.visit(visitor)
    return visitor.rebinds



# ---------------------------------------------------------------------------
# Names that cannot be compiled
#
# A model name reaches the generated code in these positions: a cells
# name is a cpdef method of the .pxd and a member of the C vtable struct;
# a cells parameter is a parameter of the .pxd declarations; a reference,
# a space parameter and a child space are cdef attributes of the .pxd and
# members of the C object struct.  Local variables and the .py sources are
# not affected: Cython reads a .py file with Python's keywords only, and
# it prefixes the C names of locals and parameters.  The names below were
# measured by compiling each candidate in each position, with each C type
# mx2cy declares, with Cython 3.2 and MSVC 14.37.  The C macros are those
# of the standard headers that Python.h and Cython's code include, of
# Python's own headers and of Cython itself; some of the error numbers,
# structmember.h names and Py_ constants were added from the headers
# without a build of each.  The list is not exhaustive: a macro it misses
# is still reported by the C compiler.

CYTHON_KEYWORDS = frozenset([
    "include", "cimport", "cdef", "cpdef", "ctypedef",
    "DEF", "IF", "ELIF", "ELSE",
])
"""frozenset: Cython's reserved words: invalid wherever a name is declared
in a .pxd file."""

C_OBJECT_MACROS = frozenset("""
NULL errno stdin stdout stderr EOF BUFSIZ FILENAME_MAX FOPEN_MAX L_tmpnam
SEEK_CUR SEEK_END SEEK_SET TMP_MAX WEOF EXIT_FAILURE EXIT_SUCCESS
MB_CUR_MAX RAND_MAX
EDOM ERANGE EILSEQ EINVAL ENOMEM ENOENT EACCES EAGAIN EBADF EBUSY EEXIST
EINTR EIO EPERM EPIPE ECHILD EFAULT
CHAR_BIT SCHAR_MIN SCHAR_MAX UCHAR_MAX CHAR_MIN CHAR_MAX MB_LEN_MAX
SHRT_MIN SHRT_MAX USHRT_MAX INT_MIN INT_MAX UINT_MAX LONG_MIN LONG_MAX
ULONG_MAX LLONG_MIN LLONG_MAX ULLONG_MAX
INT8_MIN INT8_MAX INT16_MAX INT32_MAX INT64_MAX UINT8_MAX UINT16_MAX
UINT32_MAX UINT64_MAX SIZE_MAX PTRDIFF_MAX PTRDIFF_MIN INTMAX_MAX
INTMAX_MIN UINTMAX_MAX INTPTR_MAX WCHAR_MAX WCHAR_MIN
INFINITY NAN HUGE_VAL HUGE_VALF HUGE_VALL
M_PI M_E M_LN2 M_LN10 M_LOG2E M_LOG10E M_PI_2 M_PI_4 M_1_PI M_2_PI
M_2_SQRTPI M_SQRT2 M_SQRT1_2
FP_NAN FP_INFINITE FP_ZERO FP_SUBNORMAL FP_NORMAL MATH_ERRNO
MATH_ERREXCEPT math_errhandling
NDEBUG cdecl
DOMAIN SING OVERFLOW UNDERFLOW TLOSS PLOSS
E2BIG ENOSPC ESRCH ENOTDIR EISDIR ENFILE EMFILE ENOTTY EFBIG ESPIPE EROFS
EMLINK EDEADLK ENAMETOOLONG ENOLCK ENOSYS ENOTEMPTY ETIMEDOUT ECONNRESET
EWOULDBLOCK ENXIO ENOEXEC EXDEV ENODEV EDEADLOCK STRUNCATE EADDRINUSE
EADDRNOTAVAIL EAFNOSUPPORT EALREADY EBADMSG ECANCELED ECONNABORTED
ECONNREFUSED EDESTADDRREQ EHOSTUNREACH EIDRM EINPROGRESS EISCONN ELOOP
EMSGSIZE ENETDOWN ENETRESET ENETUNREACH ENOBUFS ENODATA ENOLINK ENOMSG
ENOPROTOOPT ENOSR ENOSTR ENOTCONN ENOTRECOVERABLE ENOTSOCK ENOTSUP
EOPNOTSUPP EOTHER EOVERFLOW EOWNERDEAD EPROTO EPROTONOSUPPORT EPROTOTYPE
ETIME ETXTBSY
PLATFORM COMPILER LONG_BIT WORD_BIT
T_SHORT T_INT T_LONG T_FLOAT T_DOUBLE T_STRING T_OBJECT T_CHAR T_BYTE
T_UBYTE T_USHORT T_UINT T_ULONG T_STRING_INPLACE T_BOOL T_OBJECT_EX
T_LONGLONG T_ULONGLONG T_PYSSIZET T_NONE READONLY READ_RESTRICTED
PY_WRITE_RESTRICTED RESTRICTED PY_AUDIT_READ
Py_True Py_False Py_None Py_NotImplemented Py_Ellipsis
""".split())
"""frozenset: Object-like C macros: a struct member of the name, which
every cells, reference, space parameter and child space is, does not
compile.  Besides those of the standard C headers, these are the
SVID error types of MSVC's ``math.h`` (``DOMAIN``, ``OVERFLOW``, ...),
the POSIX error numbers of ``errno.h``, and macros of Python's
``pyconfig.h`` (``PLATFORM``, ``LONG_BIT``, ...), ``structmember.h``
(``T_INT``, ``READONLY``, ...) and ``object.h`` (``Py_True``, ...)."""

C_FUNCTION_MACROS = frozenset("""
isnan isinf isfinite isnormal signbit fpclassify isgreater isgreaterequal
isless islessequal islessgreater isunordered offsetof likely unlikely
""".split()) | (frozenset({"min", "max"}) if sys.platform == "win32"
                else frozenset())
"""frozenset: Function-like C macros (``min`` and ``max`` only with MSVC):
a cells of the name does not compile where compiled code calls it."""

C_TYPE_WORDS = {
    "long long": frozenset([
        "signed", "unsigned", "long", "short", "char", "int", "double",
        "float", "bint", "void", "complex"]),
    "double": frozenset(["complex"]),
    "bint": frozenset(["complex"]),
}
"""dict: C type of a .pxd declaration -> the names Cython reads as part of
that type when they follow it, as in ``cdef public long long double``.
``int`` is the exception for a parameter, where ``long long int`` is read
as an unnamed parameter of that type, which compiles; and a cells named
``str`` cannot return ``str``."""


def _space_path(module: str, cls: str) -> str:
    """``"Parent.Child"`` for module ``Pkg._m_Parent._mx_classes`` and
    class ``_c_Child``: the space's name in the model."""
    names = [p[len(MODULE_PREF):] for p in module.split(".")[1:]
             if p[:len(MODULE_PREF)] == MODULE_PREF]
    return ".".join(names + [cls[len(SPACE_PREF):]])


class ReservedNameError(ValueError):
    """A name in the model cannot be compiled by Cython or the C compiler."""


def _raise_problems(problems):
    if problems:
        raise ReservedNameError(
            "mx2cy cannot compile these names, which Cython or the C "
            "compiler reserves; rename them in the model:\n  "
            + "\n  ".join(problems))


def check_reserved_names(visitors) -> None:
    """Reject the model if a name in it cannot be compiled, whatever its type.

    Run on the parsed sources before the sample run, so that the error
    comes before the time the run takes.  Checks the cells, parameters,
    references, space parameters (the parameters of ``__call__``) and
    child spaces of the space classes against
    :data:`CYTHON_KEYWORDS`, :data:`C_OBJECT_MACROS` and
    :data:`C_FUNCTION_MACROS`, each where it breaks the build: a cells
    only if it gets a ``cpdef`` (it has no closure and is not called
    with keyword arguments), a parameter only if its cells gets a
    ``cpdef`` or a ``cdef`` formula, and a function-like macro only for
    a cells called as a method somewhere in the model.  The names that
    break only after a C type are checked by :func:`check_type_words`
    once the types are known.

    mx2cy fails rather than renaming the names: a cells, reference or
    space name is the model's interface, reached from outside the model
    by that name, and a parameter name is part of the signature that
    keyword calls use, so a rename would silently change what code using
    the compiled model has to call.

    Parameters
    ----------
    visitors : iterable of ModuleVisitor
        The parsed modules of the model.  Only space modules
        (``_mx_classes``) are checked: the model class stays a Python
        class.

    Raises
    ------
    ReservedNameError
        Listing every offending name.
    """
    visitors = list(visitors)
    kwarg_names = set().union(*(v.kwarg_called_names for v in visitors))
    called_names = set().union(*(v.attr_called_names for v in visitors))
    problems = []
    for visitor in visitors:
        if visitor.module.split(".")[-1] != MX_SPACE_MOD:
            continue

        def check(name, cls, what, macros=True, calls=False):
            if name in CYTHON_KEYWORDS:
                reason = "is a reserved word of Cython"
            elif macros and name in C_OBJECT_MACROS:
                reason = ("is a macro of the C headers the compiled "
                          "module includes")
            elif calls and name in C_FUNCTION_MACROS and name in called_names:
                reason = ("is a function-like macro of the C headers the "
                          "compiled module includes, and it is called")
            else:
                return
            problems.append(
                f"{_space_path(visitor.module, cls)}: {what} {name!r} {reason}")

        for cls, cells in visitor.cells_info.items():
            closures = visitor.closure_funcs.get(cls, ())
            formulas = visitor.formula_defs.get(cls, ())
            generators = visitor.generator_funcs.get(cls, ())
            for name, info in cells.items():
                if info.is_special():
                    for p in info.params:   # __call__: the space parameters
                        check(p, cls, "space parameter")
                    continue
                has_cpdef = name not in closures and name not in kwarg_names
                has_cdef = (name in formulas
                            and FORMULA_PREF + name not in generators)
                if has_cpdef:
                    check(name, cls, "cells", calls=True)
                if has_cpdef or has_cdef:
                    for p in info.params:
                        # C parameters are prefixed: no macro clash
                        check(p, cls, f"parameter of cells '{name}'",
                              macros=False)
        for cls, refs in visitor.ref_info.items():
            for name in refs:
                check(name, cls, "reference")
        for cls, spaces in visitor.spaces.items():
            for name in spaces:
                check(name, cls, "child space")
    _raise_problems(problems)


def check_type_words(module_infos) -> None:
    """Reject the model if a name cannot follow its C type in a .pxd.

    Run once the C types are final, before the sources are rewritten:
    checks each name that :class:`~modelx_cython.transformer.PXDGenerator`
    declares after a C type -- cells after their return types,
    parameters after their types, references after theirs -- against
    :data:`C_TYPE_WORDS`.

    Parameters
    ----------
    module_infos : iterable of ModuleInfo
        The built modules of the model.

    Raises
    ------
    ReservedNameError
        Listing every offending name.
    """
    problems = []
    for info in module_infos:
        for cls_name, cls_info in info.classes.items():
            where = _space_path(info.fqname, cls_name)

            def check(name, ctype, what, param=False):
                bad = C_TYPE_WORDS.get(ctype, frozenset())
                if param:
                    bad = bad - {"int"}
                elif what == "cells" and ctype == "str":
                    bad = frozenset(["str"])
                if name in bad:
                    problems.append(
                        f"{where}: {what} {name!r} declared as {ctype} "
                        "is read by Cython as part of the C type")

            for cells in cls_info.cells.values():
                if cells.is_special():
                    continue
                has_cpdef = not (cells.body_has_closure
                                 or cells.called_with_kwargs)
                has_cdef = (cells.has_formula_def
                            and not cells.formula_is_generator)
                if has_cpdef:
                    check(cells.name, cells.get_rettype_expr(c_style=True),
                          "cells")
                if (has_cpdef or has_cdef) and cells.has_typeinfo():
                    for p in cells.params:
                        # the public method and the _f_ formula can
                        # differ: see CombinedCellsInfo.arg_type
                        ctypes = set()
                        if has_cpdef:
                            ctypes.add(cells.get_argtype_expr(p, c_style=True))
                        if has_cdef:
                            ctypes.add(cells.get_argtype_expr(
                                p, c_style=True, formula=True))
                        for ctype in sorted(ctypes):
                            check(p, ctype,
                                  f"parameter of cells '{cells.name}'",
                                  param=True)
            for ref in cls_info.refs.values():
                if ref.name not in cls_info.spaces:
                    check(ref.name, ref.get_type_expr(c_style=True),
                          "reference")
    _raise_problems(problems)
