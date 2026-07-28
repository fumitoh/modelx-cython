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
    SPACE_PREF,
    MX_SELF,
    MX_ASSIGN_REFS,
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
    """

    params: Sequence[str]
    params_with_defaults: Sequence[str]

    def __init__(self, module, cls, name, params, params_with_defaults=()) -> None:
        """Store identifying names and the parameter name lists."""
        super().__init__(module, cls, name)
        self.params: Sequence[str] = params
        self.params_with_defaults: Sequence[str] = params_with_defaults

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
        self.wrapper = cst.metadata.MetadataWrapper(cst.parse_module(source))
        self.wrapper.visit(self)
        self.kwarg_called_names = self._collect_kwarg_called_names()

    def _collect_kwarg_called_names(self):
        """Names of functions/methods called with keyword arguments anywhere
        in the module. Cython compiles C-level calls as positional-only, so
        cells called with keyword arguments must stay plain Python methods."""
        names = set()
        for call in m.findall(self.wrapper.module, m.Call()):
            if any(a.keyword is not None or a.star == "**" for a in call.args):
                func = call.func
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
        underscore, into ``self.spaces``."""
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
        with ``__`` other than ``__call__``."""

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

                ci = LexicalCellsInfo(
                    module=self.module,
                    cls=cls_name,
                    name=name,
                    params=params,
                    params_with_defaults=params_with_defaults
                )
                self.cells_info.setdefault(cls_name, {})[name]  = ci

        return False

