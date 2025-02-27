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
    module: str
    cls: str
    name: str

    def __init__(self, module, cls, name):
        self.module: str = module
        self.cls: str = cls
        self.name: str = name

    @cached_property
    @abstractmethod
    def fqname(self):
        pass


class LexicalCellsInfo(LexicalBaseMemberInfo):

    params: Sequence[str]

    def __init__(self, module, cls, name, params) -> None:
        super().__init__(module, cls, name)
        self.params: Sequence[str] = params

    def is_special(self):
        return self.name[:2] == self.name[-2:] == "__"

    @cached_property
    def fqname(self):
        pref = "" if self.is_special() else FORMULA_PREF
        result = self.module + "." + self.cls + "." + pref + self.name
        return result


class LexicalRefInfo(LexicalBaseMemberInfo):

    @cached_property
    def fqname(self):
        return self.module + "." + self.cls + "." + self.name


class ParentScopeAddin:

    def get_parent(self, node, level=0):
        while level:
            node = self.get_metadata(ParentNodeProvider, node)
            level -= 1
        return node

    def _get_scope(self, node, level=0):
        return self.get_metadata(ScopeProvider, self.get_parent(node, level=level))

    def is_space_scope(self, node, level=0):
        scope = self._get_scope(node, level)
        return bool(
            isinstance(scope, ClassScope)
            and scope.name[: len(SPACE_PREF)] == SPACE_PREF
            and isinstance(scope.parent, GlobalScope)
        )


class ModuleVisitor(m.MatcherDecoratableVisitor, ParentScopeAddin):
    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(self, module, source):
        super().__init__()
        self.module = module
        self.source = source
        self.cells_info = {}
        self.ref_info = {}  # {class_name: {name: CombinedRefInfo}}
        self.classes = []
        self.spaces = {}  # Parent class name to list of child space names
        self.cimports = []
        self.wrapper = cst.metadata.MetadataWrapper(cst.parse_module(source))
        self.wrapper.visit(self)

    @m.leave(m.ClassDef())
    def collect_classes(self, original_node):
        name = original_node.name.value
        if name[:len(SPACE_PREF)] == SPACE_PREF:
            self.classes.append(name)

    @m.call_if_inside(m.ClassDef())
    @m.call_if_inside(m.FunctionDef(name=cst.Name("__init__")))
    @m.leave(m.SimpleStatementLine())
    def collect_space_info(self, original_node):
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

        if self.is_space_scope(original_node):
            cls_name = cst.ensure_type(
                self.get_parent(original_node, level=2),
                cst.ClassDef,
            ).name.value

            if original_node.name.value[: len(FORMULA_PREF)] == FORMULA_PREF:
                # _f_ methods
                pass
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
                params = [
                    p.name.value
                    for p in original_node.params.params
                             + original_node.params.posonly_params
                    if p.name.value != MX_SELF
                ]

                ci = LexicalCellsInfo(
                    module=self.module,
                    cls=cls_name,
                    name=name,
                    params=params
                )
                self.cells_info.setdefault(cls_name, {})[name]  = ci

        return False

