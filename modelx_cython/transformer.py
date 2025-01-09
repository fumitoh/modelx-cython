# Copyright (c) 2023 Fumito Hamamura <fumito.ham@gmail.com>

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

from typing import Union, Sequence, Mapping
try:
    from types import NoneType
except ImportError: # Python -3.9
    NoneType = type(None)
from dataclasses import dataclass
import textwrap
from functools import cached_property
import libcst as cst
from libcst._flatten_sentinel import FlattenSentinel
from libcst._nodes.module import Module
from libcst._nodes.statement import BaseStatement, ClassDef
from libcst._removal_sentinel import RemovalSentinel
import libcst.matchers as m
from libcst.metadata import ParentNodeProvider, ScopeProvider, GlobalScope, ClassScope

from modelx_cython.config import TranslationSpec
from modelx_cython.tracer import RuntimeCellsInfo, get_type_expr

from modelx_cython.consts import (
    FORMULA_PREF,
    CY_MOD,
    GLOBAL_PREF,
    VAR_PREF,
    HAS_PREF,
    SPACE_PREF,
    MODULE_PREF,
    MX_SELF,
    MX_SYS_MOD,
    MX_SPACE_MOD,
    CY_BOOL_T,
    MX_ASSIGN_REFS,
    MX_COPY_REFS,
    is_user_defined,
)


class LexicalCellsInfo:
    module_name: str
    cls_name: str
    name: str
    params: Sequence[str]

    def __init__(self, module_name, cls_name, name, params, spec) -> None:
        self.module_name: str = module_name
        self.cls_name: str = cls_name
        self.name: str = name
        self.params: Sequence[str] = params
        self.spec = spec

    @property
    def fqname(self):
        pref = "" if self.is_special() else FORMULA_PREF
        return self.module_name + "." + self.cls_name + "." + pref + self.name

    def is_special(self):
        return self.name[:2] == self.name[-2:] == "__"


class CombinedCellsInfo(LexicalCellsInfo):
    _runtime_info: RuntimeCellsInfo

    def __init__(self, cellsinfo, typeinfo) -> None:
        super().__init__(
            cellsinfo.module_name, cellsinfo.cls_name, cellsinfo.name, cellsinfo.params, cellsinfo.spec
        )
        self._runtime_info = typeinfo

    def has_typeinfo(self):
        return bool(self._runtime_info)

    def has_args(self):
        return bool(self.params)

    def get_argtype_expr(self, arg: str, with_module=True, use_double=False) -> str:
        return self._runtime_info.get_argtype_expr(arg, with_module=with_module, use_double=use_double)

    def get_rettype_expr(self, with_module=True, use_double=False):

        ret_t = self.spec.get(TranslationSpec.RET_T)
        if ret_t:
            return ret_t
        elif self.has_typeinfo():
            return self._runtime_info.get_rettype_expr(with_module=with_module, use_double=use_double)
        else:
            return "object"

    def is_arrayable(self, sizes):
        if self.has_typeinfo():
            return self._runtime_info.is_arrayable(sizes)
        else:
            return False

    def get_decltype_expr(self, sizes: Mapping[str, int], rettype_expr="", with_module=True, use_double=False):
        return self._runtime_info.get_decltype_expr(sizes, rettype_expr=rettype_expr, with_module=with_module, use_double=use_double)


@dataclass
class LexicalRefInfo:
    module_name: str
    cls_name: str
    name: str
    type_: type

class SpaceAddin:

    def get_parent(self, node, level=0):
        while level:
            node = self.get_metadata(ParentNodeProvider, node)
            level -= 1
        return node
    
    def get_scope(self, node, level=0):
        return self.get_metadata(ScopeProvider, self.get_parent(node, level=level))
    
    def is_space_scope(self, node, level=0):
        scope = self.get_scope(node, level)
        return bool(
            isinstance(scope, ClassScope)
            and scope.name[: len(SPACE_PREF)] == SPACE_PREF
            and isinstance(scope.parent, GlobalScope)
        )

class SpaceVisitor(m.MatcherDecoratableVisitor, SpaceAddin):
    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(self, module_name, source, spec, cells_info: dict, ref_info: dict):
        super().__init__()
        self.module_name = module_name
        self.source = source
        self.spec = spec
        self.cells_info = {}
        self.ref_info = {}
        self.classes = []
        self.spaces = {}    # Parent class name to list of child space names
        self._rt_cells_info = cells_info
        self._rt_ref_info = ref_info
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
            except Exception:   # igonore other than assignments, such as 'pass'
                return

            rt_info = self._rt_ref_info.get(
                self.module_name + "." + cls_name + "." + name, None
            )
            self.ref_info[self.module_name, cls_name, name] = LexicalRefInfo(
                self.module_name,
                cls_name,
                name,
                rt_info.type_ if rt_info else None,
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
                spec = self.spec.get_spec(self.module_name + "." + cls_name).get("cells", {}).get(name, {})

                ci = LexicalCellsInfo(
                    module_name=self.module_name,
                    cls_name=cls_name,
                    name=name,
                    params=params,
                    spec=spec
                )
                self.cells_info.setdefault(cls_name, {})[name] = CombinedCellsInfo(
                    ci, self._rt_cells_info.get(ci.fqname, None)
                )

        return False


class PXDGenerator:

    pxd_template = textwrap.dedent("""\
    from {package} cimport {MX_SYS_MOD}
    {child_cimports}

    {class_defs}
    """)

    cls_template = textwrap.dedent("""\
    cdef class {class_name}({MX_SYS_MOD}.BaseSpace):

    {private_var_defs}
    {public_var_defs}

        cpdef {MX_COPY_REFS}({class_name} self, object base, object base_root)

    {private_meth_defs}
    {public_meth_defs}
    """)

    def __init__(self, visitor: SpaceVisitor):
        self.module_name = visitor.module_name
        self.classes = visitor.classes
        self.cells_info = visitor.cells_info
        self.ref_info = visitor.ref_info
        self.spaces = visitor.spaces
        self.spec = visitor.spec

    @cached_property
    def package(self) -> str:
        return self.module_name.split(".")[0]

    @cached_property
    def code(self):
        return self.pxd_template.format(
            package=self.package,
            MX_SYS_MOD=MX_SYS_MOD,
            child_cimports=self.child_cimports,
            class_defs=self.class_defs
        )

    @cached_property
    def class_defs(self):
        stmts = []
        for cls in self.classes:
            stmts.append(self.a_class_def(cls))
        return "\n\n".join(stmts)

    def a_class_def(self, name):
        return self.cls_template.format(
            class_name=name,
            MX_SYS_MOD=MX_SYS_MOD,
            private_var_defs=textwrap.indent(self.private_var_defs(name), ' ' * 4),
            public_var_defs=textwrap.indent(self.public_var_defs(name), ' ' * 4),
            MX_COPY_REFS=MX_COPY_REFS,
            private_meth_defs=textwrap.indent(self.private_meth_defs(name), ' ' * 4),
            public_meth_defs=textwrap.indent(self.public_meth_defs(name), ' ' * 4)
        )

    @cached_property
    def child_cimports(self):
        # cimports for child spaces
        stmts = []
        for space in self.spaces:
            parent = ".".join(self.module_name.split(".")[:-1])
            child = MODULE_PREF + space[len(SPACE_PREF):]   # replace _c_ with _m_
            stmts.append(f"from cython.cimports.{parent} import {child}\n")

        return "".join(stmts)

    def private_var_defs(self, cls_name):

        decl_stmts = []
        for cells in self.cells_info[cls_name].values():

            assert cells.module_name == self.module_name
            assert cells.cls_name == cls_name

            if cells.is_special():
                continue

            if cells.has_args():
                if cells.is_arrayable(self.get_arg_sizes(cls_name)):

                    var_name = VAR_PREF + cells.name
                    var_type = cells.get_decltype_expr(self.get_arg_sizes(cls_name), with_module=False, use_double=True)
                    decl_stmts.append(f"cdef {var_type} {var_name}\n")

                    has_name = HAS_PREF + cells.name
                    has_type = cells.get_decltype_expr(
                                self.get_arg_sizes(cls_name),
                                rettype_expr=CY_BOOL_T, with_module=False, use_double=True)
                    decl_stmts.append(f"cdef {has_type} {has_name}\n")

                else:
                    decl_stmts.append(f"cdef dict {VAR_PREF + cells.name}\n")
            else:
                rettype = cells.get_rettype_expr(with_module=False, use_double=True)
                decl_stmts.append(f"cdef {rettype} {VAR_PREF + cells.name}\n")
                decl_stmts.append(f"cdef {CY_BOOL_T} {HAS_PREF + cells.name}\n")

        return "".join(decl_stmts)

    def public_var_defs(self, cls_name):

        decl_stmts = []
        for ref in self.ref_info.values():
            if ref.module_name != self.module_name or ref.cls_name != cls_name:
                continue

            stmt = f"cdef public {get_type_expr(ref.type_, with_module=False, use_double=True)} {ref.name}\n"
            decl_stmts.append(stmt)

        # Declare child spaces
        for space in self.spaces.get(cls_name, []):
            mod_name = MODULE_PREF + cls_name[len(SPACE_PREF):]  # Replace prefix for submodule
            rel_path = mod_name + "." + MX_SPACE_MOD + "." + SPACE_PREF + space

            stmt = f"cdef public {rel_path} {space}\n"
            decl_stmts.append(stmt)

        return "".join(decl_stmts)

    def _add_param_type_hints(
        self, cls_name: str, cells_name: str
    ) -> str:

        cells = self.cells_info[cls_name][cells_name]
        params = [f"{cls_name} {MX_SELF}"] # add self first

        # Add parameter type hints
        if cells and cells.has_typeinfo() and cells.has_args():
            for param in cells.params:
                type_ = cells.get_argtype_expr(param, with_module=False, use_double=True)
                params.append(f"{type_} {param}")
        else:
            for p in cells.params:
                params.append(f"object {p}")

        return ", ".join(params)

    def private_meth_defs(self, cls_name):

        decl_stmts = []
        for cells in self.cells_info[cls_name].values():

            if cells.is_special():
                continue

            if cells and cells.has_typeinfo():
                rettype = cells.get_rettype_expr(with_module=False, use_double=True)
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name
                )
                decl_stmts.append(
                    f"cdef {rettype} {FORMULA_PREF + cells.name}({parameters})\n"
                )
            else:
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name
                )
                decl_stmts.append(
                    f"cdef object {FORMULA_PREF + cells.name}({parameters})\n"
                )

        return "".join(decl_stmts)

    def public_meth_defs(self, cls_name):

        decl_stmts = []
        for cells in self.cells_info[cls_name].values():

            if cells.is_special():
                continue

            if cells and cells.has_typeinfo():
                rettype = cells.get_rettype_expr(with_module=False, use_double=True)
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name
                )
                decl_stmts.append(
                    f"cpdef {rettype} {cells.name}({parameters})\n"
                )
            else:
                parameters = self._add_param_type_hints(
                    cls_name=cls_name, cells_name=cells.name
                )
                decl_stmts.append(
                    f"cpdef object {cells.name}({parameters})\n"
                )

        return "".join(decl_stmts)

    #TODO: Dupolicate def
    def get_arg_sizes(self, cls_name: str) -> Mapping[str, int]:
        space = self.spec.get_spec(self.module_name + "." + cls_name)
        params = space.get("cells_params", {})
        return {k: v["size"] for k, v in params.items() if "size" in v}


class SpaceTransformer(m.MatcherDecoratableTransformer, SpaceAddin):
    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(
        self,
        visitor: SpaceVisitor,
    ) -> None:
        super().__init__()
        self.module_name = visitor.module_name
        self.wrapper = cst.metadata.MetadataWrapper(cst.parse_module(visitor.source))
        self.module = self.wrapper.module
        self.spec = visitor.spec
        self.cells_info = visitor.cells_info
        self.ref_info = visitor.ref_info
        self.spaces = visitor.spaces

    @property
    def package(self) -> str:
        return self.module_name.split(".")[0]

    @property
    def transformed(self):
        return self.wrapper.visit(self)

    def get_arg_sizes(self, cls_name: str) -> Mapping[str, int]:
        space = self.spec.get_spec(self.module_name + "." + cls_name)
        params = space.get("cells_params", {})
        return {k: v["size"] for k, v in params.items() if "size" in v}


    def leave_Module(self, original_node: Module, updated_node: Module) -> Module:

        # cimports for child spaces
        stmts = []
        for space in self.spaces:

            parent = ".".join(self.module_name.split(".")[:-1])
            child = MODULE_PREF + space[len(SPACE_PREF):]   # replace _c_ with _m_
            stmts.append(cst.parse_statement(
                f"from cython.cimports.{parent} import {child}",
                config=updated_node.config_for_parsing,
            ))

        return updated_node.with_changes(
            body=(
                cst.parse_statement(
                    f"from cython.cimports.{self.package} import {MX_SYS_MOD}",
                    config=updated_node.config_for_parsing,
                ),
                *stmts,
                cst.parse_statement(
                    f"import cython as {CY_MOD}", config=updated_node.config_for_parsing
                ),
                *updated_node.body,
            )
        )

    def leave_ClassDef(
        self, original_node: ClassDef, updated_node: ClassDef
    ) -> Union[BaseStatement, FlattenSentinel[BaseStatement], RemovalSentinel]:
        cls_name: str = original_node.name.value
        if cls_name[: len(SPACE_PREF)] == SPACE_PREF and isinstance(
            self.get_metadata(ScopeProvider, original_node), GlobalScope
        ):
            decl_stmts = []
            for cells in self.cells_info[cls_name].values():

                assert cells.module_name == self.module_name
                assert cells.cls_name == cls_name

                if cells.is_special():
                    continue

                if cells.has_args():
                    if cells.is_arrayable(self.get_arg_sizes(cls_name)):
                        decl_stmts.append(
                            cst.parse_statement(
                                VAR_PREF
                                + cells.name
                                + ": "
                                + cells.get_decltype_expr(self.get_arg_sizes(cls_name)),
                                config=self.module.config_for_parsing,
                            )
                        )
                        decl_stmts.append(
                            cst.parse_statement(
                                HAS_PREF
                                + cells.name
                                + ": "
                                + cells.get_decltype_expr(
                                    self.get_arg_sizes(cls_name),
                                    rettype_expr=f"{CY_MOD}.{CY_BOOL_T}",
                                ),
                                config=self.module.config_for_parsing,
                            )
                        )
                    else:
                        decl_stmts.append(
                            cst.parse_statement(
                                VAR_PREF + cells.name + ": dict",
                                config=self.module.config_for_parsing,
                            )
                        )
                else:
                    rettype = cells.get_rettype_expr()
                    decl_stmts.append(
                        cst.parse_statement(
                            VAR_PREF + cells.name + ": " + rettype,
                            config=self.module.config_for_parsing,
                        )
                    )

                    decl_stmts.append(
                        cst.parse_statement(
                            HAS_PREF + cells.name + ": " + CY_MOD + "." + CY_BOOL_T,
                            config=self.module.config_for_parsing,
                        )
                    )

            is_first = True
            for ref in self.ref_info.values():

                if ref.module_name != self.module_name or ref.cls_name != cls_name:
                    continue

                stmt = cst.parse_statement(
                    f"{ref.name}: {get_type_expr(ref.type_)}",
                    config=self.module.config_for_parsing,
                )
                if is_first:
                    stmt = stmt.with_changes(
                        leading_lines=tuple(stmt.leading_lines) + (cst.EmptyLine(),)
                    )
                    is_first = False

                decl_stmts.append(stmt)

            # Declare child spaces
            is_first = True
            for space in self.spaces.get(cls_name, []):

                mod_name = MODULE_PREF + cls_name[len(SPACE_PREF):]  # Replace prefix for submodule
                rel_path = mod_name + "." + MX_SPACE_MOD + "." + SPACE_PREF + space

                stmt = cst.parse_statement(
                    f"{space}: {rel_path}",
                    config=self.module.config_for_parsing,
                )
                if is_first:
                    stmt = stmt.with_changes(
                        leading_lines=tuple(stmt.leading_lines) + (cst.EmptyLine(),)
                    )
                    is_first = False

                decl_stmts.append(stmt)


            decorator = cst.Decorator(
                decorator=cst.Attribute(value=cst.Name(CY_MOD), attr=cst.Name("cclass"))
            )

            if decl_stmts:
                # Add blank lines below classdef
                decl_stmts[0] = decl_stmts[0].with_changes(
                    leading_lines=tuple(decl_stmts[0].leading_lines) + (cst.EmptyLine(),)
                )
                indented_block = cst.ensure_type(
                    updated_node.body, cst.IndentedBlock
                ).with_changes(body=tuple(decl_stmts) + updated_node.body.body)
                return updated_node.with_changes(
                    decorators=(decorator,), body=indented_block
                )
            else:
                return updated_node.with_changes(decorators=(decorator,))
        else:
            return updated_node

    @m.call_if_inside(m.ClassDef())
    @m.call_if_inside(m.FunctionDef(name=cst.Name("__init__")))
    @m.leave(m.SimpleStatementLine())
    def remove_cache_assigns(self, original_node, updated_node):
        funcdef = self.get_parent(original_node, level=2)
        clsdef = self.get_parent(funcdef, level=2)
        if (
            m.matches(funcdef, m.FunctionDef(name=cst.Name("__init__")))
            and m.matches(clsdef, m.ClassDef())
            and clsdef.name.value[: len(SPACE_PREF)] == SPACE_PREF
            and m.matches(original_node, m.SimpleStatementLine(body=(m.Assign(),)))
            and m.matches(
                original_node.body[0],
                m.Assign(
                    targets=(m.AssignTarget(target=m.Attribute(value=m.Name(MX_SELF))),)
                ),
            )
            and (
                original_node.body[0].targets[0].target.attr.value[: len(HAS_PREF)]
                == HAS_PREF
                or original_node.body[0].targets[0].target.attr.value[: len(VAR_PREF)]
                == VAR_PREF
            )
        ):
            return cst.RemoveFromParent()

        return updated_node

    def _add_param_type_hints(
        self, funcdef: cst.FunctionDef, cls_name: str
    ) -> Union[cst.Parameters, NoneType]:
        param_list = list(funcdef.params.params + funcdef.params.posonly_params)[
            1:
        ]  # remove self

        name = funcdef.name.value
        if name[:len(FORMULA_PREF)] == FORMULA_PREF:
            name = name[len(FORMULA_PREF):]

        if (d := self.cells_info.get(cls_name)):
            cells = d.get(name)
        else:
            cells = None

        # Add parameter type hints
        if cells and cells.has_typeinfo() and cells.has_args():
            updated_params = [funcdef.params.params[0]]  # add self first
            for param in param_list:
                param_name = param.name.value
                if cells.get_argtype_expr(param_name):
                    updated_params.append(
                        param.with_changes(
                            annotation=cst.Annotation(
                                annotation=cst.parse_expression(
                                    cells.get_argtype_expr(param_name),
                                    config=self.module.config_for_parsing,
                                )
                            )
                        )
                    )
                else:
                    updated_params.append(param)

            return funcdef.params.with_changes(params=tuple(updated_params))

        return None


    @m.call_if_inside(m.ClassDef())
    @m.call_if_inside(m.FunctionDef(name=cst.Name(MX_COPY_REFS)))
    @m.call_if_inside(m.SimpleStatementLine())
    @m.call_if_inside(m.Assign())
    @m.call_if_inside(m.Attribute())
    @m.leave(m.Name(value="base"))
    def rename_base(self, original_node, updated_node):
        return updated_node.with_changes(value="base_")

    @m.call_if_inside(m.ClassDef())
    @m.leave(m.FunctionDef())
    def update_method(self, original_node, updated_node):

        if self.is_space_scope(original_node):
            cls_name = cst.ensure_type(
                self.get_parent(original_node, level=2),
                cst.ClassDef,
            ).name.value

            meth_name = original_node.name.value

            if meth_name[: len(FORMULA_PREF)] == FORMULA_PREF:
                # _f_ methods
                if (d := self.cells_info.get(cls_name)):
                    cells = d.get(meth_name[len(FORMULA_PREF):])
                else:
                    cells = None

                decorators = [
                    cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name(CY_MOD), attr=cst.Name("cfunc")
                        )
                    )
                ]
                if cells and cells.has_typeinfo():
                    returns = cst.Annotation(
                        annotation=cst.parse_expression(
                            cells.get_rettype_expr(),
                            config=self.module.config_for_parsing,
                        )
                    )
                    if cells.has_args():
                        parameters = self._add_param_type_hints(
                            updated_node, cls_name=cls_name
                        )
                        return updated_node.with_changes(
                            decorators=decorators, params=parameters, returns=returns
                        )
                    else:
                        return updated_node.with_changes(
                            decorators=decorators, returns=returns
                        )
                else:
                    return updated_node.with_changes(decorators=decorators)

            elif meth_name == MX_COPY_REFS:
                decorators = [
                    cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name(CY_MOD), attr=cst.Name("ccall")
                        )
                    )
                ]

                decl_stmt = cst.parse_statement(
                    f"base_: {cls_name} = {CY_MOD}.cast({cls_name}, base)",
                    config=self.module.config_for_parsing,
                )
                decl_stmt = decl_stmt.with_changes(leading_lines=(cst.EmptyLine(),))
                stmts = list(cst.ensure_type(updated_node.body, cst.IndentedBlock).body)
                stmts.insert(0, decl_stmt)
                indented_block = updated_node.body.with_changes(body=stmts)

                return updated_node.with_changes(
                    decorators=decorators, body=indented_block
                )

            elif meth_name[: len(GLOBAL_PREF)] == GLOBAL_PREF:
                # other _mx_ methods
                return updated_node

            elif meth_name == "__call__":
                # Special methods
                if (d := self.cells_info.get(cls_name)):
                    cells = d.get(meth_name)
                else:
                    cells = None

                if cells and cells.has_typeinfo() and cells.has_args():
                    parameters = self._add_param_type_hints(
                        updated_node, cls_name=cls_name
                    )
                    return updated_node.with_changes(params=parameters)

            elif meth_name[:2] == "__":
                # Special methods
                pass

            else:
                # cells
                cells: CombinedCellsInfo = self.cells_info[cls_name][meth_name]

                decorators = [
                    cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name(CY_MOD), attr=cst.Name("ccall")
                        )
                    )
                ]

                if cells.has_typeinfo():
                    # Return type
                    returns = cst.Annotation(
                        annotation=cst.parse_expression(
                            cells.get_rettype_expr(),
                            config=self.module.config_for_parsing,
                        )
                    )

                    if cells.is_arrayable(self.get_arg_sizes(cls_name)):
                        # Add parameter type hints
                        parameters = self._add_param_type_hints(
                            updated_node, cls_name=cls_name
                        )
                        # Construct indented_block to replace the original one
                        if_expr = f"{MX_SELF}.{HAS_PREF}{meth_name}[{', '.join(cells.params)}]"
                        expr_node = cst.parse_expression(
                            if_expr, config=self.module.config_for_parsing
                        )
                        stmt_node = cst.parse_statement(
                            if_expr + " = True", config=self.module.config_for_parsing
                        )
                        # updated_node.body.body[0].test
                        # FunctionDef.body: IndentedBlock
                        # IndentedBlock.body: tuple(If,)
                        # If.test: Expr
                        # If.orelse: Else
                        # Else.body: IndentedBlock
                        # IndentedBlock.body: tuple(SimpleStatementLine,...)
                        if_node = cst.ensure_type(updated_node.body.body[0], cst.If)
                        stmts = list(
                            cst.ensure_type(if_node.orelse, cst.Else).body.body
                        )
                        stmts.insert(-1, stmt_node)

                        if_node = if_node.with_changes(
                            test=expr_node,
                            orelse=if_node.orelse.with_changes(
                                body=if_node.orelse.body.with_changes(body=stmts)
                            ),
                        )
                        indented_block = cst.ensure_type(
                            updated_node.body, cst.IndentedBlock
                        ).with_changes(body=(if_node,))

                        return updated_node.with_changes(
                            decorators=decorators,
                            params=parameters,
                            returns=returns,
                            body=indented_block,
                        )

                    else:
                        return updated_node.with_changes(
                            decorators=decorators, returns=returns
                        )
                else:
                    return updated_node.with_changes(decorators=decorators)

        return updated_node
