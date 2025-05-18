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

from typing import Union, Sequence, Mapping
try:
    from types import NoneType
except ImportError: # Python -3.9
    NoneType = type(None)

import textwrap
from functools import cached_property
import libcst as cst
from libcst._flatten_sentinel import FlattenSentinel
from libcst._nodes.module import Module
from libcst._nodes.statement import BaseStatement, ClassDef
from libcst._removal_sentinel import RemovalSentinel
import libcst.matchers as m
from libcst.metadata import ParentNodeProvider, ScopeProvider, GlobalScope, ClassScope

from modelx_cython.parser import ParentScopeAddin
from modelx_cython.builder import ModuleInfo, CombinedCellsInfo

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
    MX_ASSIGN_REFS,
    MX_COPY_REFS,
    is_user_defined,
)

from modelx_cython.typedefs import CY_BOOL_T

class PXDGenerator:

    pxd_template = textwrap.dedent("""\
    {cmodule_imports}
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

    def __init__(self, module: ModuleInfo):
        self.module = module

    @cached_property
    def package(self) -> str:
        return self.module.fqname.split(".")[0]

    @cached_property
    def cmodule_imports(self) -> str:
        stmts = []
        for ci in self.module.cimports:
            stmts.append(f"cimport {ci}")
        return "\n".join(stmts)

    @cached_property
    def code(self):
        return self.pxd_template.format(
            package=self.package,
            cmodule_imports=self.cmodule_imports,
            MX_SYS_MOD=MX_SYS_MOD,
            child_cimports=self.child_cimports,
            class_defs=self.class_defs
        )

    @cached_property
    def class_defs(self):
        stmts = []
        for cls in self.module.classes.keys():
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
        parent = ".".join(self.module.fqname.split(".")[:-1])
        stmts = []
        for child in self.module.sub_modules:
            stmts.append(f"from cython.cimports.{parent} import {child}\n")

        return "".join(stmts)

    def private_var_defs(self, cls_name):

        cls_info = self.module.classes[cls_name]
        decl_stmts = []
        for cells in cls_info.cells.values():

            assert cells.module == self.module.fqname
            assert cells.cls == cls_name

            if cells.is_special():
                continue

            if cells.has_args():
                if cells.has_typeinfo() and cells.is_arrayable():

                    var_name = VAR_PREF + cells.name
                    var_type = cells.get_array_decl_expr(c_style=True)
                    decl_stmts.append(f"cdef {var_type} {var_name}\n")

                    has_name = HAS_PREF + cells.name
                    has_type = cells.get_array_decl_expr(
                                rettype_expr=CY_BOOL_T, c_style=True)
                    decl_stmts.append(f"cdef {has_type} {has_name}\n")

                else:
                    decl_stmts.append(f"cdef dict {VAR_PREF + cells.name}\n")
            else:
                rettype = cells.get_rettype_expr(c_style=True)
                decl_stmts.append(f"cdef {rettype} {VAR_PREF + cells.name}\n")
                decl_stmts.append(f"cdef {CY_BOOL_T} {HAS_PREF + cells.name}\n")

        return "".join(decl_stmts)

    def public_var_defs(self, cls_name):

        decl_stmts = []
        for ref in self.module.classes[cls_name].refs.values():

            assert ref.module == self.module.fqname and ref.cls == cls_name

            stmt = f"cdef public {ref.get_type_expr(c_style=True)} {ref.name}\n"
            decl_stmts.append(stmt)

        # Declare child spaces
        for space in self.module.classes[cls_name].spaces:
            mod_name = MODULE_PREF + cls_name[len(SPACE_PREF):]  # Replace prefix for submodule
            rel_path = mod_name + "." + MX_SPACE_MOD + "." + SPACE_PREF + space

            stmt = f"cdef public {rel_path} {space}\n"
            decl_stmts.append(stmt)

        return "".join(decl_stmts)

    def _add_param_type_hints(
        self, cls_name: str, cells_name: str
    ) -> str:

        cells = self.module.classes[cls_name].cells[cells_name]
        params = [f"{cls_name} {MX_SELF}"]  # add self first

        # Add parameter type hints
        if cells and cells.has_typeinfo() and cells.has_args():
            for param in cells.params:
                type_ = cells.get_argtype_expr(param, c_style=True)
                params.append(f"{type_} {param}")
        else:
            for p in cells.params:
                params.append(f"object {p}")

        return ", ".join(params)

    def private_meth_defs(self, cls_name):

        decl_stmts = []
        for cells in self.module.classes[cls_name].cells.values():

            if cells.is_special():
                continue

            if cells and cells.has_typeinfo():
                rettype = cells.get_rettype_expr(c_style=True)
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
        for cells in self.module.classes[cls_name].cells.values():

            if cells.is_special():
                continue

            if cells and cells.has_typeinfo():
                rettype = cells.get_rettype_expr(c_style=True)
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


class ModuleTransformer(m.MatcherDecoratableTransformer, ParentScopeAddin):
    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(
        self,
        source: str,
        module: ModuleInfo
    ) -> None:
        super().__init__()
        self.wrapper = cst.metadata.MetadataWrapper(cst.parse_module(source))
        self._module_node = self.wrapper.module
        self.module = module
        self.package = module.fqname.split(".")[0]

    @property   # cannot use cached_property in Transformer
    def transformed(self):
        return self.wrapper.visit(self)

    def leave_Module(self, original_node: Module, updated_node: Module) -> Module:

        # cimports for child spaces
        parent = ".".join(self.module.fqname.split(".")[:-1])
        stmts = []
        for child in self.module.sub_modules:
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
            cls_info = self.module.classes[cls_name]
            decl_stmts = []
            for cells in cls_info.cells.values():

                assert cells.module == self.module.fqname
                assert cells.cls == cls_name

                if cells.is_special():
                    continue

                if cells.has_args():
                    if cells.has_typeinfo() and cells.is_arrayable():
                        decl_stmts.append(
                            cst.parse_statement(
                                VAR_PREF
                                + cells.name
                                + ": "
                                + cells.get_array_decl_expr(),
                                config=self._module_node.config_for_parsing,
                            )
                        )
                        decl_stmts.append(
                            cst.parse_statement(
                                HAS_PREF
                                + cells.name
                                + ": "
                                + cells.get_array_decl_expr(
                                    rettype_expr=f"{CY_MOD}.{CY_BOOL_T}",
                                ),
                                config=self._module_node.config_for_parsing,
                            )
                        )
                    else:
                        decl_stmts.append(
                            cst.parse_statement(
                                VAR_PREF + cells.name + ": dict",
                                config=self._module_node.config_for_parsing,
                            )
                        )
                else:
                    rettype = cells.get_rettype_expr()
                    decl_stmts.append(
                        cst.parse_statement(
                            VAR_PREF + cells.name + ": " + rettype,
                            config=self._module_node.config_for_parsing,
                        )
                    )

                    decl_stmts.append(
                        cst.parse_statement(
                            HAS_PREF + cells.name + ": " + CY_MOD + "." + CY_BOOL_T,
                            config=self._module_node.config_for_parsing,
                        )
                    )

            is_first = True
            for ref in self.module.classes[cls_name].refs.values():

                assert ref.module == self.module.fqname and ref.cls == cls_name

                stmt = cst.parse_statement(
                    f"{ref.name}: {ref.get_type_expr()}",
                    config=self._module_node.config_for_parsing,
                )
                if is_first:
                    stmt = stmt.with_changes(
                        leading_lines=tuple(stmt.leading_lines) + (cst.EmptyLine(),)
                    )
                    is_first = False

                decl_stmts.append(stmt)

            # Declare child spaces
            is_first = True
            for space in self.module.classes[cls_name].spaces:

                mod_name = MODULE_PREF + cls_name[len(SPACE_PREF):]  # Replace prefix for submodule
                rel_path = mod_name + "." + MX_SPACE_MOD + "." + SPACE_PREF + space

                stmt = cst.parse_statement(
                    f"{space}: {rel_path}",
                    config=self._module_node.config_for_parsing,
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

        cells = self.module.classes[cls_name].cells[name]

        # Add parameter type hints
        updated_params = [funcdef.params.params[0]]  # add self first
        for param in param_list:
            param_name = param.name.value
            if cells.get_argtype_expr(param_name):
                updated_params.append(
                    param.with_changes(
                        annotation=cst.Annotation(
                            annotation=cst.parse_expression(
                                cells.get_argtype_expr(param_name),
                                config=self._module_node.config_for_parsing,
                            )
                        )
                    )
                )
            else:
                updated_params.append(param)

        return funcdef.params.with_changes(params=tuple(updated_params))

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
            cls_info = self.module.classes[cls_name]

            if meth_name[: len(FORMULA_PREF)] == FORMULA_PREF:
                # _f_ methods
                cells = cls_info.cells.get(meth_name[len(FORMULA_PREF):])

                decorators = [
                    cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name(CY_MOD), attr=cst.Name("cfunc")
                        )
                    )
                ]
                returns = cst.Annotation(
                    annotation=cst.parse_expression(
                        cells.get_rettype_expr(),
                        config=self._module_node.config_for_parsing,
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
                    return updated_node.with_changes(decorators=decorators, returns=returns)


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
                    config=self._module_node.config_for_parsing,
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
                cells = cls_info.cells[meth_name]

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
                cells: CombinedCellsInfo = cls_info.cells[meth_name]

                decorators = [
                    cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name(CY_MOD), attr=cst.Name("ccall")
                        )
                    )
                ]
                # Return type
                returns = cst.Annotation(
                    annotation=cst.parse_expression(
                        cells.get_rettype_expr(),
                        config=self._module_node.config_for_parsing,
                    )
                )
                if cells.has_args():
                    # Add parameter type hints
                    parameters = self._add_param_type_hints(
                        updated_node, cls_name=cls_name
                    )
                    if cells.has_typeinfo() and cells.is_arrayable():

                        # Construct indented_block to replace the original one
                        c_idx_expr = ''.join([f"[{p}]" for p in cells.params])
                        param_expr = f"{', '.join([p for p in cells.params])}"

                        has_expr = f"{MX_SELF}.{HAS_PREF}{meth_name}{c_idx_expr}"
                        v_expr = f"{MX_SELF}.{VAR_PREF}{meth_name}{c_idx_expr}"
                        f_expr = f"{MX_SELF}.{FORMULA_PREF}{meth_name}({param_expr})"

                        args = tuple(cells.params)
                        size = cls_info.cells_arg_sizes[args]

                        idx_range = " and ".join(
                            [f"(0 <= {p} < {i})" for p, i in zip(args, size)])

                        if_stmt = textwrap.dedent(f"""\
                        if {idx_range}:
                            if {has_expr}:
                                return {v_expr}
                            else:
                                val = {f_expr}
                                {v_expr} = val
                                {has_expr} = True
                                return val
                        else:
                            raise IndexError("array index out of range")
                        """)
                        if_node = cst.parse_statement(
                            if_stmt, config=self._module_node.config_for_parsing
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
                            decorators=decorators,
                            params=parameters,
                            returns=returns,
                            body=self._add_dict_assign(meth_name, updated_node)
                        )
                else:   # No type info, no arg
                    return updated_node.with_changes(
                        decorators=decorators,
                        returns=returns
                    )


        return updated_node

    def _add_dict_assign(self, meth_name: str, updated_node) -> cst.IndentedBlock:
        """Add dict assignment in method

        Example:
            if self._v_meth is None:
                self._v_meth = {}
        """
        init_stmt = cst.parse_statement(
            f"if {MX_SELF}.{VAR_PREF}{meth_name} is None:\n    {MX_SELF}.{VAR_PREF}{meth_name} = {{}}",
            config=self._module_node.config_for_parsing
        )

        return cst.ensure_type(
            updated_node.body, cst.IndentedBlock
        ).with_changes(body=(init_stmt,) + updated_node.body.body)

