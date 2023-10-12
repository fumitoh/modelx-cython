from typing import Union
from types import NoneType
from dataclasses import dataclass
import libcst as cst
from libcst._flatten_sentinel import FlattenSentinel
from libcst._nodes.module import Module
from libcst._nodes.statement import BaseStatement, ClassDef
from libcst._removal_sentinel import RemovalSentinel
import libcst.matchers as m
from libcst.metadata import ParentNodeProvider, ScopeProvider, GlobalScope, ClassScope

from modelx_cython.tracer import TypeInfo, DynamicTypeInfo

from modelx_cython.consts import (
    FORMULA_PREF,
    CY_MOD,
    GLOBAL_PREF,
    VAR_PREF,
    HAS_PREF,
    SPACE_PREF,
    MX_SELF,
    MX_SYS_MOD,
    CY_BOOL_T,
    MX_ASSIGN_REFS,
    MX_COPY_REFS,
    is_user_defined,
)


class LexicalCellsInfo:
    module_name: str
    cls_name: str
    name: str
    params: list[str]

    def __init__(self, module_name, cls_name, name, params) -> None:
        self.module_name: str = module_name
        self.cls_name: str = cls_name
        self.name: str = name
        self.params: list[str] = params

    @property
    def keystr(self):
        pref = "" if self.is_special() else FORMULA_PREF
        return self.module_name + "." + self.cls_name + "." + pref + self.name

    def is_special(self):
        return self.name[:2] == self.name[-2:] == "__"


class CombinedCellsInfo(LexicalCellsInfo):
    _typeinfo: DynamicTypeInfo

    def __init__(self, cellsinfo, typeinfo) -> None:
        super().__init__(
            cellsinfo.module_name, cellsinfo.cls_name, cellsinfo.name, cellsinfo.params
        )
        self._typeinfo = typeinfo

    def has_typeinfo(self):
        return bool(self._typeinfo)

    def has_args(self):
        return bool(self.params)

    def get_rettype_expr(self):
        if self.has_typeinfo():
            return self._typeinfo.get_rettype_expr()
        else:
            return "object"

    def is_arrayable(self, sizes):
        if self.has_typeinfo():
            return self._typeinfo.is_arrayable(sizes)
        else:
            return False

    def get_decltype_expr(self, sizes: dict[str, int], rettype_expr=""):
        return self._typeinfo.get_decltype_expr(sizes, rettype_expr=rettype_expr)


@dataclass
class RefInfo:
    module_name: str
    cls_name: str
    name: str
    type_expr: Union[str, NoneType]

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
        return  bool(
            isinstance(scope, ClassScope)
            and scope.name[: len(SPACE_PREF)] == SPACE_PREF
            and isinstance(scope.parent, GlobalScope)
        )

class SpaceVisitor(m.MatcherDecoratableVisitor, SpaceAddin):
    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(self, module_name, source, type_info: dict, ref_type_info: dict):
        super().__init__()
        self.module_name = module_name
        self.cells_info = {}
        self.ref_info = {}
        self.space_info = {}
        self.type_info = type_info
        self.ref_type_info = ref_type_info
        self.wrapper = cst.metadata.MetadataWrapper(cst.parse_module(source))
        self.wrapper.visit(self)


    @m.call_if_inside(m.ClassDef())
    @m.call_if_inside(m.FunctionDef(name=cst.Name("__init__")))
    @m.leave(m.SimpleStatementLine())
    def collect_space_info(self, original_node):
        if self.is_space_scope(original_node, level=2):
            node = original_node

            try:
                target = cst.ensure_type(node.body[0], cst.Assign).targets[0].target
            except Exception:
                return
            
            if (target.value.value == MX_SELF and is_user_defined(target.attr.value)):
                self.space_info[target.attr.value] = self.wrapper.module.code_for_node(node.body[0].value)

    @m.call_if_inside(m.ClassDef())
    @m.call_if_inside(m.FunctionDef(name=cst.Name(MX_ASSIGN_REFS)))
    @m.leave(m.SimpleStatementLine())
    def collect_refs_info(self, original_node):

        if self.is_space_scope(original_node, level=2):
            # SimpleStatemetn in IndentedBlock in FunctionDef in IndentedBlock in ClassDef

            node = original_node
            for _ in range(4):
                node = self.get_metadata(ParentNodeProvider, node)

            cls_name = cst.ensure_type(node, cst.ClassDef).name.value

            try:
                name = cst.ensure_type(
                    cst.ensure_type(original_node.body[0], cst.Assign).targets[0],
                    cst.AssignTarget,
                ).target.attr.value
            except Exception:   # igonore other than assignments, such as 'pass'
                return

            ref_type_info = self.ref_type_info.get(
                self.module_name + "." + cls_name + "." + name, None
            )
            self.ref_info[self.module_name, cls_name, name] = RefInfo(
                self.module_name,
                cls_name,
                name,
                ref_type_info.type_expr if ref_type_info else None,
            )

    @m.call_if_inside(m.ClassDef())
    @m.visit(m.FunctionDef())
    def collect_methods(self, original_node):

        if self.is_space_scope(original_node):
            cls_name = cst.ensure_type(
                self.get_metadata(
                    ParentNodeProvider,
                    self.get_metadata(ParentNodeProvider, original_node),
                ),
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
                    module_name=self.module_name,
                    cls_name=cls_name,
                    name=name,
                    params=params,
                )
                self.cells_info[cls_name, name] = CombinedCellsInfo(
                    ci, self.type_info.get(ci.keystr, None)
                )

        return False


class SpaceTransformer(m.MatcherDecoratableTransformer, SpaceAddin):
    METADATA_DEPENDENCIES = (ScopeProvider, ParentNodeProvider)

    def __init__(
        self,
        module_name: str,
        source: str,
        type_info: dict,
        ref_type_info: dict,
        config: dict,
    ) -> None:
        super().__init__()
        self.module_name = module_name
        self.wrapper = cst.metadata.MetadataWrapper(cst.parse_module(source))
        self.module = self.wrapper.module
        self.type_info = type_info
        self.config = config
        space = SpaceVisitor(module_name, source, type_info, ref_type_info)
        self.cells_info = space.cells_info
        self.ref_info = space.ref_info
        self.space_info = space.space_info

    @property
    def package(self) -> str:
        return self.module_name.split(".")[0]

    @property
    def transformed(self):
        return self.wrapper.visit(self)

    def get_arg_sizes(self, cls_name: str) -> dict[str, int]:
        space = self.config.get(cls_name[len(SPACE_PREF) :], {})
        params = space.get("cells_params", {})
        return {k: v["size"] for k, v in params.items() if "size" in v}

    def leave_Module(self, original_node: Module, updated_node: Module) -> Module:
        return updated_node.with_changes(
            body=(
                cst.parse_statement(
                    f"from cython.cimports.{self.package} import {MX_SYS_MOD}",
                    config=updated_node.config_for_parsing,
                ),
                cst.parse_statement(
                    f"import cython as {CY_MOD}", config=updated_node.config_for_parsing
                ),
                *updated_node.body,
            )
        )

    def leave_ClassDef(
        self, original_node: ClassDef, updated_node: ClassDef
    ) -> BaseStatement | FlattenSentinel[BaseStatement] | RemovalSentinel:
        cls_name: str = original_node.name.value
        if cls_name[: len(SPACE_PREF)] == SPACE_PREF and isinstance(
            self.get_metadata(ScopeProvider, original_node), GlobalScope
        ):
            decl_stmts = []
            for cells in self.cells_info.values():
                if cells.is_special():
                    pass
                elif cells.has_args():
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
                                    rettype_expr=CY_BOOL_T,
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
                            HAS_PREF + cells.name + ": " + CY_BOOL_T,
                            config=self.module.config_for_parsing,
                        )
                    )

            is_first = True
            for ref in self.ref_info.values():
                stmt = cst.parse_statement(
                    f"{ref.name} = {CY_MOD}.declare({ref.type_expr}, visibility='public')",
                    config=self.module.config_for_parsing,
                )
                if is_first:
                    stmt = stmt.with_changes(
                        leading_lines=stmt.leading_lines + (cst.EmptyLine(),)
                    )
                    is_first = False

                decl_stmts.append(stmt)

            is_first = True
            for space in self.space_info.keys():

                stmt = cst.parse_statement(
                    f"{space} = {CY_MOD}.declare('object', visibility='public')",
                    config=self.module.config_for_parsing,
                )
                if is_first:
                    stmt = stmt.with_changes(
                        leading_lines=stmt.leading_lines + (cst.EmptyLine(),)
                    )
                    is_first = False

                decl_stmts.append(stmt)


            decorator = cst.Decorator(
                decorator=cst.Attribute(value=cst.Name(CY_MOD), attr=cst.Name("cclass"))
            )

            if decl_stmts:
                # Add blank lines below classdef
                decl_stmts[0] = decl_stmts[0].with_changes(
                    leading_lines=decl_stmts[0].leading_lines + (cst.EmptyLine(),)
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
        funcdef = self.get_metadata(
            ParentNodeProvider, self.get_metadata(ParentNodeProvider, original_node)
        )
        clsdef = self.get_metadata(
            ParentNodeProvider, self.get_metadata(ParentNodeProvider, funcdef)
        )
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
        paramstr_list = [p.name.value for p in param_list]

        type_info = self._get_type_info(funcdef, cls_name)

        if paramstr_list:
            # Add parameter type hints
            if type_info:
                updated_params = [funcdef.params.params[0]]  # add self first
                for param in param_list:
                    param_name = param.name.value
                    if type_info.get_argtype_expr(param_name):
                        updated_params.append(
                            param.with_changes(
                                annotation=cst.Annotation(
                                    annotation=cst.parse_expression(
                                        type_info.get_argtype_expr(param_name),
                                        config=self.module.config_for_parsing,
                                    )
                                )
                            )
                        )
                    else:
                        updated_params.append(param)

                return funcdef.params.with_changes(params=tuple(updated_params))

        return None

    def _get_type_info(
        self, funcdef: cst.FunctionDef, cls_name: str
    ) -> Union[TypeInfo, None]:
        name = funcdef.name.value

        pref: str = (
            ""
            if (
                name[: len(FORMULA_PREF)] == FORMULA_PREF
                or name[:2] == name[-2:] == "__"
            )
            else FORMULA_PREF
        )
        return self.type_info.get(self.module_name + "." + cls_name + "." + pref + name)

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
                self.get_metadata(
                    ParentNodeProvider,
                    self.get_metadata(ParentNodeProvider, original_node),
                ),
                cst.ClassDef,
            ).name.value

            meth_name = original_node.name.value

            has_params = bool(
                (updated_node.params.params + updated_node.params.posonly_params)[1:]
            )  # True if not empty

            if meth_name[: len(FORMULA_PREF)] == FORMULA_PREF:
                # _f_ methods
                type_info = self._get_type_info(original_node, cls_name)

                decorators = [
                    cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name(CY_MOD), attr=cst.Name("cfunc")
                        )
                    )
                ]
                if type_info:
                    returns = cst.Annotation(
                        annotation=cst.parse_expression(
                            type_info.get_rettype_expr(),
                            config=self.module.config_for_parsing,
                        )
                    )
                    if has_params:
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
                cells = self.cells_info.get((cls_name, meth_name))

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
                cells: CombinedCellsInfo = self.cells_info[cls_name, meth_name]

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
