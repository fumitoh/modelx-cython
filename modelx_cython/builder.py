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
except ImportError:  # Python -3.9
    NoneType = type(None)
from dataclasses import dataclass
from functools import cached_property

from modelx_cython.config import TranslationSpec
from modelx_cython.tracer import RuntimeCellsInfo, get_type_expr, MxCallTraceLogger
from modelx_cython.parser import ModuleVisitor, LexicalCellsInfo, LexicalRefInfo

from modelx_cython.consts import (
    SPACE_PREF,
    MODULE_PREF
)


class CombinedCellsInfo(LexicalCellsInfo):  # TODO: Inherit both Lexical and Runtime and assert shared members match
    _rt: RuntimeCellsInfo
    _spec: dict

    def __init__(self, lex_info, rt_info, spec) -> None:
        super().__init__(
            lex_info.module_name, lex_info.cls_name, lex_info.name, lex_info.params
        )
        self._rt = rt_info
        self._spec = spec

    def has_typeinfo(self):
        return bool(self._rt)

    def has_args(self):
        return bool(self.params)

    def get_argtype_expr(self, arg: str, with_module=True, use_double=False) -> str:
        if arg in self._rt.arg_types:
            return get_type_expr(self._rt.arg_types[arg], with_module=with_module, use_double=use_double)
        else:
            return ""

    def get_rettype_expr(self, with_module=True, use_double=False):

        ret_t = self._spec.get(TranslationSpec.RET_T)
        if ret_t:
            return ret_t
        elif self.has_typeinfo():
            val_t = get_type_expr(self._rt.ret_type.value_type, with_module=with_module, use_double=use_double)
            if self._rt.ret_type.ndim:
                return val_t + "[" + ", ".join(":" * self._rt.ret_type.ndim) + "]"
            else:
                return val_t
        else:
            return "object"

    def is_arrayable(self, sizes):
        if self.has_typeinfo():
            return self._rt.is_arrayable(sizes)
        else:
            return False

    def get_decltype_expr(self, sizes: Mapping[str, int], rettype_expr="", with_module=True, use_double=False):
        if not rettype_expr:
            rettype_expr = self.get_rettype_expr(with_module=with_module, use_double=use_double)
        return (
            rettype_expr
            + "["
            + ", ".join([str(sizes[arg]) for arg in self._rt.arg_types.keys()])
            + "]"
        )

@dataclass
class CombinedRefInfo:
    module_name: str
    cls_name: str
    name: str
    type_: type = None
    mx_class: str = ''
    decl_type_expr: str = ''
    is_relative: bool = False

    # TODO: Define __init__ similar to CombinedCellsInfo and replace init_with_rt

    @classmethod
    def init_with_rt(cls, module_name,
                     cls_name,
                     name,
                     rt_info):
        if rt_info:

            if rt_info.mx_class:
                if rt_info.mx_class[:len(module_name)] == module_name:
                    # Defined in a child space
                    decl_type_expr = rt_info.mx_class[len(module_name) + 1:]
                    is_relative = True
                else:
                    decl_type_expr = rt_info.mx_class
                    is_relative = False
            else:
                decl_type_expr = ''
                is_relative = False

            return cls(module_name=module_name,
                       cls_name=cls_name,
                       name=name,
                       type_=rt_info.type_,
                       mx_class=rt_info.mx_class,
                       decl_type_expr=decl_type_expr,
                       is_relative=is_relative)
        else:
            return cls(module_name=module_name,
                       cls_name=cls_name,
                       name=name)

    def get_type_expr(self, with_module=True, use_double=False):
        if self.decl_type_expr:
            return self.decl_type_expr
        else:
            return get_type_expr(self.type_, with_module=with_module, use_double=use_double)


class ClassInfo:

    name: str
    module: 'ModuleInfo'
    visitor: ModuleVisitor
    logger: MxCallTraceLogger
    cells:  dict  # name -> CombinedCellsInfo
    refs:   dict  # name -> CombinedRefInfo
    spaces: list
    params: dict  # name -> CombinedRefInfo

    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.visitor = module.visitor
        self.logger = module.logger
        self.cells = {}
        self.refs = {}
        self.spaces = []
        self._init_cells()
        self._init_spaces()
        self._init_refs()
        self._add_space_params()

    def _init_cells(self):
        for name, lex_info in self.visitor.cells_info[self.name].items():
            rt_info = self.logger.cells_info.get(lex_info.fqname, None)
            self.cells[name] = CombinedCellsInfo(
                lex_info,
                rt_info,
                self.module.spec.get_spec(self.module.module_name + "." + self.name).get(TranslationSpec.CELLS, {}).get(name, {})
            )

    def _init_spaces(self):
        self.spaces.extend(self.visitor.spaces.get(self.name, []))

    def _init_refs(self):
        for name, lex_info in self.visitor.ref_info.get(self.name, {}).items():
            rt_info = self.logger.ref_info.get(
                lex_info.fqname, None
            )
            self.refs[name] = CombinedRefInfo.init_with_rt(
                self.module.module_name,
                self.name,
                name,
                rt_info=rt_info
            )

    def _add_space_params(self):
        key = self.module.module_name + "." + self.name
        params = self.logger.param_info.get(key, None)
        if params:
            for param, rt_info in params.items():
                self.refs[param] = CombinedRefInfo.init_with_rt(
                    module_name=self.module.module_name,
                    cls_name=self.name,
                    name=param,
                    rt_info=rt_info,
                )

    @cached_property
    def fqname(self):
        return self.module.module_name + "." + self.name

    @cached_property
    def cells_arg_sizes(self):  # TODO: Define self.spec and use it instead
        return self.module.spec.get_arg_sizes(self.module.module_name, cls_name=self.name)


class ModuleInfo:

    module_name: str
    visitor: ModuleVisitor
    logger: MxCallTraceLogger
    spec: TranslationSpec
    classes: dict   # class name -> ClassInfo

    def __init__(self, module_name:str, visitor: ModuleVisitor, logger: MxCallTraceLogger,
                 spec: TranslationSpec):

        self.module_name = module_name
        self.visitor = visitor
        self.logger = logger
        self.spec = spec
        self.classes = {}
        self._init_classes()

    def _init_classes(self):
        for c in self.visitor.classes:
            self.classes[c] = ClassInfo(c, self)

    @cached_property
    def cimports(self):
        result = []
        for cls in self.classes.values():
            for r in cls.refs.values():
                if r.decl_type_expr and not r.is_relative:
                    mod = ".".join(r.decl_type_expr.split(".")[:-1])
                    if mod not in result:
                        result.append(mod)
        return result

    @cached_property
    def sub_modules(self):
        result = []
        for cls in self.visitor.spaces:
            result.append(MODULE_PREF + cls[len(SPACE_PREF):])    # replace _c_ with _m_
        return result


