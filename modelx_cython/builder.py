# Copyright (c) 2023-2025 Fumito Hamamura <fumito.ham@gmail.com>
import numbers
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
import logging

try:
    from types import NoneType
except ImportError:  # Python -3.9
    NoneType = type(None)

from functools import cached_property

from modelx_cython.config import TransSpec
from modelx_cython.tracer import RuntimeCellsInfo, get_type_expr, MxCallTraceLogger
from modelx_cython.parser import ModuleVisitor, LexicalCellsInfo, LexicalRefInfo

from modelx_cython.consts import (
    SPACE_PREF,
    MODULE_PREF
)

_logger = logging.getLogger(__name__)


class CombinedCellsInfo(LexicalCellsInfo):  # TODO: Inherit both Lexical and Runtime and assert shared members match
    parent: 'ClassInfo'
    _rt: RuntimeCellsInfo
    _spec: dict

    def __init__(self, cls_info, lx_info, rt_info, spec) -> None:
        super().__init__(
            lx_info.module, lx_info.cls, lx_info.name, lx_info.params
        )
        self.parent = cls_info
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

        ret_t = self._spec.get(TransSpec.RET_T)
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

    def is_arrayable(self):
        if self.has_args() and self.has_typeinfo():
            for p in self.params:
                tp = self._rt.arg_types.get(p, type)
                if issubclass(tp, numbers.Integral):
                    assert p in self.parent.cells_arg_sizes
                    continue
                else:
                    return False
            return True
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


class CombinedRefInfo:
    module: str
    cls: str
    name: str
    type_: type = None
    mx_class: str = ''
    decl_type_expr: str = ''
    is_relative: bool = False

    def __init__(self, module,
                 cls,
                 name,
                 rt_info):
        if rt_info:

            if rt_info.mx_class:
                if rt_info.mx_class[:len(module)] == module:
                    # Defined in a child space
                    decl_type_expr = rt_info.mx_class[len(module) + 1:]
                    is_relative = True
                else:
                    decl_type_expr = rt_info.mx_class
                    is_relative = False
            else:
                decl_type_expr = ''
                is_relative = False

            self.module = module
            self.cls = cls
            self.name = name
            self.type_ = rt_info.type_
            self.mx_class = rt_info.mx_class
            self.decl_type_expr = decl_type_expr
            self.is_relative = is_relative
        else:
            self.module = module
            self.cls = cls
            self.name = name

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
        self._cells_max_args = {}
        self._max_arg_cells = {}    # keep cells fqname for logging
        self.refs = {}
        self.spaces = []
        self._init_cells()
        self._init_spaces()
        self._init_refs()
        self._add_space_params()

    def _init_cells(self):
        for name, lx_info in self.visitor.cells_info[self.name].items():
            rt_info = self.logger.cells_info.get(lx_info.fqname, None)
            self.cells[name] = CombinedCellsInfo(
                self,
                lx_info, rt_info,
                self.module.spec.get_spec(self.fqname).get(TransSpec.CELLS, {}).get(name, {})
            )
            if rt_info:
                for k, v in rt_info.max_args.items():
                    if k not in self._cells_max_args or v > self._cells_max_args[k]:
                        self._cells_max_args[k] = v
                        self._max_arg_cells[k] = lx_info.fqname

    def _init_spaces(self):
        self.spaces.extend(self.visitor.spaces.get(self.name, []))

    def _init_refs(self):
        for name, lx_info in self.visitor.ref_info.get(self.name, {}).items():
            rt_info = self.logger.ref_info.get(
                lx_info.fqname, None
            )
            self.refs[name] = CombinedRefInfo(
                self.module.fqname,
                self.name,
                name,
                rt_info=rt_info
            )

    def _add_space_params(self):
        params = self.logger.param_info.get(self.fqname, None)
        if params:
            for param, rt_info in params.items():
                self.refs[param] = CombinedRefInfo(
                    module=self.module.fqname,
                    cls=self.name,
                    name=param,
                    rt_info=rt_info
                )

    @cached_property
    def fqname(self):
        return self.module.fqname + "." + self.name

    @cached_property
    def cells_arg_sizes(self) -> Mapping[str, int]:
        params = self.module.spec.get_spec(self.fqname).get(TransSpec.CELLS_PARAMS, {})
        sizes = {k: v[TransSpec.SIZE] for k, v in params.items() if TransSpec.SIZE in v}
        for k, v in self._cells_max_args.items():
            if k in sizes:
                if v + 1 > sizes[k]:
                    _logger.info(f"Specified max size of {sizes[k]} for cells parameter {k} in {self.name} is replaced by {v + 1} from {self._max_arg_cells[k]}")
                    sizes[k] = v + 1
            else:
                sizes[k] = v + 1

        return sizes


class ModuleInfo:

    fqname: str
    visitor: ModuleVisitor
    logger: MxCallTraceLogger
    spec: TransSpec
    classes: dict   # class name -> ClassInfo

    def __init__(self, fqname: str, visitor: ModuleVisitor, logger: MxCallTraceLogger,
                 spec: TransSpec):

        self.fqname = fqname
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


