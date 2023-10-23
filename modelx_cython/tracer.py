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

# Portions of this source file are derived from MonkeyType, 
# a Python library for generating static type annotations from runtime types.
# Refer to the original license and copyright included with this library.


import sys
import pathlib
import numbers
import itertools
from dataclasses import dataclass
from contextlib import contextmanager
from types import FrameType
from typing import Any, Mapping, Iterator, Sequence, Optional

import numpy as np

from monkeytype.tracing import (
    CallTraceLogger,
    get_func,
    CodeFilter,
    CallTrace,
    CallTracer,
    RETURN_VALUE_OPCODE,
    YIELD_VALUE_OPCODE,
    SUPPORTED_EVENTS,
    logger,
    EVENT_CALL,
    EVENT_RETURN,
)
from monkeytype.typing import get_type

from modelx_cython.consts import (
    MX_ASSIGN_REFS,
    FORMULA_PREF,
    is_user_defined,
    MX_SELF,
    MX_MODEL_MOD,
    MX_SPACE_MOD,
    CY_MOD,
    CY_INT_T,
)


@dataclass
class TypeInfo:
    value_type: type
    is_array: bool = False
    ndim: int = 0


class DynamicTypeInfo:
    name: str
    module: str
    class_: str
    arg_types: Mapping[str, type]  # without self
    ret_type: TypeInfo

    def __init__(self, traces: Sequence[CallTrace]) -> None:
        self.name = traces[0].func.__name__
        self.module = traces[0].func.__module__
        self.class_ = ".".join(traces[0].func.__qualname__.split(".")[:-1])
        self.arg_types = self._init_arg_types(traces)
        self.ret_type = self._init_ret_type(traces)

    def get_argtype_expr(self, arg: str) -> str:
        if arg in self.arg_types:
            return get_type_expr(self.arg_types[arg])
        else:
            return ""

    def get_rettype_expr(self) -> str:
        val_t = get_type_expr(self.ret_type.value_type)
        if self.ret_type.ndim:
            return val_t + "[" + ", ".join(":" * self.ret_type.ndim) + "]"
        else:
            return val_t

    def is_arrayable(self, sizes: Mapping[str, int]) -> bool:
        return (
            self.has_args()
            and all(t is int for t in self.arg_types.values())
            and set(self.arg_types) == set(sizes)
        )

    def get_decltype_expr(self, sizes: Mapping[str, int], rettype_expr="") -> str:
        if not rettype_expr:
            rettype_expr = self.get_rettype_expr()
        return (
            rettype_expr
            + "["
            + ", ".join([str(sizes[arg]) for arg in self.arg_types.keys()])
            + "]"
        )

    def has_args(self):
        return bool(len(self.arg_types))

    def _init_arg_types(self, traces) -> Mapping[str, type]:
        types: Mapping[str, Sequence[type]] = {}
        for trace in traces:
            for arg, typ in itertools.islice(
                trace.arg_types.items(), 1, None
            ):  # remove self
                typs = types.setdefault(arg, [])
                if typ not in typs:
                    typs.append(typ)

        return {arg: self._get_arg_type(typs) for arg, typs in types.items()}

    def _get_arg_type(self, types: Sequence[type]):
        if len(types) == 1:
            return types[0]
        else:
            lt = types[0]
            for rt in types[1:]:
                lt = comp_and_get_type(lt, rt)
            return lt

    def _init_ret_type(self, traces):
        ret_types = []
        for tr in traces:
            ret_type = self.get_value_type(tr.return_value)
            if ret_type not in ret_types:
                ret_types.append(ret_type)

        if len(ret_types) == 1:
            return ret_types[0]
        else:
            lt = ret_types[0]
            for rt in ret_types[1:]:
                lt = comp_and_get_type(lt, rt)
            return lt

    @staticmethod
    def get_value_type(value):
        if isinstance(value, np.ndarray):
            return TypeInfo(
                normalize_type(value.dtype.type), is_array=True, ndim=value.ndim
            )
        else:
            return TypeInfo(normalize_type(type(value)))


@dataclass
class RefTypeInfo:
    type_expr: str

    def __init__(self, value):
        self.type_expr = get_type_expr(type(value))


def get_type_expr(typ):
    if typ is int:
        return CY_INT_T
    elif typ is float:
        return "float"
    elif type is str:
        return "str"
    else:
        return "object"


def normalize_type(typ: type):
    if issubclass(typ, numbers.Number):
        if issubclass(typ, numbers.Rational):
            return int
        else:
            return float
    else:
        if issubclass(typ, str):
            return str
        else:
            return object


def coerce_type(lt: type, rt: type):
    if issubclass(lt, numbers.Number) and issubclass(rt, numbers.Number):
        if issubclass(lt, numbers.Rational) and issubclass(rt, numbers.Rational):
            return int
        else:
            return float
    elif issubclass(lt, str) and issubclass(rt, str):
        return str
    else:
        object


def comp_and_get_type(lt: TypeInfo, rt: TypeInfo) -> TypeInfo:
    assert lt != rt  # must be different types

    if lt.is_array and rt.is_array:
        if lt.ndim == rt.ndim:
            return TypeInfo(
                coerce_type(lt.value_type, rt.value_type), is_array=True, ndim=lt.ndim
            )
        else:
            TypeInfo(object)

    elif not lt.is_array and not lt.is_array:  # both are non-array
        return TypeInfo(coerce_type(lt.value_type, rt.value_type))

    else:  # lt or rt is array
        return TypeInfo(object)


class MxCallTracer(CallTracer):
    """Add return_value to CallTrace"""

    def __init__(
        self,
        module: str,
        logger: CallTraceLogger,
        max_typed_dict_size: int,
        code_filter: Optional[CodeFilter] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        super().__init__(logger, max_typed_dict_size, code_filter, sample_rate)
        self.module = module

    def handle_return(self, frame: FrameType, arg: Any) -> None:
        # In the case of a 'return' event, arg contains the return value, or
        # None, if the block returned because of an unhandled exception. We
        # need to distinguish the exceptional case (not a valid return type)
        # from a function returning (or yielding) None. In the latter case, the
        # the last instruction that was executed should always be a return or a
        # yield.
        typ = get_type(arg, max_typed_dict_size=self.max_typed_dict_size)
        last_opcode = frame.f_code.co_code[frame.f_lasti]
        trace = self.traces.get(frame)
        if trace is None:
            return
        elif last_opcode == YIELD_VALUE_OPCODE:
            trace.add_yield_type(typ)
            trace.return_value = arg  # Monkeytyped for modelx
        else:
            if last_opcode == RETURN_VALUE_OPCODE:
                trace.return_type = typ
                trace.return_value = arg  # Monkeytyped for modelx
            del self.traces[frame]
            self.logger.log(trace)
            if trace.func.__name__ == MX_ASSIGN_REFS:
                self.logger.members[trace.funcname] = {
                    k: v
                    for k, v in frame.f_locals[MX_SELF].__dict__.items()
                    if is_user_defined(k)
                }

    def __call__(self, frame: FrameType, event: str, arg: Any) -> "CallTracer":
        code = frame.f_code
        if (
            event not in SUPPORTED_EVENTS
            or code.co_name == "trace_types"
            or self.should_trace
            and not self.should_trace(code)
        ):
            return self

        # Filter by module name here
        func = get_func(frame)
        if not func or not func.__module__.split(".")[0] == self.module:
            return self

        try:
            if event == EVENT_CALL:
                self.handle_call(frame)
            elif event == EVENT_RETURN:
                self.handle_return(frame, arg)
            else:
                logger.error("Cannot handle event %s", event)
        except Exception:
            logger.exception("Failed collecting trace")
        return self


class MxCodeFilter:

    def __call__(self, code):
        # Since called many times, check module name and function name only
        if (
            code.co_filename[-len(MX_MODEL_MOD) - 3 : -3] == MX_MODEL_MOD
            or code.co_filename[-len(MX_SPACE_MOD) - 3 : -3] == MX_SPACE_MOD
        ):
            if code.co_name[:3] == FORMULA_PREF:
                return True
            elif code.co_name == MX_ASSIGN_REFS:
                return True
            elif code.co_name == "__call__":
                return True

        return False


class MxCallTraceLogger(CallTraceLogger):
    """Log and store/print records collected by a CallTracer."""

    def __init__(self, new_model_name: str = None) -> None:
        super().__init__()
        self.new_name = new_model_name
        self._traces = {}  # funcname -> [trace]
        self.members = {}  # (module, qualname) -> value
        self.type_info = {}  # funcname -> MethodTypeInfo
        self.ref_type_info = {}
        self.modules = []

    def log(self, trace: CallTrace) -> None:
        """Log a single call trace."""
        self._traces.setdefault(trace.funcname, []).append(trace)

    def flush(self) -> None:
        for k, v in self._traces.items():
            self.type_info[k] = info = DynamicTypeInfo(v)
            if info.module not in self.modules:
                self.modules.append(info.module)
        self._traces.clear()

        for k, v in self.members.items():
            for name, value in v.items():
                names = k.split(".")
                assert names[-1] == MX_ASSIGN_REFS
                self.ref_type_info[".".join(names[:-1] + [name])] = RefTypeInfo(value)
        self.members.clear()

        if self.new_name:
            self._change_model_name()

    def _change_model_name(self):
        """Change the model name stored in members"""

        def replace_first_name(dotted_name: str, name: str):
            names = dotted_name.split(".")
            names[0] = name
            return ".".join(names)           
        
        for i, v in enumerate(self.modules):
            self.modules[i] = replace_first_name(v, self.new_name)

        for key in list(self.type_info.keys()):
            v = self.type_info.pop(key)
            v.module = replace_first_name(v.module, self.new_name)
            self.type_info[replace_first_name(key, self.new_name)] = v

        for key in list(self.ref_type_info.keys()):
            self.ref_type_info[replace_first_name(key, self.new_name)] = self.ref_type_info.pop(key)



@contextmanager
def trace_calls(
    module: str,
    logger: CallTraceLogger,
    max_typed_dict_size: int,
    code_filter: Optional[CodeFilter] = None,
    sample_rate: Optional[int] = None,
) -> Iterator[None]:
    """Enable call tracing for a block of code"""
    old_trace = sys.getprofile()
    sys.setprofile(
        MxCallTracer(module, logger, max_typed_dict_size, code_filter, sample_rate)
    )
    try:
        yield
    finally:
        sys.setprofile(old_trace)
        logger.flush()
