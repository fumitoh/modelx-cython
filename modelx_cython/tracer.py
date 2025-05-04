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

# Portions of this source file are derived from MonkeyType, 
# a Python library for generating static type annotations from runtime types.
# Refer to the original license and copyright included with this library.


import sys
import pathlib
import random
import numbers
import itertools
from dataclasses import dataclass
from contextlib import contextmanager
from types import FrameType
from typing import Any, Mapping, Iterator, Sequence, Optional, Dict, List
import logging

import numpy as np

from modelx_cython.monkeytype_tracing import (
    CallTraceLogger,
    get_func,
    CodeFilter,
    CallTrace,
    CallTracer,
    RETURN_VALUE_OPCODE,
    YIELD_VALUE_OPCODE,
    SUPPORTED_EVENTS,
    EVENT_CALL,
    EVENT_RETURN,
)

from modelx_cython.consts import (
    MX_ASSIGN_REFS,
    FORMULA_PREF,
    is_user_defined,
    MX_SELF,
    MX_MODEL_MOD,
    MX_SPACE_MOD,
    MX_SYS_MOD,
    BASE_MODEL,
    SPACE_PREF,
    SPACE_PARAMS,
    CY_MOD,
    CY_INT_T,
    CY_INT_T_P
)

if sys.version_info >= (3, 12):
    import opcode
    RETURN_CONST_OPCODE = opcode.opmap["RETURN_CONST"]

_logger = logging.getLogger(__name__)

@dataclass
class ValueInfo:
    value_type: type
    is_array: bool = False
    ndim: int = 0


class RuntimeCellsInfo:     # TODO: Create base class RuntimeBaseMemberInfo
    name: str
    module: str
    arg_types: Mapping[str, type]  # without self
    max_args: Mapping[str, int]
    ret_type: ValueInfo

    def __init__(self, traces: Sequence[CallTrace]) -> None:
        self.name = traces[0].func.__name__
        self.module = traces[0].func.__module__
        self.arg_types = self._init_arg_types(traces)
        self.max_args = self._get_max_args(traces)
        self.ret_type = self._init_ret_type(traces)

    def has_args(self):
        return bool(len(self.arg_types))

    def _init_arg_types(self, traces) -> Mapping[str, type]:
        types: Dict[str, List[type]] = {}
        for trace in traces:
            for arg, val in itertools.islice(
                trace.arg_vals.items(), 1, None
            ):  # remove self
                typ = type(val)
                typs = types.setdefault(arg, [])
                if typ not in typs:
                    typs.append(typ)

        return {arg: self._get_arg_type(typs) for arg, typs in types.items()}

    def _get_max_args(self, traces):
        result = {}
        for trace in traces:
            for arg, val in itertools.islice(
                trace.arg_vals.items(), 1, None
            ):  # remove self
                if issubclass(self.arg_types[arg], numbers.Integral):
                    if arg not in result or val > result[arg]:
                        result[arg] = val

        return result


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
            ret_type = self.get_value_type(tr.ret_val)
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
            return ValueInfo(
                normalize_type(value.dtype.type), is_array=True, ndim=value.ndim
            )
        else:
            return ValueInfo(normalize_type(type(value)))


class RuntimeValueInfo:

    def __init__(self, value, mx_class=''):
        self.type_ = type(value)
        self.mx_class = mx_class

    @classmethod
    def init_mxobj(cls, value, module):
        if (value.__class__.__module__)[:len(module)] == module:
            return cls(value, mx_class=value.__class__.__module__ + "." + value.__class__.__qualname__)
        else:
            return cls(value)



class RuntimeRefInfo(RuntimeValueInfo):
    pass


class RuntimeParamInfo(RuntimeValueInfo):
    pass


def get_type_expr(typ, with_module=True, use_double=False):
    if typ is int:
        if with_module:
            return f"{CY_MOD}.{CY_INT_T_P}"
        else:
            return CY_INT_T
    elif typ is float:
        if use_double:
            if with_module:
                return f"{CY_MOD}.double"
            else:
                return "double"
        else:
            return "float"

    elif typ is str:
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


def comp_and_get_type(lt: ValueInfo, rt: ValueInfo) -> ValueInfo:
    assert lt != rt  # must be different types

    if lt.is_array and rt.is_array:
        if lt.ndim == rt.ndim:
            return ValueInfo(
                coerce_type(lt.value_type, rt.value_type), is_array=True, ndim=lt.ndim
            )
        else:
            ValueInfo(object)

    elif not lt.is_array and not lt.is_array:  # both are non-array
        return ValueInfo(coerce_type(lt.value_type, rt.value_type))

    else:  # lt or rt is array
        return ValueInfo(object)


def replace_first_name(dotted_name: str, name: str):
    names = dotted_name.split(".")
    names[0] = name
    return ".".join(names)


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

    def handle_call(self, frame: FrameType) -> None:
        if self.sample_rate and random.randrange(self.sample_rate) != 0:
            return
        func = self._get_func(frame)
        if func is None:
            return
        code = frame.f_code
        # I can't figure out a way to access the value sent to a generator via
        # send() from a stack frame.
        if frame in self.traces:
            # resuming a generator; we've already seen this frame
            return
        arg_names = code.co_varnames[0: code.co_argcount]
        arg_vals = {}
        for name in arg_names:
            if name in frame.f_locals:
                arg_vals[name] = frame.f_locals[name]

        self.traces[frame] = CallTrace(func, arg_vals)

    def handle_return(self, frame: FrameType, arg: Any) -> None:
        # In the case of a 'return' event, arg contains the return value, or
        # None, if the block returned because of an unhandled exception. We
        # need to distinguish the exceptional case (not a valid return type)
        # from a function returning (or yielding) None. In the latter case, the
        # the last instruction that was executed should always be a return or a
        # yield.
        last_opcode = frame.f_code.co_code[frame.f_lasti]
        trace = self.traces.get(frame)
        if trace is None:
            return
        elif last_opcode == YIELD_VALUE_OPCODE:
            trace.ret_val = arg
        else:
            if sys.version_info >= (3, 12):
                ret_opcodes = (RETURN_VALUE_OPCODE, RETURN_CONST_OPCODE)
            else:
                ret_opcodes = (RETURN_VALUE_OPCODE,)
            if last_opcode in ret_opcodes:
                trace.ret_val = arg

            # Extract and store local variables
            # code = frame.f_code
            # arg_names = code.co_varnames[0: code.co_argcount]
            # loc_vals = {}
            # for loc in frame.f_locals:
            #     if loc not in arg_names:
            #         loc_vals[loc] = frame.f_locals[loc]
            # trace.loc_vals = loc_vals

            del self.traces[frame]
            self.logger.log(trace)

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
                _logger.error("Cannot handle event %s", event)
        except Exception:
            _logger.exception("Failed collecting trace")
        return self


class MxCodeFilter:

    def __call__(self, code):
        # Since called many times, check module name and function name only
        if (
            code.co_filename[-len(MX_MODEL_MOD) - 3: -3] == MX_MODEL_MOD
            or code.co_filename[-len(MX_SPACE_MOD) - 3: -3] == MX_SPACE_MOD
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

    def __init__(self, module: str, new_model_name: str = None) -> None:
        super().__init__()
        self.module = module
        self.new_name = new_model_name
        self._traces = {}  # funcname -> [trace]
        self.cells_info = {}  # funcname -> MethodTypeInfo
        self.ref_info = {}
        self.modules = []
        self.param_info = {}  # class name -> {param: RuntimeParamInfo}

    def log(self, trace: CallTrace) -> None:
        """Log a single call trace."""
        self._traces.setdefault(trace.funcname, []).append(trace)

    def flush(self) -> None:
        for k, v in self._traces.items():

            name_split = k.split(".")
            tr0 = v[0]  # First trace
            if name_split[-1] == MX_ASSIGN_REFS:
                # Extract refs
                for name, val in tr0.arg_vals[MX_SELF].__dict__.items():
                    if is_user_defined(name):
                        fqname = ".".join(name_split[:-1] + [name])
                        self.ref_info[fqname] = RuntimeRefInfo.init_mxobj(val, self.module)
            else:
                # Extract cells
                self.cells_info[k] = info = RuntimeCellsInfo(v)
                if info.module not in self.modules:
                    self.modules.append(info.module)

        self._traces.clear()
        self._get_params()

        if self.new_name:
            self._update_model_name()

    def _update_model_name(self):
        """Change the model name stored in members"""

        
        for i, v in enumerate(self.modules):
            self.modules[i] = replace_first_name(v, self.new_name)

        for key in list(self.cells_info.keys()):
            v = self.cells_info.pop(key)
            v.module = replace_first_name(v.module, self.new_name)
            self.cells_info[replace_first_name(key, self.new_name)] = v

        for key in list(self.ref_info.keys()):
            v = self.ref_info.pop(key)
            if v.mx_class:
                v.mx_class = replace_first_name(v.mx_class, self.new_name)
            self.ref_info[replace_first_name(key, self.new_name)] = v

        for key in list(self.param_info.keys()):
            params = self.param_info[key]
            for v in list(params.values()):
                if v.mx_class:
                    v.mx_class = replace_first_name(v.mx_class, self.new_name)
            self.param_info[replace_first_name(key, self.new_name)] = params

    def _get_params(self):
        module = sys.modules[self.module + "." + MX_MODEL_MOD]
        base_cls = getattr(sys.modules[self.module + "." + MX_SYS_MOD], BASE_MODEL)
        model_cls = next(v for v in module.__dict__.values() if isinstance(v, type) and issubclass(v, base_cls))
        model = next(v for v in module.__dict__.values() if isinstance(v, model_cls))

        for space in model._mx_spaces.values():
            self._walk_space(space, {})

    def _walk_space(self, top_space, params):
        for space in top_space._mx_walk():

            if params:
                key = space.__class__.__module__ + "." + space.__class__.__name__
                val = self.param_info.get(key, None)
                if (not val) or len(val) < len(params):
                    self.param_info[key] = params

            if hasattr(space, "__call__"):
                space_name = space.__class__.__name__[len(SPACE_PREF):]
                param_list = getattr(sys.modules[space.__class__.__module__], SPACE_PARAMS + "_" + space_name)

                for k, v in space._mx_itemspaces.items():
                    next_params = params.copy()
                    if isinstance(k, tuple):
                        for name, arg in zip(param_list, k):
                            next_params[name] = RuntimeParamInfo.init_mxobj(arg, self.module)
                    else:
                        next_params[param_list[0]] = RuntimeParamInfo.init_mxobj(k, self.module)

                    self._walk_space(v, next_params)


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
