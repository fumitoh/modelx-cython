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

"""Runtime type tracing for exported modelx models.

Run the user's sample script under a :func:`sys.setprofile` profiler
and record the concrete argument and return values of every cells call
so that later stages can annotate the generated Cython modules with
static types.  The machinery is built on vendored MonkeyType code in
:mod:`modelx_cython.monkeytype_tracing`.

The entry point is :func:`trace_calls`, which installs an
:class:`MxCallTracer`; its caller (:func:`modelx_cython.cli.run_sample`)
restricts the tracer with an :class:`MxCodeFilter` to formula methods
(``_f_*``), ``_mx_assign_refs`` and ``__call__`` defined in
``_mx_model.py`` / ``_mx_classes.py`` modules.  Collected traces are
accumulated by :class:`MxCallTraceLogger`, whose :meth:`flush` distills
them into :class:`RuntimeCellsInfo`, :class:`RuntimeRefInfo` and
:class:`RuntimeParamInfo` objects consumed by
:mod:`modelx_cython.builder`.
"""

import sys
import pathlib
import random
import numbers
import itertools
from dataclasses import dataclass
from contextlib import contextmanager
from types import FrameType, MemberDescriptorType
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
    SPACE_PARAMS
)

if (3, 12) <= sys.version_info < (3, 14):
    import opcode
    RETURN_CONST_OPCODE = opcode.opmap["RETURN_CONST"]

_logger = logging.getLogger(__name__)


@dataclass
class ReturnTypeInfo:
    """Return type observed for a traced cells method.

    Attributes
    ----------
    value_type : type
        Type of the returned value, or the element (dtype) type when
        ``is_array`` is ``True``.  May be widened to
        :class:`numbers.Integral`, :class:`numbers.Real` or ``object``
        when traces disagree.
    is_array : bool
        ``True`` if the returned value is a :class:`numpy.ndarray`.
    ndim : int
        Number of array dimensions; ``0`` for non-array values.
    """
    value_type: type
    is_array: bool = False
    ndim: int = 0


class RuntimeCellsInfo:     # TODO: Create base class RuntimeBaseMemberInfo
    """Aggregate runtime type information for one cells method.

    Built from all :class:`CallTrace` records collected for a single
    traced method (normally a cells formula method, ``_f_*``).  Argument types are merged across traces:
    a single observed type is kept as-is; multiple integral types widen
    to :class:`numbers.Integral`; multiple ``str`` subclasses collapse
    to ``str``; any other mixture falls back to ``object`` (logging a
    message when integral and non-integral values are mixed).  For
    integral arguments, the largest observed value is recorded in
    ``max_args``.

    Return types are merged pairwise across traces: equal types are
    kept; arrays of equal ``ndim`` have their element types widened to
    :class:`numbers.Integral`, :class:`numbers.Real` or ``object``;
    arrays of differing ``ndim``, or a mixture of array and non-array
    returns, collapse to a plain ``object`` return type.  Each kind of
    fallback is logged once, with the arguments of the conflicting
    calls.

    Attributes
    ----------
    fqname : str
        Fully qualified dotted name of the traced method.
    name : str
        Bare method name (e.g. ``_f_foo``).
    module : str
        Dotted name of the module defining the method.
    arg_types : dict of str to type
        Merged argument types, excluding ``self``.
    max_args : dict of str to int
        Largest value observed for each integral argument.
    ret_type : ReturnTypeInfo
        Merged return type information.
    """
    name: str
    module: str
    arg_types: Dict[str, type]  # without self
    max_args: Dict[str, int]
    ret_type: ReturnTypeInfo

    def __init__(self, traces: Sequence[CallTrace]) -> None:
        """Build merged type information from a method's traces.

        Parameters
        ----------
        traces : sequence of CallTrace
            Non-empty traces of a single cells method; identity fields
            are taken from the first trace.
        """
        self.fqname = traces[0].funcname
        self.name = traces[0].func.__name__
        self.module = traces[0].func.__module__
        self.arg_types = {}
        self.max_args = {}
        self._init_arg_types(traces)
        self.ret_type = self._init_ret_type(traces)

    def has_args(self):
        """Return whether the cells method takes arguments.

        Returns
        -------
        bool
            ``True`` when at least one non-``self`` argument type was
            recorded.
        """
        return bool(len(self.arg_types))

    def _init_arg_types(self, traces):
        """Merge observed argument types into arg_types and max_args."""
        arg_type_val: Dict[str, dict[type, Any]] = {}

        for trace in traces:

            for arg, val in itertools.islice(
                trace.arg_vals.items(), 1, None
            ):  # remove self
                tp = type(val)
                types = arg_type_val.setdefault(arg, {})
                if tp not in types:
                    types[tp] = val
                elif issubclass(tp, numbers.Integral):
                    if val > types[tp]:
                        types[tp] = val

        for arg, type_val in arg_type_val.items():

            if len(type_val) == 1:
                for tp, val in type_val.items():
                    self.arg_types[arg] = tp
                    if issubclass(tp, numbers.Integral):
                        self.max_args[arg] = val
                    break
            elif all(issubclass(tp, numbers.Integral) for tp in type_val.keys()):
                self.arg_types[arg] = numbers.Integral
                self.max_args[arg] = max(v for v in type_val.values())

            elif all(issubclass(tp, str) for tp in type_val.keys()):
                self.arg_types[arg] = str

            else:
                if (any(issubclass(tp, numbers.Integral) for tp in type_val.keys()) and
                        any(not issubclass(tp, numbers.Integral) for tp in type_val.keys())):
                    msg = ', '.join(k.__name__ + " " + str(v) for k, v in type_val.items())
                    _logger.info(f"varying types given to argument '{arg}' in {self.fqname}: {msg}")

                self.arg_types[arg] = object

    def _init_ret_type(self, traces):
        """Merge observed return types into one ReturnTypeInfo,
        widening or falling back to ``object`` as described in the
        class docstring."""
        last_tp = None
        last_args = None
        has_last = False
        was_dtype_logged = False
        was_ndim_logged = False
        was_vtype_logged = False
        was_mixed_logged = False

        def get_arg_expr(args):
            return ", ".join(f"{k}={str(v)}" for k, v in itertools.islice(args.items(), 1, None))

        for tr in traces:
            val = tr.ret_val
            if isinstance(val, np.ndarray):
                tp = ReturnTypeInfo(
                    val.dtype.type, is_array=True, ndim=val.ndim
                )
            else:
                tp = ReturnTypeInfo(type(val))

            if has_last:
                if last_tp == tp:
                    continue
                elif last_tp.is_array and tp.is_array:
                    if last_tp.ndim == tp.ndim:
                        assert last_tp.value_type != tp.value_type
                        if issubclass(last_tp.value_type, numbers.Integral) and issubclass(tp.value_type, numbers.Integral):
                            last_tp.value_type = numbers.Integral
                        elif issubclass(last_tp.value_type, numbers.Real) and issubclass(tp.value_type, numbers.Real):
                            last_tp.value_type = numbers.Real
                        else:
                            if not was_dtype_logged:
                                # Log varying return types with their arguments
                                args0 = get_arg_expr(last_args)
                                args1 = get_arg_expr(tr.arg_vals)
                                msg0 = f"{last_tp.value_type.__name__} for {args0}"
                                msg1 = f"{tp.value_type.__name__} for {args1}"
                                _logger.info(f"varying array types returned from {self.fqname}:{msg0} and {msg1}")
                                was_dtype_logged = True

                            last_tp.value_type = object

                    else:
                        if not was_ndim_logged:
                            args0 = get_arg_expr(last_args)
                            args1 = get_arg_expr(tr.arg_vals)
                            msg0 = f"{last_tp.ndim} for {args0}"
                            msg1 = f"{tp.ndim} for {args1}"
                            _logger.info(f"varying array dimensions returned from {self.fqname}:{msg0} and {msg1}")
                            was_ndim_logged = True

                        last_tp = ReturnTypeInfo(object)

                elif not last_tp.is_array and not tp.is_array:

                    if issubclass(last_tp.value_type, numbers.Integral) and issubclass(tp.value_type, numbers.Integral):
                        last_tp.value_type = numbers.Integral
                    elif issubclass(last_tp.value_type, numbers.Real) and issubclass(tp.value_type, numbers.Real):
                        last_tp.value_type = numbers.Real
                    else:
                        if not was_vtype_logged:
                            args0 = get_arg_expr(last_args)
                            args1 = get_arg_expr(tr.arg_vals)
                            msg0 = f"{last_tp.value_type.__name__} for {args0}"
                            msg1 = f"{tp.value_type.__name__} for {args1}"
                            _logger.info(f"varying types returned from {self.fqname}:{msg0} and {msg1}")
                            was_vtype_logged = True
                        last_tp.value_type = object

                else:   # one is an array, the other is not
                    if not was_mixed_logged:
                        args0 = get_arg_expr(last_args)
                        args1 = get_arg_expr(tr.arg_vals)
                        msg0 = f"{'array of ' if last_tp.is_array else ''}{last_tp.value_type.__name__} for {args0}"
                        msg1 = f"{'array of ' if tp.is_array else ''}{tp.value_type.__name__} for {args1}"
                        _logger.info(f"varying array and non-array types returned from {self.fqname}:{msg0} and {msg1}")
                        was_mixed_logged = True

                    last_tp = ReturnTypeInfo(object)

            else:
                last_tp = tp
                last_args = tr.arg_vals
                has_last = True

        return last_tp


class RuntimeValueInfo:
    """Runtime type information for a single traced value.

    Attributes
    ----------
    type_ : type
        Concrete type of the value.
    mx_class : str
        Fully qualified name of the value's class when it is defined
        inside the traced model package (i.e. a modelx object),
        otherwise an empty string.
    """

    def __init__(self, value, mx_class=''):
        """Record the type of ``value`` and an optional class name."""
        self.type_ = type(value)
        self.mx_class = mx_class

    @classmethod
    def init_mxobj(cls, value, module):
        """Create an instance, detecting model-defined classes.

        Parameters
        ----------
        value : object
            The traced value.
        module : str
            Name of the model's top-level package.

        Returns
        -------
        RuntimeValueInfo
            With ``mx_class`` set to the value's fully qualified class
            name if the class's module name starts with ``module``,
            otherwise with ``mx_class`` left empty.
        """
        if (value.__class__.__module__)[:len(module)] == module:
            return cls(value, mx_class=value.__class__.__module__ + "." + value.__class__.__qualname__)
        else:
            return cls(value)



class RuntimeRefInfo(RuntimeValueInfo):
    """Runtime type information for a ref observed during tracing."""
    pass


class RuntimeParamInfo(RuntimeValueInfo):
    """Runtime type information for an itemspace parameter value."""
    pass





def replace_first_name(dotted_name: str, name: str):
    """Replace the first component of a dotted name.

    Parameters
    ----------
    dotted_name : str
        A dotted name such as ``"model.space.cells"``.
    name : str
        Replacement for the first component.

    Returns
    -------
    str
        ``dotted_name`` with its first component replaced by ``name``.
    """
    names = dotted_name.split(".")
    names[0] = name
    return ".".join(names)


def instance_attrs(obj: Any) -> Iterator[tuple]:
    """Yield the instance attributes of ``obj`` as name-value pairs.

    Covers both storage kinds an exported model can use.  modelx
    declares ``__slots__`` on the generated space classes from v0.33.0
    on, and a slotted instance has no ``__dict__``; earlier exports,
    and exports made with ``use_slots=False``, store the attributes in
    ``__dict__`` instead.  A class can also have both, so names already
    yielded from ``__dict__`` are not yielded again.

    The slots are read from the ``member_descriptor`` objects that the
    class bodies define rather than from their ``__slots__`` tuples:
    a private slot name is mangled with the declaring class, so the
    tuple entry and the attribute name differ, while the descriptor is
    always stored under the name the attribute is reachable by.

    Parameters
    ----------
    obj : object
        Instance whose attributes to yield.

    Yields
    ------
    tuple of (str, object)
        Attribute name and its value: first the ``__dict__`` entries in
        insertion order, then the slots, class by class along the MRO.
        CPython sorts a class's slot descriptors, so the slot names of
        one class come out in ASCII order and not in the order
        ``__slots__`` declares them.  A slot that has never been
        assigned is skipped.  A name is yielded once, from the first
        place it is found, which for a class carrying both a
        ``__dict__`` entry and a slot of that name is the ``__dict__``
        entry, whereas ``getattr`` would resolve to the slot.
    """
    seen = set()

    for name, val in getattr(obj, "__dict__", {}).items():
        seen.add(name)
        yield name, val

    for klass in type(obj).__mro__:
        for name, member in vars(klass).items():
            if not isinstance(member, MemberDescriptorType) or name in seen:
                continue
            seen.add(name)
            try:
                yield name, getattr(obj, name)
            except AttributeError:  # slot never assigned
                pass


class MxCallTracer(CallTracer):
    """Add return_value to CallTrace

    A :class:`CallTracer` subclass that records the concrete argument
    and return *values* (not just their types) into :class:`CallTrace`
    records, and that only traces functions whose top-level module name
    equals ``module``.
    """

    def __init__(
        self,
        module: str,
        logger: CallTraceLogger,
        max_typed_dict_size: int,
        code_filter: Optional[CodeFilter] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Initialize the tracer.

        Parameters
        ----------
        module : str
            Top-level package name of the model; only functions in
            this package are traced.
        logger : CallTraceLogger
            Receives each completed trace via its ``log`` method.
        max_typed_dict_size : int
            Stored by the base class; unused by this subclass.
        code_filter : CodeFilter, optional
            Predicate limiting which code objects are traced.
        sample_rate : int, optional
            If given, only one in ``sample_rate`` calls is traced.
        """
        super().__init__(logger, max_typed_dict_size, code_filter, sample_rate)
        self.module = module

    def handle_call(self, frame: FrameType) -> None:
        """Start a trace for a call, capturing argument values.

        Unlike the base class, the actual argument values (not their
        types) are stored in the new :class:`CallTrace`.  Honors
        ``sample_rate``, skips frames whose function cannot be
        resolved, and ignores resumed generator frames already present
        in ``self.traces``.
        """
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
        """Record the return (or yield) value and log the trace.

        The last executed opcode distinguishes a genuine return from an
        unwind due to an unhandled exception: on ``YIELD_VALUE`` the
        value is stored as the return value and the frame stays active;
        on ``RETURN_VALUE`` (or ``RETURN_CONST`` on Python 3.12/3.13)
        ``arg`` is stored as the return value.  In every non-yield case
        the trace is removed from ``self.traces`` and passed to
        ``self.logger.log`` -- after an exceptional unwind, without a
        return value.
        """
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
            if (3, 12) <= sys.version_info < (3, 14):
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
        """Profile callback dispatching call and return events.

        Skips unsupported events, code objects named ``trace_types``
        and code rejected by the code filter, then additionally
        requires the function's top-level module name to equal
        ``self.module``.  Exceptions raised while collecting a trace
        are logged, never propagated.
        """
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
    """Code filter selecting modelx model code objects to trace.

    Accepts only code objects whose function name is a formula
    (``_f_*``), ``_mx_assign_refs``, or ``__call__``, and whose file
    name ends with ``_mx_model.py`` or ``_mx_classes.py`` (only a
    filename-suffix comparison is performed, for speed).
    """

    def __call__(self, code):
        """Return ``True`` if the code object should be traced."""
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
    """Log and store/print records collected by a CallTracer.

    Accumulates raw :class:`CallTrace` records grouped by fully
    qualified function name, then on :meth:`flush` distills them into
    runtime type information for cells, refs and itemspace parameters.

    Attributes
    ----------
    module : str
        Top-level package name of the traced model.
    new_name : str or None
        If set, :meth:`flush` renames the model root component of all
        recorded dotted names to this name.
    cells_info : dict
        Fully qualified cells name to :class:`RuntimeCellsInfo`,
        populated by :meth:`flush`.
    ref_info : dict
        Fully qualified ref name to :class:`RuntimeRefInfo`, populated
        by :meth:`flush`.
    modules : list of str
        Dotted names of the modules in which traced cells are defined.
    param_info : dict
        Fully qualified space class name to a dict mapping parameter
        names to :class:`RuntimeParamInfo`.
    """

    def __init__(self, module: str, new_model_name: str = None) -> None:
        """Initialize empty stores.

        Parameters
        ----------
        module : str
            Top-level package name of the traced model.
        new_model_name : str, optional
            New model root name applied by :meth:`flush`.
        """
        super().__init__()
        self.module = module
        self.new_name = new_model_name
        self._traces = {}  # funcname -> [trace]
        self.cells_info = {}  # funcname -> MethodTypeInfo
        self.ref_info = {}
        self.modules = []
        self.param_info = {}  # class name -> {param: RuntimeParamInfo}

    def log(self, trace: CallTrace) -> None:
        """Log a single call trace.

        The trace is appended to an internal list keyed by its fully
        qualified function name, for aggregation in :meth:`flush`.
        """
        self._traces.setdefault(trace.funcname, []).append(trace)

    def flush(self) -> None:
        """Aggregate the accumulated traces into runtime info objects.

        Traces of ``_mx_assign_refs`` provide refs: each user-defined
        attribute of the traced ``self`` (taken from the first trace
        only) becomes a :class:`RuntimeRefInfo` in ``ref_info``.  All
        other traces (formula methods and space ``__call__`` methods
        admitted by the code filter) are treated as cells: each group
        builds a :class:`RuntimeCellsInfo` in ``cells_info`` and
        registers its module in ``modules``.  The raw traces are then cleared,
        itemspace parameters are collected by walking the model, and,
        if a new model name was given, the model root component of all
        stored names is renamed to it.
        """
        for k, v in self._traces.items():

            name_split = k.split(".")
            tr0 = v[0]  # First trace
            if name_split[-1] == MX_ASSIGN_REFS:
                # Extract refs
                for name, val in instance_attrs(tr0.arg_vals[MX_SELF]):
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
        """Change the model name stored in members

        Replace the first (model root) component of the keys, module
        names and ``mx_class`` values with ``self.new_name``.
        """

        
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
        """Find the model instance in the ``_mx_model`` module and walk
        each of its top-level spaces to collect itemspace parameters."""
        module = sys.modules[self.module + "." + MX_MODEL_MOD]
        base_cls = getattr(sys.modules[self.module + "." + MX_SYS_MOD], BASE_MODEL)
        model_cls = next(v for v in module.__dict__.values() if isinstance(v, type) and issubclass(v, base_cls))
        model = next(v for v in module.__dict__.values() if isinstance(v, model_cls))

        for space in model._mx_spaces.values():
            self._walk_space(space, {})

    def _walk_space(self, top_space, params):
        """Recursively record itemspace parameter values as
        :class:`RuntimeParamInfo` in ``param_info``, keyed by space
        class name."""
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
    """Enable call tracing for a block of code

    A context manager that installs an :class:`MxCallTracer` as the
    profile function via :func:`sys.setprofile` for the duration of
    the ``with`` block.  On exit, the previous profile function is
    restored and ``logger.flush()`` is called.

    Parameters
    ----------
    module : str
        Top-level package name of the model to trace.
    logger : CallTraceLogger
        Receives and aggregates the collected traces.
    max_typed_dict_size : int
        Passed through to the tracer (unused by it).
    code_filter : CodeFilter, optional
        Predicate limiting which code objects are traced.
    sample_rate : int, optional
        If given, only one in ``sample_rate`` calls is traced.

    Yields
    ------
    None
        Control returns to the caller with tracing active.
    """
    old_trace = sys.getprofile()
    sys.setprofile(
        MxCallTracer(module, logger, max_typed_dict_size, code_filter, sample_rate)
    )
    try:
        yield
    finally:
        sys.setprofile(old_trace)
        logger.flush()
