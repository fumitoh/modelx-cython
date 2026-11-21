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
(``_f_*``), the public methods of uncached cells, ``_mx_assign_refs``
and ``__call__`` defined in ``_mx_model.py`` / ``_mx_classes.py``
modules.  Collected traces are
accumulated by :class:`MxCallTraceLogger`, whose :meth:`flush` distills
them into :class:`RuntimeCellsInfo`, :class:`RuntimeRefInfo` and
:class:`RuntimeParamInfo` objects consumed by
:mod:`modelx_cython.builder`.
"""

import sys
import inspect
import pathlib
import random
import numbers
import operator
import enum
import itertools
from enum import IntEnum
from dataclasses import dataclass
from contextlib import contextmanager
from types import CodeType, FrameType, FunctionType, MemberDescriptorType
from typing import Any, Mapping, Iterator, Sequence, Optional, Dict, List
import logging

import numpy as np

from modelx_cython.helpers import helper_capability, registered_helper
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
    vector_shape: object = None     # vector cells: see vector_shape()


def vector_shape(val):
    """The vector shape of a traced value (vector cells, ``vectors.py``).

    'V' for a list of exact floats, 'I' for a list of exact ints within int64,
    'E' for an empty list (either), ('T', kinds) for a non-empty tuple of such
    lists, else None. Keyed on the value only.
    """
    if type(val) is list:
        if not val:
            return 'E'
        if all(type(x) is float for x in val):
            return 'V'
        if all(type(x) is int and -(1 << 63) <= x < (1 << 63) for x in val):
            return 'I'
        return None
    if type(val) is tuple and val:
        kinds = tuple(vector_shape(x) if type(x) is list else None for x in val)
        return ('T', kinds) if None not in kinds else None
    return None


def merge_vector_shapes(a, b):
    """The shape both traced shapes fit, else None ('E' fits 'V' and 'I')."""
    if a == b:
        return a
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a[1]) != len(b[1]):
            return None
        kinds = tuple(merge_vector_shapes(x, y) for x, y in zip(a[1], b[1]))
        return ('T', kinds) if None not in kinds else None
    if a == 'E' and b in ('V', 'I'):
        return b
    if b == 'E' and a in ('V', 'I'):
        return a
    return None


class RuntimeCellsInfo:     # TODO: Create base class RuntimeBaseMemberInfo
    """Aggregate runtime type information for one cells method.

    Built from all :class:`CallTrace` records collected for a single
    traced method: a cells formula method (``_f_*``), the public
    method of an uncached cells, which is the formula itself, or a
    space's ``__call__``.  Argument types are merged across traces:
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
    Scalar integral/nonintegral real mixtures may widen to double when values
    are exactly preserved. Explicit object Cell declarations retain scalar types.

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

    def __init__(self, traces: Sequence[CallTrace], receiver_calls=None) -> None:
        """Build merged type information from a method's traces.

        Parameters
        ----------
        traces : sequence of CallTrace
            Non-empty traces of a single cells method; identity fields
            are taken from the first trace.
        receiver_calls : dict, optional
            Calls per receiver counted over all traced calls, when the
            logger kept only the traces that add an observation.
        """
        self.fqname = traces[0].funcname
        self.name = traces[0].func.__name__
        self.module = traces[0].func.__module__
        self.arg_types = {}
        self.max_args = {}
        self.min_args = {}
        self.observed_min_args = {}
        self.observed_max_args = {}
        self.enum_args = {}         # argument -> (module, qualname, member count) of its IntEnum
        if receiver_calls is None:
            receiver_calls = {}
            for trace in traces:
                _count_receiver(receiver_calls, trace)
        self.max_evaluations_per_receiver = max(count for _, count in receiver_calls.values())
        # D111: a call that raised gives its arguments but no return value; a Cell whose every
        # traced call raised keeps them all (its result is then traced as None, as before).
        returned = [trace for trace in traces if not getattr(trace, 'raised', False)] or list(traces)
        arrays = [trace.ret_val for trace in returned if isinstance(trace.ret_val, np.ndarray)]
        formats = {array.dtype.str for array in arrays}
        self.array_buffer_compatible = (len(formats) == 1 and all(
            array.ndim > 0 and array.dtype.isnative
            and array.dtype in (np.dtype('float64'), np.dtype('int64'))
            for array in arrays)) if arrays else True
        self.has_negative_args = False
        values = [float(trace.ret_val) for trace in returned
                  if isinstance(trace.ret_val, (int, float, np.integer, np.floating))
                  and not isinstance(trace.ret_val, (bool, np.bool_))]
        values = [value for value in values if value == value]
        self.max_ret_value = max(values) if values else None   # param_bound_checks margin
        self._init_arg_types(traces)
        self.ret_type = self._init_ret_type(returned)
        from .provider_manifest import observed_provider_binding
        self.observed_provider_binding = observed_provider_binding(returned)
        self.table_schema = table_schema(returned)
        self.frame_schema = frame_schema(returned)

    admitted_table = False     # auto-tables: the traced value was a pandas frame admitted as a table
    pandas_fallbacks = {}      # auto-tables: {formula line: {operation: calls}} on admitted tables
    admission_record = None    # auto-tables: what admission decided for this zero-argument formula
    frame_row_result = False   # auto-tables (D119): every traced value is a row of an admitted frame

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
        wide_arguments = set()
        # Public Cell formulas are uncached. Preserve an authored object
        # argument there even when the sample supplies only one scalar type.
        annotations = traces[0].func.__annotations__ if is_user_defined(self.name) else {}

        for trace in traces:

            for arg, val in itertools.islice(
                trace.arg_vals.items(), 1, None
            ):  # remove self
                tp = type(val)
                integral = issubclass(tp, numbers.Integral)
                if integral:
                    val = operator.index(val)
                    if val < 0:
                        self.has_negative_args = True
                    if not -(1 << 63) <= val < (1 << 63):
                        wide_arguments.add(arg)
                    self.min_args[arg] = min(self.min_args.get(arg, 0), val)
                    self.observed_min_args[arg] = min(self.observed_min_args.get(arg, val), val)
                    self.observed_max_args[arg] = max(self.observed_max_args.get(arg, val), val)
                types = arg_type_val.setdefault(arg, {})
                if tp not in types:
                    types[tp] = val
                elif integral:
                    if val > types[tp]:
                        types[tp] = val

        for arg, type_val in arg_type_val.items():

            if annotations.get(arg) is object or arg in wide_arguments:
                self.arg_types[arg] = object
                continue

            if len(type_val) == 1:
                for tp, val in type_val.items():
                    self.arg_types[arg] = tp
                    if issubclass(tp, numbers.Integral):
                        self.max_args[arg] = val
                        self._widen_to_enum(arg, tp)
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

    def _widen_to_enum(self, arg, tp):
        """An argument whose every traced value is a member of one IntEnum class can take any
        member: its domain is the class's member values, not the members the sample used."""
        if not (issubclass(tp, enum.IntEnum) and '<locals>' not in tp.__qualname__):
            return
        values = [int(member) for member in tp]
        if not values or not all(-(1 << 63) <= v < (1 << 63) for v in values):
            return
        low, high = min(values), max(values)
        self.max_args[arg] = max(self.max_args[arg], high)
        self.min_args[arg] = min(self.min_args.get(arg, 0), low)
        self.observed_min_args[arg] = min(self.observed_min_args.get(arg, low), low)
        self.observed_max_args[arg] = max(self.observed_max_args.get(arg, high), high)
        if low < 0:
            self.has_negative_args = True
        self.enum_args[arg] = (tp.__module__, tp.__qualname__, len(values))

    def _init_ret_type(self, traces):
        """Merge observed return types into one ReturnTypeInfo,
        widening or falling back to ``object`` as described in the
        class docstring (see :class:`_ReturnTypeMerger`)."""
        # A concrete object annotation declares the result ABI. A spec
        # can override this later, and string annotations are unevaluated.
        if any(trace.func.__annotations__.get('return') is object for trace in traces):
            return ReturnTypeInfo(object)
        merger = _ReturnTypeMerger(self.fqname)
        for tr in traces:
            merger.add(tr)
        return merger.ret_type


def _count_receiver(receiver_calls, trace):
    """Count a call of ``trace``'s receiver. The receiver is held with its count, as a kept trace
    would hold it, so that its id is not reused by another receiver while the counts are open."""
    receiver = trace.arg_vals.get(MX_SELF)
    entry = receiver_calls.get(id(receiver))
    if entry is None:
        receiver_calls[id(receiver)] = [receiver, 1]
    else:
        entry[1] += 1


# Returns read by value only (type, magnitude): a repeat of one adds nothing to RuntimeCellsInfo
_PLAIN_RETURNS = frozenset((int, float, bool, str, type(None), np.float64, np.float32, np.int64, np.int32, np.bool_))


def _add_arg_vals(arg_type_val: Dict[str, Dict[type, Any]], minimums, trace) -> bool:
    """Record the argument values of ``trace`` in ``arg_type_val``.

    ``arg_type_val`` maps each argument, ``self`` excluded, to its
    observed types, each mapped to the value quoted for it: the first
    value seen, or for an integral type the largest. Preserve a new
    lower integral value too, so deduplication cannot hide a wide argument.
    """
    changed = False
    for arg, val in itertools.islice(trace.arg_vals.items(), 1, None):
        tp = type(val)
        types = arg_type_val.setdefault(arg, {})
        if tp not in types:
            types[tp] = val
            changed = True
        elif issubclass(tp, numbers.Integral) and val > types[tp]:
            types[tp] = val
            changed = True
        if issubclass(tp, numbers.Integral):
            value = int(val)
            if arg not in minimums or value < minimums[arg]:
                minimums[arg] = value
                changed = True
    return changed


class _ReturnTypeMerger:
    """Merge the return types of a method's traces, one by one.

    The merge of :class:`RuntimeCellsInfo` (see its docstring), held as
    a state that each trace updates, so that
    :meth:`MxCallTraceLogger.log` can tell a trace that changes nothing.
    Each kind of fallback is logged at most once, quoting the arguments
    of the first trace and of the trace that caused it.

    Parameters
    ----------
    fqname : str
        Name of the method, for the log messages.
    log : bool, default True
        False to merge without logging, as the trace logger does.
    """

    def __init__(self, fqname, log=True):
        self.fqname = fqname
        self.log = log
        self.ret_type: Optional[ReturnTypeInfo] = None
        self._first_args = None
        self._logged = set()
        self._array_formats = set()
        self._integral_double_exact = True

    @staticmethod
    def _arg_expr(args):
        return ", ".join(f"{k}={str(v)}"
                         for k, v in itertools.islice(args.items(), 1, None))

    def _fallback(self, kind, tr, msg0, msg1):
        """Log the fallback ``kind`` unless already logged; return
        whether it was logged now."""
        if kind in self._logged:
            return False
        self._logged.add(kind)
        if self.log:
            args0 = self._arg_expr(self._first_args)
            args1 = self._arg_expr(tr.arg_vals)
            _logger.info(f"varying {kind} returned from {self.fqname}:"
                         f"{msg0} for {args0} and {msg1} for {args1}")
        return True

    def add(self, tr) -> bool:
        """Merge the return value of trace ``tr``.

        Returns
        -------
        bool
            Whether the merge changed: the merged type, or a fallback
            logged for the first time.  A trace for which it is False
            can be left out without changing the merged type or the
            messages logged.
        """
        val = tr.ret_val
        changed = False
        if isinstance(val, np.ndarray):
            tp = ReturnTypeInfo(val.dtype.type, is_array=True, ndim=val.ndim)
            # Buffer compatibility depends on byte order and rank as well
            # as the NumPy scalar type used to merge return types.
            if not self.log:
                signature = (val.dtype.str, val.ndim > 0)
                changed = signature not in self._array_formats
                self._array_formats.add(signature)
        else:
            tp = ReturnTypeInfo(object if isinstance(val, numbers.Integral)
                                and not -(1 << 63) <= val < (1 << 63) else type(val),
                                vector_shape=vector_shape(val))
            if self._integral_double_exact and isinstance(val, numbers.Integral):
                try:
                    integer = operator.index(val)
                    self._integral_double_exact = int(float(integer)) == integer
                except (TypeError, ValueError, OverflowError):
                    self._integral_double_exact = False
                changed = not self._integral_double_exact

        last_tp = self.ret_type
        if last_tp is None:
            self.ret_type = tp
            self._first_args = tr.arg_vals
            return True
        if last_tp == tp:
            return changed
        if (last_tp.value_type is tp.value_type and not tp.is_array and not last_tp.is_array
                and last_tp.vector_shape != tp.vector_shape):
            shape = merge_vector_shapes(last_tp.vector_shape, tp.vector_shape)
            if shape == last_tp.vector_shape:
                return changed
            last_tp.vector_shape = shape
            return True

        before = (last_tp.value_type, last_tp.is_array, last_tp.ndim)
        logged = False
        if last_tp.is_array and tp.is_array:
            if last_tp.ndim == tp.ndim:
                assert last_tp.value_type != tp.value_type
                if (issubclass(last_tp.value_type, numbers.Integral)
                        and issubclass(tp.value_type, numbers.Integral)):
                    last_tp.value_type = numbers.Integral
                elif (issubclass(last_tp.value_type, numbers.Real)
                      and issubclass(tp.value_type, numbers.Real)):
                    last_tp.value_type = numbers.Real
                else:
                    logged = self._fallback(
                        "array types", tr, last_tp.value_type.__name__,
                        tp.value_type.__name__)
                    last_tp.value_type = object
            else:
                logged = self._fallback(
                    "array dimensions", tr, last_tp.ndim, tp.ndim)
                self.ret_type = ReturnTypeInfo(object)

        elif not last_tp.is_array and not tp.is_array:
            if (issubclass(last_tp.value_type, numbers.Integral)
                    and issubclass(tp.value_type, numbers.Integral)):
                last_tp.value_type = numbers.Integral
            elif (issubclass(last_tp.value_type, numbers.Real)
                  and issubclass(tp.value_type, numbers.Real)):
                last_tp.value_type = numbers.Real if self._integral_double_exact else object
            else:
                logged = self._fallback(
                    "types", tr, last_tp.value_type.__name__,
                    tp.value_type.__name__)
                last_tp.value_type = object

        else:   # one is an array, the other is not
            def describe(t):
                return (f"{'array of ' if t.is_array else ''}"
                        f"{t.value_type.__name__}")
            logged = self._fallback(
                "array and non-array types", tr, describe(last_tp),
                describe(tp))
            self.ret_type = ReturnTypeInfo(object)

        after = (self.ret_type.value_type, self.ret_type.is_array,
                 self.ret_type.ndim)
        return changed or logged or after != before


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
        self.type_ = (object if isinstance(value, numbers.Integral)
                      and not -(1 << 63) <= value < (1 << 63) else type(value))
        self.mx_class = mx_class
        self.helper = registered_helper(value) if type(value) is FunctionType else None
        from .helper_members import module_helpers
        self.helper_members = module_helpers(value)
        from .helpers import helper_parameters, helper_result_type
        self.helper_parameters = helper_parameters(value) if self.helper is not None else None
        self.helper_result_type = helper_result_type(value) if self.helper is not None else None
        from .helpers import CAPABILITIES
        for capability, (name, _) in CAPABILITIES.items():
            setattr(self, name, helper_capability(value, capability)
                    if self.helper is not None else None)
        self.enum_safe = (isinstance(value, type) and issubclass(value, IntEnum)
                          and type(value) is type(IntEnum)
                          and value.__eq__ is int.__eq__ and value.__ne__ is int.__ne__)
        self.enum = ((value.__module__, value.__qualname__,
                      {name: int.__int__(member) for name, member in value.__members__.items()})
                     if isinstance(value, type) and issubclass(value, IntEnum)
                     and '<locals>' not in value.__qualname__ else None)

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
            name if the class is defined in the ``module`` package,
            otherwise with ``mx_class`` left empty.
        """
        # Compare the top-level component, not a raw prefix: an
        # unrelated package whose name merely extends the model's
        # (e.g. "Model_nomx_cy" against "Model_nomx") would otherwise
        # pass for a model package.
        if value.__class__.__module__.split(".")[0] == module:
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


def _frame_schema(value):
    """The schema of one pandas DataFrame or Series (see ``frame_schema``), or None."""
    modules = {klass.__module__.split('.')[0] for klass in type(value).__mro__}
    if 'pandas' not in modules:
        return None
    names = {klass.__name__ for klass in type(value).__mro__}
    if 'DataFrame' in names:
        kind, columns = 'frame', tuple((label, _column_kind(value[label].dtype) if
                                         list(value.columns).count(label) == 1 else 'obj')
                                        for label in value.columns)
    elif 'Series' in names:
        kind, columns = 'series', ((None, _column_kind(value.dtype)),)
    else:
        return None
    index = value.index
    levels = tuple('int' if getattr(level, 'dtype', None) is not None and level.dtype.kind in 'iu' else 'obj'
                   for level in (index.levels if index.nlevels > 1 else (index,)))
    names = tuple(index.names) if all(isinstance(n, (str, int, type(None))) for n in index.names) else None
    return {'kind': kind, 'levels': levels, 'unique': bool(index.is_unique), 'columns': columns, 'names': names}


def _column_kind(dtype):
    return {'float64': 'f64', 'int64': 'i64'}.get(str(dtype), 'obj')


def frame_schema(traces):
    """T-3: the schema of a traced pandas DataFrame or Series result, or None.

    ``{'kind': 'frame'|'series', 'levels': ('int'|'obj', ...), 'unique': bool, 'columns': ((label,
    'f64'|'i64'|'obj'), ...)}`` (a Series has one column, labelled None). The compiler lowers reads onto a
    PreparedTable from it (frame_reads.py); the frame itself is never replaced. None when a traced value is not
    a pandas frame or Series, or when the traced values disagree (a column kind or level kind differs; a
    column present in one only is dropped, so its reads stay pandas).
    """
    schema = None
    for trace in traces or ():
        found = _frame_schema(trace.ret_val)
        if found is None:
            return None
        if schema is None:
            schema = found
            continue
        if found['kind'] != schema['kind'] or found['levels'] != schema['levels']:
            return None
        mine = dict(found['columns'])
        schema = dict(schema, unique=schema['unique'] and found['unique'],
                      names=schema.get('names') if schema.get('names') == found.get('names') else None,
                      columns=tuple((label, kind) for label, kind in schema['columns'] if mine.get(label) == kind))
    return schema


# M1 (W1.1): schema of a traced helper Table result, through the helper's contract.
TABLE_TYPES = ('modelx_helpers.immutable_inputs.Table',)


def table_schema(traces):
    """Key-level and column kinds of a traced helper Table result, or None.

    Read through the live helper's ``table_schema(t)`` (W1.H contract, M1 design section 3),
    which returns the kinds without expanding compact storage. The tracer never reads table
    storage itself. None when the result is not a helper Table, when the helper does not
    provide the contract yet (round-2 helper), or when the sampled tables disagree.
    """
    if not traces:
        return None
    owner = type(traces[0].ret_val)
    if f'{owner.__module__}.{owner.__qualname__}' not in TABLE_TYPES:
        return None
    contract = getattr(sys.modules.get(owner.__module__), 'table_schema', None)
    if not callable(contract):
        return None     # DEPENDENCY W1.H: modelx_helpers.immutable_inputs.table_schema
    schema, seen = None, set()
    for trace in traces:
        value = trace.ret_val
        if type(value) is not owner:
            return None
        if id(value) in seen:
            continue
        seen.add(id(value))
        try:
            found = contract(value)
        except (TypeError, ValueError):
            return None
        if schema is None:
            schema = found
        elif found != schema:
            # kinds must agree; a differing row count (W1.H 'rows') is kept as None
            if not (isinstance(found, dict) and isinstance(schema, dict)
                    and {k: v for k, v in found.items() if k != 'rows'}
                    == {k: v for k, v in schema.items() if k != 'rows'}):
                return None
            schema = {**schema, 'rows': None}
    return schema


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
            else:
                trace.raised = True     # unwound by an exception: no return value (D111)

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

        # Filter by module name here.  The function is resolved once per
        # code object: an uncached cells is traced on every call.
        func = self._get_func(frame)
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


def is_uncached_cells_code(code: CodeType) -> bool:
    """Whether a code object is the public method of an uncached cells.

    modelx exports a cached cells as a ``_f_<name>`` formula method plus
    a public ``<name>`` method that looks up the cache and calls
    ``self._f_<name>`` on a miss.  An uncached cells has no ``_f_``
    method: its public ``<name>`` method is the formula itself.  So a
    method with a user-defined name whose code does not refer to
    ``_f_<name>`` is an uncached cells, provided it is a method defined
    directly in a space class (first parameter ``self``; on Python 3.11
    and later also a qualified name ``_c_<Space>.<name>``) and not a
    function nested in a formula, nor a lambda or generator expression,
    whose names are not identifiers.  A formula cannot refer to
    ``_f_<name>`` itself, because modelx names never start with an
    underscore.  Before Python 3.11 a module-level function whose first
    parameter is ``self`` passes these tests too; exported
    ``_mx_classes.py`` and ``_mx_model.py`` modules define none.

    Parameters
    ----------
    code : types.CodeType
        A code object defined in an ``_mx_classes.py`` or
        ``_mx_model.py`` module.
    """
    # The exported collection adapter also has no _f_ method.
    if "_mx_collect_cache" in code.co_names and "collect_from_space" in code.co_names:
        return False
    name = code.co_name
    qualname = getattr(code, "co_qualname", None)   # Python 3.11+
    if qualname is not None:
        # a method defined directly in a space class
        parts = qualname.split(".")
        if len(parts) != 2 or parts[0][:len(SPACE_PREF)] != SPACE_PREF:
            return False
    return bool(
        name.isidentifier()
        and is_user_defined(name)
        and not code.co_flags & inspect.CO_NESTED
        and code.co_argcount >= 1
        and code.co_varnames[0] == MX_SELF
        and FORMULA_PREF + name not in code.co_names
    )


class MxCodeFilter:
    """Code filter selecting modelx model code objects to trace.

    Accepts only code objects whose file name ends with
    ``_mx_model.py`` or ``_mx_classes.py`` (only a filename-suffix
    comparison is performed, for speed), and whose function is a
    formula (``_f_*``), ``_mx_assign_refs``, ``__call__``, or the
    public method of an uncached cells
    (see :func:`is_uncached_cells_code`).  The public methods of cached
    cells are not traced: they are called on every cache hit, and their
    ``_f_`` formulas already carry the types.
    """

    def __init__(self):
        # code object -> is_uncached_cells_code(code), as computed once
        self._public = {}

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
            elif code.co_name in ("__call__", "_mx_assign_params", "_mx_copy_params"):
                return True
            else:
                result = self._public.get(code)
                if result is None:
                    result = self._public[code] = is_uncached_cells_code(code)
                return result

        return False


def integer_magnitude(value) -> int:
    """The largest |integer| in ``value``: an int (not a bool), or the integer columns of a NumPy
    array, pandas Series or DataFrame (an input table read by the model); 0 for anything else."""
    if type(value) is int or (isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))):
        return abs(int(value))
    arrays = ()
    if isinstance(value, np.ndarray):
        arrays = (value,)
    elif type(value).__module__.startswith('pandas.'):
        if hasattr(value, 'dtypes') and hasattr(value, 'columns'):          # DataFrame
            arrays = [value[c].to_numpy() for c, t in zip(value.columns, value.dtypes)
                      if getattr(t, 'kind', '') in 'iu']
            index = getattr(value, 'index', None)
            if index is not None and getattr(index.dtype, 'kind', '') in 'iu':
                arrays.append(index.to_numpy())
        elif hasattr(value, 'dtype') and hasattr(value, 'to_numpy'):         # Series or Index
            arrays = [value.to_numpy()]
            index = getattr(value, 'index', None)
            if index is not None and getattr(index.dtype, 'kind', '') in 'iu':
                arrays.append(index.to_numpy())
    largest = 0
    for array in arrays:
        if getattr(array.dtype, 'kind', '') in 'iu' and array.size:
            largest = max(largest, int(max(abs(int(array.max())), abs(int(array.min())))))
    return largest


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
        # funcname -> (argument types and values, integral minima, return
        # type merger) for uncached traces: see log
        self._kept = {}
        # funcname -> the same state for cached traces, with calls per receiver, the result
        # types seen and the largest scalar result: see log
        self._cached = {}
        self.cells_info = {}  # funcname -> MethodTypeInfo
        self.ref_info = {}
        self.modules = []
        self.param_info = {}  # class name -> {param: RuntimeParamInfo}
        self._assigned_params = {}
        # integer guards: the largest |integer| traced (Cell arguments and results, References,
        # ItemSpace parameters, integer columns of input tables) and where it was seen
        self.max_integer = 0
        self.max_integer_at = None

    def log(self, trace: CallTrace) -> None:
        """Log a single call trace.

        Retain only the traces that contribute an argument or return
        observation. Cached formula calls also count their calls per
        receiver, for cache sizing. A cached trace is kept as well when
        its result is not a plain scalar (arrays, frames, tables and
        other objects are read trace by trace), when its result type is
        new, or when it raises the largest scalar result, so that
        :class:`RuntimeCellsInfo` derives from the kept traces exactly
        what it derives from all of them.
        """
        name = trace.func.__name__
        self._integer_magnitudes(trace.funcname, (*trace.arg_vals.values(), getattr(trace, 'ret_val', None)))
        if name == "_mx_assign_params":
            values = iter(trace.arg_vals.items())
            _, target = next(values)
            self._record_params(target, values)
        elif name == "_mx_copy_params":
            source, target = trace.arg_vals.values()
            cls = type(source)
            params = getattr(sys.modules[cls.__module__], SPACE_PARAMS + "_" + cls.__name__[len(SPACE_PREF):])
            self._record_params(target, ((param, getattr(source, param)) for param in params))
        elif name.startswith(FORMULA_PREF) or name in (MX_ASSIGN_REFS, "__call__"):
            name = trace.funcname
            state = self._cached.get(name)
            if state is None:
                state = self._cached[name] = {}, {}, _ReturnTypeMerger(name, log=False), {}, set(), [None]
            types, minimums, merger, receivers, ret_types, largest = state
            _count_receiver(receivers, trace)
            value, raised = getattr(trace, 'ret_val', None), getattr(trace, 'raised', False)
            keep = _add_arg_vals(types, minimums, trace)
            if type(value) not in ret_types:
                ret_types.add(type(value))
                keep = True
            if type(value) not in _PLAIN_RETURNS:
                keep = True
            elif not raised:
                keep = merger.add(trace) or keep
                if type(value) not in (bool, np.bool_, str, type(None)) and value == value:
                    try:
                        value = float(value)
                    except OverflowError:
                        value, keep = None, True
                    if value is not None and (largest[0] is None or value > largest[0]):
                        largest[0], keep = value, True
            if keep:
                self._traces.setdefault(name, []).append(trace)
        else:
            name = trace.funcname
            state = self._kept.get(name)
            if state is None:
                state = self._kept[name] = {}, {}, _ReturnTypeMerger(name, log=False)
            types, minimums, merger = state
            args_changed = _add_arg_vals(types, minimums, trace)
            ret_changed = False if getattr(trace, 'raised', False) else merger.add(trace)
            if args_changed or ret_changed:
                self._traces.setdefault(name, []).append(trace)

    def _integer_magnitudes(self, where, values):
        for value in values:
            magnitude = integer_magnitude(value)
            if magnitude > self.max_integer:
                self.max_integer, self.max_integer_at = magnitude, where

    def _record_params(self, target, values):
        cls = type(target)
        params = self._assigned_params.setdefault(cls.__module__ + "." + cls.__name__, {})
        for name, value in values:
            if name not in params:
                params[name] = RuntimeParamInfo.init_mxobj(value, self.module)

    def flush(self) -> None:
        """Aggregate the accumulated traces into runtime info objects.

        Traces of ``_mx_assign_refs`` provide refs: each user-defined
        attribute of the traced ``self`` (taken from the first trace
        only) becomes a :class:`RuntimeRefInfo` in ``ref_info``.  All
        other traces (formula methods, space ``__call__`` methods and
        the public methods of uncached cells admitted by the code
        filter) are treated as cells: each group builds a
        :class:`RuntimeCellsInfo` in ``cells_info`` and registers its
        module in ``modules``.  A public method is kept only if it is
        defined directly in a space class (qualified name
        ``_c_<Space>.<name>``).  The raw traces are then cleared,
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
                        self._integer_magnitudes(fqname, (val,))
                        self.ref_info[fqname] = RuntimeRefInfo.init_mxobj(val, self.module)
            else:
                if is_user_defined(name_split[-1]):
                    # The public method of an uncached cells
                    qualname = tr0.func.__qualname__.split(".")
                    if not (len(qualname) == 2
                            and qualname[0][:len(SPACE_PREF)] == SPACE_PREF):
                        continue
                # Extract cells
                self.cells_info[k] = info = RuntimeCellsInfo(v, self._cached[k][3] if k in self._cached else None)
                if info.module not in self.modules:
                    self.modules.append(info.module)

        self._traces.clear()
        self._kept.clear()
        self._cached.clear()
        self._get_params()
        # Collection receivers are absent from the retained ItemSpace tree.
        for cls, assigned in self._assigned_params.items():
            params = self.param_info.setdefault(cls, {})
            for name, info in assigned.items():
                params.setdefault(name, info)
        self._assigned_params.clear()

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
    the ``with`` block. The previous profile function is restored on exit.
    A successful trace is then flushed to the logger.

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
