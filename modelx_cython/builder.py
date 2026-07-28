# Copyright (c) 2023-2025 Fumito Hamamura <fumito.ham@gmail.com>
"""Combine lexical, runtime, and spec information into build metadata.

This module merges the three sources of information gathered by the
mx2cy pipeline before code generation:

* the lexical structure of each model module, parsed by
  :class:`~modelx_cython.parser.ModuleVisitor`;
* the runtime type samples recorded by
  :class:`~modelx_cython.tracer.MxCallTraceLogger` during the sample
  run;
* the user's translation spec
  (:class:`~modelx_cython.config.TransSpec`).

The result is a tree of :class:`ModuleInfo` -> :class:`ClassInfo` ->
:class:`CombinedCellsInfo` / :class:`CombinedRefInfo` objects that the
transformer queries to emit Cython pure-Python-mode annotations and
.pxd declarations.
"""
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

from typing import Union, Sequence, Mapping, Dict, Optional, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from modelx_cython.usage import UsageVerdict

try:
    from types import NoneType
except ImportError:  # Python -3.9
    NoneType = type(None)

from functools import cached_property

from modelx_cython.typedefs import get_type_expr
from modelx_cython.config import TransSpec
from modelx_cython.tracer import RuntimeCellsInfo, MxCallTraceLogger
from modelx_cython.parser import ModuleVisitor, LexicalCellsInfo, LexicalRefInfo

from modelx_cython.consts import (
    FORMULA_PREF,
    SPACE_PREF,
    MODULE_PREF,
    CY_MOD
)
from modelx_cython.typedefs import str_to_type, normalize_type

_logger = logging.getLogger(__name__)

# Policy for array-returning cells with no call sites inside the model
# (consumed only by external user scripts, which the usage analysis cannot
# see). True keeps the const-memoryview return type; False falls back to
# object.
MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS: bool = True
"""bool: Return-type policy for arrays used only outside the model.

When usage analysis finds no call site inside the model for an
array-returning cells, ``True`` keeps the const-memoryview return type
and ``False`` falls back to ``object``.  Consulted by
:attr:`CombinedCellsInfo.use_memoryview`.
"""


class CombinedCellsInfo(LexicalCellsInfo):
    """Merged lexical, runtime, and spec information for one cells.

    Extends :class:`~modelx_cython.parser.LexicalCellsInfo` with the
    runtime type samples traced for the cells (if any) and the
    per-cells entries of the user's spec, and derives from them the
    Cython type expressions for the cells' parameters and return
    value.

    Attributes
    ----------
    parent : ClassInfo
        The :class:`ClassInfo` of the space class owning this cells.
    usage : UsageVerdict, optional
        Static classification of how the cells' return value is
        consumed.  ``None`` at construction time; assigned later by
        :func:`modelx_cython.usage.apply_verdicts`, between the build
        and transform phases.
    _rt : RuntimeCellsInfo
        Runtime type information traced for this cells, or ``None``
        when the sample run never called it.
    _spec : dict
        The per-cells spec dict taken from the user's spec file.
    _spec_ret_t : str
        Value of the spec key ``"return_type"``, or ``""`` if absent.
    _force_memoryview : bool
        True when the spec sets ``"return_type": "memoryview"``.

    Raises
    ------
    ValueError
        If the spec requests ``"return_type": "memoryview"`` but the
        traced return value is not a real-valued numpy array with at
        least one dimension.  If no type information was sampled at
        all, the request is ignored with a warning instead.
    """

    parent: 'ClassInfo'
    _rt: RuntimeCellsInfo
    _spec: dict
    _spec_ret_t: str
    usage: 'Optional[UsageVerdict]' = None  # set by usage.apply_verdicts
    # True if some formula calls this cells with keyword arguments, which
    # C-level (cpdef) calls do not support; set in cli.main_handler
    called_with_kwargs: bool = False

    def __init__(self, cls_info, lx_info, rt_info, spec) -> None:
        """Merge lexical, runtime and spec info, validating the spec's
        ``"return_type": "memoryview"`` request when present.
        """
        super().__init__(
            lx_info.module, lx_info.cls, lx_info.name, lx_info.params,
            lx_info.params_with_defaults
        )
        self.parent = cls_info
        visitor = getattr(cls_info, "visitor", None)
        if visitor is not None:
            # True if a _f_<name> formula method exists in the source.
            # Cells that modelx exports as uncached have no _f_ method:
            # their public method is the formula itself.
            self.has_formula_def = lx_info.name in visitor.formula_defs.get(
                lx_info.cls, ())
            # Closures (nested defs, lambdas, generator expressions, yield)
            # cannot be compiled inside cpdef (ccall) functions, so public
            # methods containing them are left as plain Python methods and
            # omitted from the pxd. cdef (cfunc) formula methods support
            # closures but not yield.
            self.body_has_closure = lx_info.name in visitor.closure_funcs.get(
                lx_info.cls, ())
            self.formula_is_generator = (
                FORMULA_PREF + lx_info.name
                in visitor.generator_funcs.get(lx_info.cls, ()))
        else:   # constructed without a ClassInfo (tests)
            self.has_formula_def = True
            self.formula_is_generator = False
            self.body_has_closure = False
        self._rt = rt_info
        self._spec = spec
        self._spec_ret_t = spec.get(TransSpec.RET_T, "")
        self._force_memoryview = self._spec_ret_t == TransSpec.RET_MEMORYVIEW
        if self._force_memoryview:
            if self.has_typeinfo():
                if not (self.is_array_returned and self.is_real_value
                        and self.ret_ndim >= 1):
                    raise ValueError(
                        f"invalid value for spec '{TransSpec.RET_T}': "
                        f"'{TransSpec.RET_MEMORYVIEW}' requires a real-valued "
                        f"numpy array return, but '{self.fqname}' does not "
                        "return one")
            else:
                _logger.warning(
                    f"spec '{TransSpec.RET_T}': '{TransSpec.RET_MEMORYVIEW}' "
                    f"for '{self.fqname}' is ignored because no type "
                    "information was sampled")

    @cached_property
    def norm_type(self) -> type:
        """Normalized Python type of the cells' return value.

        When the spec supplies a ``"return_type"`` (other than
        ``"memoryview"``), it is looked up in
        :data:`~modelx_cython.typedefs.str_to_type`; an unknown name
        raises ``ValueError``.  Otherwise the traced return value type
        is normalized with
        :func:`~modelx_cython.typedefs.normalize_type`.  Falls back to
        ``object`` when no type information was sampled.
        """
        if self.has_typeinfo():
            if self._spec_ret_t and not self._force_memoryview:
                if self._spec_ret_t in str_to_type:
                    return str_to_type[self._spec_ret_t]
                else:
                    raise ValueError(f"invalid value for spec '{TransSpec.RET_T}': {self._spec_ret_t}")
            else:
                return normalize_type(self._rt.ret_type.value_type)
        else:
            return object

    @cached_property
    def is_real_value(self):
        """Whether the normalized return type is numeric.

        True when :attr:`norm_type` is a subclass of
        :class:`numbers.Real`, which includes ``bool`` and integral
        types.  Requires type information.
        """
        assert self.has_typeinfo()
        return issubclass(self.norm_type, numbers.Real)

    @cached_property
    def is_array_returned(self):
        """Whether the traced return value is an array.

        Requires type information.
        """
        assert self.has_typeinfo()
        return self._rt.ret_type.is_array

    def has_typeinfo(self):
        """Return True if runtime type information was traced.

        False when the sample run never called this cells.
        """
        return bool(self._rt)

    def has_args(self):
        """Return True if the cells has one or more parameters."""
        return bool(self.params)

    @property
    def ret_ndim(self) -> int:
        """Number of dimensions of the traced return value.

        0 for scalar returns.  Requires type information.
        """
        assert self.has_typeinfo()
        return self._rt.ret_type.ndim

    @property
    def has_spec_rettype(self) -> bool:
        """Whether the spec supplies a ``"return_type"`` for this
        cells.
        """
        return bool(self._spec_ret_t)

    @property
    def use_memoryview(self) -> bool:
        """Whether the array return type is emitted as a memoryview.

        Deliberately a plain (uncached) property: :attr:`usage` is
        assigned after construction, between the build and transform
        phases, so the result may change.  The decision is:

        * ``True`` when the spec forces it with
          ``"return_type": "memoryview"``, or when no usage verdict
          was assigned (legacy behavior);
        * ``False`` when usage analysis could not prove that every
          model-internal use is scalar element access (including
          unresolvable calls that merely share the cells' name);
        * :data:`MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS` when the value
          is consumed only by element access but no call site exists
          inside the model;
        * ``True`` otherwise (element access only, with internal
          uses).
        """
        # A plain property, not cached: `usage` is assigned after
        # construction, between the build and transform phases.
        if self._force_memoryview:
            return True
        if self.usage is None:
            return True     # no analysis info: keep legacy behavior
        if not self.usage.only_element_access:
            return False
        if not self.usage.has_internal_uses:
            return MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS
        return True

    def get_argtype_expr(self, arg: str, c_style=False) -> str:
        """Return the Cython type expression for a parameter.

        Parameters
        ----------
        arg : str
            Parameter name; must have been sampled by the tracer when
            type information exists.
        c_style : bool, default False
            If True, emit a C-style type (e.g. ``long long``) for .pxd
            files; otherwise a pure-Python-mode expression
            (e.g. ``_mx_cy.longlong``).

        Returns
        -------
        str
            The type expression, or ``"object"`` when no type
            information was sampled.
        """
        if self.has_typeinfo():
            assert arg in self._rt.arg_types
            return get_type_expr(self._rt.arg_types[arg], c_style=c_style)
        else:
            return "object"

    def get_rettype_expr(self, c_style=False):
        """Return the Cython type expression for the return value.

        For real-valued array returns, emits a const-qualified typed
        memoryview (e.g. ``const double[:, :]`` in C style,
        ``_mx_cy.const[_mx_cy.double][:, :]`` in pure-Python mode) so
        that read-only arrays, such as those returned by pandas under
        copy-on-write, can be coerced; when :attr:`use_memoryview` is
        False the expression falls back to ``"object"`` instead.
        Non-array returns use the expression for :attr:`norm_type`,
        and cells without type information yield ``"object"``.

        Parameters
        ----------
        c_style : bool, default False
            If True, emit C-style types for .pxd files.
        """

        if self.has_typeinfo():
            typ = get_type_expr(self.norm_type, c_style=c_style)
            if self.is_real_value and self.is_array_returned:
                if not self.use_memoryview:
                    return "object"
                # const element type so that read-only arrays, such as those
                # returned by pandas under copy-on-write, can be coerced
                suffix = "[" + ", ".join(":" * self._rt.ret_type.ndim) + "]"
                if c_style:
                    return "const " + typ + suffix
                else:
                    return f"{CY_MOD}.const[{typ}]" + suffix
            else:
                return typ
        else:
            return "object"

    def is_arg_int(self, arg: str):
        """Return True if the sampled type of parameter ``arg`` is
        integral.
        """
        assert self.has_args() and self.has_typeinfo()
        return issubclass(self._rt.arg_types[arg], numbers.Integral)

    @cached_property
    def is_int_args(self):
        """Whether every parameter's sampled type is integral.

        Requires the cells to have parameters and type information.
        """
        assert self.has_args() and self.has_typeinfo()
        for p in self.params:
            if self.is_arg_int(p):
                continue
            else:
                return False
        return True

    def is_arrayable(self):
        """Return True if results can be cached in a fixed-size C
        array.

        A cells is arrayable when all its parameters are integral and
        it returns a real-valued scalar (not an array).  Requires
        parameters and type information.
        """
        assert self.has_args() and self.has_typeinfo()
        if self.is_int_args and self.is_real_value and not self.is_array_returned:
            return True
        else:
            return False

    def get_array_decl_expr(self, rettype_expr="", c_style=False):
        """Return the C array declaration expression for the cache.

        Appends one ``[size]`` per parameter to the return type
        expression (e.g. ``double[10][20]``), taking sizes from the
        parent class's :attr:`ClassInfo.cells_arg_sizes` entry for
        this cells' parameter tuple.

        Parameters
        ----------
        rettype_expr : str, default ""
            Element type expression; computed with
            :meth:`get_rettype_expr` when empty.
        c_style : bool, default False
            Passed to :meth:`get_rettype_expr` when ``rettype_expr``
            is empty.
        """
        assert self.is_arrayable()
        if not rettype_expr:
            rettype_expr = self.get_rettype_expr(c_style=c_style)

        sizes = self.parent.cells_arg_sizes[tuple(self.params)]
        return rettype_expr + "".join([f"[{str(i)}]" for i in sizes])


class CombinedRefInfo:
    """Merged lexical and runtime information for one ref.

    Describes a ref (a named value held by a space) or a space
    parameter, combining its lexical location with the type sampled at
    runtime.  When the sampled value is an instance of a model space
    class, that class is recorded so the transformer can declare the
    ref with the space class instead of a plain type.

    Attributes
    ----------
    module : str
        Fully-qualified name of the module defining the owning class.
    cls : str
        Name of the space class that owns the ref.
    name : str
        Name of the ref.
    type_ : type
        Python type of the sampled value, or ``None`` when the ref was
        never sampled.
    mx_class : str
        Fully-qualified class name of the model space held by the ref,
        or ``''`` when the value is not a model space.
    decl_type_expr : str
        Class path used to declare the ref: relative to ``module``
        when the class's fully-qualified name starts with the
        module's name (in practice, classes defined in submodules),
        fully qualified otherwise; ``''`` for non-space refs.
    is_relative : bool
        True when ``decl_type_expr`` is relative to ``module``.
    """

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
        """Initialize from lexical identity and optional runtime info.

        When ``rt_info`` is falsy, only ``module``, ``cls`` and
        ``name`` are set; the class-level defaults remain in effect.
        """
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

    def get_type_expr(self, c_style=False):
        """Return the type expression used to declare this ref.

        Returns ``decl_type_expr`` when the ref holds a model space;
        otherwise maps the sampled Python type through
        :func:`modelx_cython.typedefs.get_type_expr`.

        Parameters
        ----------
        c_style : bool, default False
            If True, emit a C-style type for .pxd files.
        """
        if self.decl_type_expr:
            return self.decl_type_expr
        elif self.type_ is None:    # no runtime info sampled
            return "object"
        else:
            return get_type_expr(self.type_, c_style=c_style)


class ClassInfo:
    """Combined information for one space class in a model module.

    Aggregates, for a single ``_c_``-prefixed class, the cells, refs,
    child spaces and space parameters found lexically by the module
    visitor, joined with the runtime information recorded by the trace
    logger and with the user's spec.

    Attributes
    ----------
    name : str
        The space class name (with the ``_c_`` prefix).
    module : ModuleInfo
        The :class:`ModuleInfo` of the module defining the class.
    visitor : ModuleVisitor
        The module's lexical visitor (shared with ``module``).
    logger : MxCallTraceLogger
        The runtime trace logger (shared with ``module``).
    cells : dict
        Maps cells name to :class:`CombinedCellsInfo`.
    refs : dict
        Maps ref name to :class:`CombinedRefInfo`; also holds the
        space parameters added by :meth:`_add_space_params`.
    spaces : list
        Names of the child spaces assigned in the class's
        ``__init__``.
    params : dict
        Annotated only; never assigned.  Space parameters are stored
        in ``refs`` instead.
    _cells_max_args : dict
        Maps a tuple of parameter names to maximum observed argument
        values.  Currently these are the maxima of the first traced
        cells seen for each parameter tuple; larger values observed
        for later cells sharing the tuple are compared but not
        merged back.
    _max_arg_cells : dict
        Maps a parameter tuple to ``{param: cells fqname}`` naming,
        for logging, the cells that produced the largest value
        compared so far for each parameter (which, for later cells,
        may not be the value kept in ``_cells_max_args``).
    """

    name: str
    module: 'ModuleInfo'
    visitor: ModuleVisitor
    logger: MxCallTraceLogger
    cells:  dict  # name -> CombinedCellsInfo
    refs:   dict  # name -> CombinedRefInfo
    spaces: list
    params: dict  # name -> CombinedRefInfo
    _cells_max_args: Dict[Tuple[str], Tuple[int]]
    _max_arg_cells: Dict[Tuple[str], Dict[str, str]]  # {(arg,) : {arg: fqname}}

    def __init__(self, name, module):
        """Build the cells, space, ref and parameter tables for the
        class named ``name`` in ``module``.
        """
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
        """Create a CombinedCellsInfo per lexical cells and record
        observed maximum argument values per parameter tuple (see the
        ``_cells_max_args`` attribute for the exact semantics).
        """
        # .get: model classes and container-only spaces have no cells
        for name, lx_info in self.visitor.cells_info.get(self.name, {}).items():
            rt_info = self.logger.cells_info.get(lx_info.fqname, None)
            self.cells[name] = CombinedCellsInfo(
                self,
                lx_info, rt_info,
                self.module.spec.get_spec(self.fqname).get(TransSpec.CELLS, {}).get(name, {})
            )
            if rt_info:
                args = tuple(rt_info.max_args)
                maxes = tuple(rt_info.max_args.values())
                if args not in self._cells_max_args:
                    self._cells_max_args[args] = maxes
                    self._max_arg_cells[args] = {k: lx_info.fqname for k in args}
                else:
                    d = dict(zip(args, self._cells_max_args[args]))
                    for k, v in rt_info.max_args.items():
                        if v > d[k]:
                            d[k] = v
                            self._max_arg_cells[args][k] = lx_info.fqname
                    self._cells_max_args[args] = tuple(d.values())

    def _init_spaces(self):
        """Copy the child space names found by the visitor."""
        self.spaces.extend(self.visitor.spaces.get(self.name, []))

    def _init_refs(self):
        """Create a CombinedRefInfo per lexical ref, joined with any
        traced runtime value info.
        """
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
        """Add the space's traced parameters to ``refs`` as
        CombinedRefInfo entries.
        """
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
        """Fully-qualified class name:
        ``<module fqname>.<class name>``.
        """
        return self.module.fqname + "." + self.name

    @cached_property
    def cells_arg_sizes(self) -> Mapping[Tuple[str], Tuple[int]]:
        """C array cache sizes per cells parameter tuple.

        Returns
        -------
        Mapping[Tuple[str], Tuple[int]]
            Maps a tuple of parameter names to the array size for each
            parameter.

        Notes
        -----
        Sizes come first from the spec: the ``"cells_param_size"``
        entry for this class, or, failing that, the deprecated
        ``"cells_params"`` entry with its nested ``"size"`` keys.
        Scalar (single-parameter) spec keys are normalized to
        1-tuples.  Each spec size is then overridden by the maximum
        argument value observed at runtime plus one whenever that is
        larger (an info message is logged); parameter tuples absent
        from the spec get their sizes solely from the observed maxima
        plus one.  Note that for a parameter tuple shared by several
        traced cells, the observed maxima currently come from the
        first traced cells only (see ``_cells_max_args``).
        """
        # params = self.module.spec.get_spec(self.fqname).get(TransSpec.CELLS_PARAMS, {})

        sizes = {}
        if TransSpec.CELLS_PARAM_SIZE in self.module.spec.get_spec(self.fqname):
            params = self.module.spec.get_spec(self.fqname)[TransSpec.CELLS_PARAM_SIZE]
            # Tuplize 1-arg
            for k, v in params.items():
                if isinstance(k, tuple):
                    sizes[k] = v
                else:
                    sizes[(k,)] = (v,)

        elif TransSpec.CELLS_PARAMS in self.module.spec.get_spec(self.fqname):  # deprecated
            params = self.module.spec.get_spec(self.fqname)[TransSpec.CELLS_PARAMS]
            # Tuplize 1-arg
            for k, v in params.items():
                if TransSpec.SIZE in v:
                    if isinstance(k, tuple):
                        sizes[k] = v[TransSpec.SIZE]
                    else:
                        sizes[(k,)] = (v[TransSpec.SIZE],)

        for args, maxes in self._cells_max_args.items():
            if args in sizes:
                d = dict(zip(args, sizes[args]))
                for k, v in zip(args, maxes):
                    if v + 1 > d[k]:
                        _logger.info(f"Specified max size of {d[k]} for cells parameter {k} in {self.name} is replaced by {v + 1} from {self._max_arg_cells[args][k]}")
                        d[k] = v + 1
                sizes[args] = tuple(d.values())
            else:
                sizes[args] = tuple(i + 1 for i in maxes)

        return sizes


class ModuleInfo:
    """Combined information for one model module.

    Ties together the lexical visitor, the runtime trace logger and
    the user's spec for a single module, and builds a
    :class:`ClassInfo` for each space class the visitor found.

    Attributes
    ----------
    fqname : str
        Fully-qualified (dotted) module name.
    visitor : ModuleVisitor
        Lexical information parsed from the module source.
    logger : MxCallTraceLogger
        Runtime type information from the sample run.
    spec : TransSpec
        The user's translation spec.
    classes : dict
        Maps space class name to :class:`ClassInfo`.
    """

    fqname: str
    visitor: ModuleVisitor
    logger: MxCallTraceLogger
    spec: TransSpec
    classes: dict   # class name -> ClassInfo

    def __init__(self, fqname: str, visitor: ModuleVisitor, logger: MxCallTraceLogger,
                 spec: TransSpec):
        """Store the inputs and build a ClassInfo per space class."""

        self.fqname = fqname
        self.visitor = visitor
        self.logger = logger
        self.spec = spec
        self.classes = {}
        self._init_classes()

    def _init_classes(self):
        """Create a ClassInfo for every class found by the visitor."""
        for c in self.visitor.classes:
            self.classes[c] = ClassInfo(c, self)

    @cached_property
    def cimports(self):
        """Modules that this module's .pxd file must cimport.

        Collects, in first-seen order without duplicates, the module
        part of every ref's absolute (non-relative) declared class
        path.
        """
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
        """Child module names for space classes with child spaces.

        For each class in this module that assigns child spaces,
        returns the corresponding ``_m_``-prefixed module name (the
        ``_c_`` class prefix replaced with ``_m_``).
        """
        result = []
        for cls in self.visitor.spaces:
            result.append(MODULE_PREF + cls[len(SPACE_PREF):])    # replace _c_ with _m_
        return result


