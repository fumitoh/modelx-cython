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

import libcst as cst

from modelx_cython.typedefs import (
    get_type_expr, CY_BOOL_T, CY_FLOAT_T, CY_INT_T)
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


def _unsigned(expr: cst.BaseExpression) -> cst.BaseExpression:
    """``expr`` without its unary plus and minus signs."""
    while (isinstance(expr, cst.UnaryOperation)
           and isinstance(expr.operator, (cst.Plus, cst.Minus))):
        expr = expr.expression
    return expr


def _is_str_literal(expr: cst.BaseExpression) -> bool:
    """Whether ``expr`` is a ``str`` literal: not bytes, not an
    f-string."""
    if isinstance(expr, cst.SimpleString):
        return "b" not in expr.prefix.lower()
    if isinstance(expr, cst.ConcatenatedString):
        return _is_str_literal(expr.left) and _is_str_literal(expr.right)
    return False


def default_fits(default: cst.BaseExpression, ctype: str) -> bool:
    """Whether a parameter default can be declared with C type ``ctype``.

    Only a literal of the type fits a C type: an integer literal, with
    an optional sign, a ``long long``; an integer or float literal a
    ``double``; ``True`` or ``False`` a ``bint``; and a string literal
    or ``None`` a ``str``.  Anything fits ``object``.  This is stricter
    than Cython, which also takes, for example, ``True`` for a ``long
    long`` or a bytes literal for a ``double``, so that the compiled
    default is the exported one.
    """
    if ctype == CY_INT_T:
        return isinstance(_unsigned(default), cst.Integer)
    elif ctype == CY_FLOAT_T:
        return isinstance(_unsigned(default), (cst.Integer, cst.Float))
    elif ctype == CY_BOOL_T:
        return (isinstance(default, cst.Name)
                and default.value in ("True", "False"))
    elif ctype == "str":
        return (_is_str_literal(default)
                or (isinstance(default, cst.Name) and default.value == "None"))
    else:
        return ctype == "object"


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
    _spec_param_t : dict
        Value of the spec key ``"param_type"``: parameter names mapped
        to type names of :data:`~modelx_cython.typedefs.str_to_type`
        that override the traced parameter types.  Empty if absent.
    param_rebinds : dict
        What the formula binds again to each parameter it rebinds, as
        collected by :func:`~modelx_cython.parser.collect_param_rebinds`
        from the ``_f_`` method, or from the public method of an
        uncached cells.  Empty when constructed without a
        :class:`ClassInfo`.
    _object_params : dict
        Parameters typed ``object`` whatever their traced or spec type,
        mapped to the reason: the signature and the formula both take
        them as Python objects.
    _formula_object_params : dict
        Parameters typed ``object`` in the ``_f_`` formula only, mapped
        to the reason; the public method of the cached cells, and with
        it the cache, keeps their type.  See
        :meth:`demote_rebound_param`.
    called_with_kwargs : bool
        True if a call with keyword arguments anywhere in the model
        names a function matching this cells' name.  Cython C-level
        (cpdef) calls are positional-only, so the public method then
        stays a plain Python method.  ``False`` at construction time;
        set by :func:`modelx_cython.cli.main_handler` after all
        modules are parsed.
    has_formula_def : bool
        True if a ``_f_<name>`` formula method exists in the source.
        Cells that modelx exports as uncached have no ``_f_`` method:
        their public method is the formula itself, and no cache
        storage is generated for them.
    body_has_closure : bool
        True if the public method's body contains a construct Cython
        compiles as a closure (nested function, lambda, generator
        expression or yield), which cpdef functions do not support;
        the method then stays a plain Python method.
    formula_is_generator : bool
        True if the ``_f_`` formula method contains a ``yield``,
        which neither cdef nor cpdef functions support; the formula
        then stays a plain Python method.

    Raises
    ------
    ValueError
        If the spec requests ``"return_type": "memoryview"`` but the
        traced return value is not a real-valued numpy array with at
        least one dimension.  If no type information was sampled at
        all, the request is ignored with a warning instead.  Also if
        the spec's ``"param_type"`` is not a dict, names a parameter
        the cells does not have, or gives a type name not in
        :data:`~modelx_cython.typedefs.str_to_type`.
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
            lx_info.params_with_defaults,
            getattr(lx_info, "param_defaults", None),
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
            formula = (FORMULA_PREF + lx_info.name if self.has_formula_def
                       else lx_info.name)
            self.param_rebinds = getattr(visitor, "param_rebinds", {}).get(
                lx_info.cls, {}).get(formula, {})
        else:   # constructed without a ClassInfo (tests)
            self.has_formula_def = True
            self.formula_is_generator = False
            self.body_has_closure = False
            self.param_rebinds = {}
        self._rt = rt_info
        self._spec = spec
        self._spec_ret_t = spec.get(TransSpec.RET_T, "")
        self._force_memoryview = self._spec_ret_t == TransSpec.RET_MEMORYVIEW
        self._spec_param_t = self._init_spec_param_t(
            spec.get(TransSpec.PARAM_T, {}))
        self._object_params = {}
        self._formula_object_params = {}
        if self.has_typeinfo():
            self._check_param_defaults()
        if self._force_memoryview:
            if self.has_typeinfo():
                if not (self.is_array_returned and self.is_real_value
                        and self.ret_ndim >= 1):
                    raise ValueError(
                        f"invalid value for spec '{TransSpec.RET_T}': "
                        f"'{TransSpec.RET_MEMORYVIEW}' requires a real-valued "
                        f"numpy array return, but '{self.formula_fqname}' does not "
                        "return one")
            else:
                _logger.warning(
                    f"spec '{TransSpec.RET_T}': '{TransSpec.RET_MEMORYVIEW}' "
                    f"for '{self.formula_fqname}' is ignored because no type "
                    "information was sampled")

    def _init_spec_param_t(self, param_t) -> Dict[str, type]:
        """Validate the spec's ``"param_type"`` and map it to types.

        Like a spec return type, it takes effect only when the sample
        run produced type information for the cells; otherwise it is
        ignored with a warning, and the parameters stay ``object``.
        """
        key = TransSpec.PARAM_T
        if not isinstance(param_t, dict):
            raise ValueError(
                f"invalid value for spec '{key}' of '{self.formula_fqname}': "
                f"expected a dict of parameter names to type names, "
                f"got {param_t!r}")
        result = {}
        for param, type_name in param_t.items():
            if param not in self.params:
                raise ValueError(
                    f"invalid spec '{key}' for '{self.formula_fqname}': it has no "
                    f"parameter named {param!r} (parameters: "
                    f"{', '.join(self.params) or 'none'})")
            if type_name not in str_to_type:
                raise ValueError(
                    f"invalid spec '{key}' for parameter {param!r} of "
                    f"'{self.formula_fqname}': {type_name!r} is not one of "
                    f"{', '.join(repr(k) for k in str_to_type)}")
            result[param] = str_to_type[type_name]
        if result and not self.has_typeinfo():
            _logger.warning(
                f"spec '{key}' for '{self.formula_fqname}' is ignored because no "
                "type information was sampled")
        return result

    def _given_arg_type(self, arg: str) -> type:
        """Type of parameter ``arg`` before any fallback: from the
        spec's ``"param_type"`` if it names ``arg``, otherwise as
        traced."""
        if arg in self._spec_param_t:
            return self._spec_param_t[arg]
        assert arg in self._rt.arg_types
        return self._rt.arg_types[arg]

    def arg_type(self, arg: str, formula=False) -> type:
        """Type of parameter ``arg``: from the spec's
        ``"param_type"`` if it names ``arg``, otherwise as traced,
        unless it falls back to ``object`` because its default value
        or, in the formula, a value bound to it is not of that type
        (see :meth:`_check_param_defaults` and
        :meth:`demote_rebound_param`).

        Requires type information.

        Parameters
        ----------
        arg : str
            Parameter name.
        formula : bool, default False
            True for the type in the method that holds the formula:
            the ``_f_`` method of a cached cells, or the public method
            of an uncached one.  False for the type in the public
            method, which also sizes the cache.
        """
        assert self.has_typeinfo()
        if arg in self._object_params:
            return object
        if formula and arg in self._formula_object_params:
            return object
        return self._given_arg_type(arg)

    @property
    def formula_fqname(self) -> str:
        """Fully-qualified name of the method that holds the formula:
        :attr:`fqname` (``..._f_<name>``) for a cached cells or a
        special method, :attr:`method_fqname` for an uncached cells,
        which has no ``_f_`` method."""
        if self.has_formula_def or self.is_special():
            return self.fqname
        return self.method_fqname

    def _object_param(self, arg, reason, formula_only=False):
        """Type parameter ``arg`` ``object``, logging ``reason``: in the
        formula only if ``formula_only``, else everywhere."""
        given = get_type_expr(self._given_arg_type(arg), c_style=True)
        target = (self._formula_object_params if formula_only
                  else self._object_params)
        target[arg] = reason
        # is_int_args is cached and depends on the parameter types
        self.__dict__.pop("is_int_args", None)
        if formula_only:
            where, method = " in the formula", self.formula_fqname
        else:
            where, method = "", self.method_fqname
        if arg in self._spec_param_t:
            _logger.warning(
                f"spec '{TransSpec.PARAM_T}' for parameter '{arg}' of "
                f"{method} is not applied{where}, and the parameter is "
                f"typed object: {reason}")
        else:
            _logger.info(
                f"parameter '{arg}' of {method} is typed object{where} "
                f"rather than {given}: {reason}")

    def _check_param_defaults(self):
        """Type ``object`` the parameters whose default value does not
        fit their C type.

        Cython rejects a C-typed parameter whose default is not of its
        type: ``None`` for a ``double`` ("Signature not compatible with
        previous declaration"), ``0.5`` for a ``long long`` ("Cannot
        assign type 'double' to 'long long'").  A default is taken to
        fit only when it is a literal of the type: an integer literal,
        with an optional sign, for ``long long``; an integer or float
        literal for ``double``; ``True`` or ``False`` for ``bint``; a
        string literal or ``None`` for ``str``.  Any default fits an
        ``object`` parameter.
        """
        for arg, default in self.param_defaults.items():
            if arg not in self.params:
                continue
            ctype = get_type_expr(self._given_arg_type(arg), c_style=True)
            if not default_fits(default, ctype):
                code = cst.Module([]).code_for_node(default).strip()
                self._object_param(
                    arg, f"its default value {code} is not a {ctype} literal")

    def demote_rebound_param(self, arg: str, reason: str) -> None:
        """Type ``object`` a parameter that the formula binds to a value
        not provably of its C type.

        For a cached cells only the ``_f_`` formula, where the binding
        is, takes the parameter as a Python object: the public method,
        which looks up the cache, keeps the C type, and so does the
        cache.  For an uncached cells the public method is the formula,
        so the parameter is ``object`` there.  Called by
        :func:`modelx_cython.powers.demote_rebound_params`.
        """
        self._object_param(arg, reason, formula_only=self.has_formula_def)

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
    def is_locked(self):
        """Whether the cells runs its formula under the model lock.

        True for a cached cells of a space that modelx exported with
        ``locked_spaces`` (see :attr:`ClassInfo.is_locked`).  Uncached
        cells have nothing to protect, and special methods such as
        ``__call__`` keep the exported body, so both are False.
        """
        return (self.parent is not None and self.parent.is_locked
                and self.has_formula_def and not self.is_special())

    @property
    def uses_dict_cache(self):
        """Whether the cells' values are cached in a Python dict.

        True for a cached cells with parameters that is not arrayable,
        which is every cells with parameters that has no type
        information.  Its ``_v_`` attribute is a ``cdef dict``.
        """
        return (self.has_formula_def and self.has_args()
                and not (self.has_typeinfo() and self.is_arrayable()))

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
        * when no call site exists inside the model,
          :data:`MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS` for a cached
          cells, and ``False`` for an uncached cells, which returned
          the array itself before uncached cells were typed, and whose
          memoryview would serve no compiled caller;
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
            return self.has_formula_def and MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS
        return True

    def get_argtype_expr(self, arg: str, c_style=False, formula=False) -> str:
        """Return the Cython type expression for a parameter.

        The type is the one the spec's ``"param_type"`` gives, if it
        names the parameter, or else the traced one, unless it falls
        back to ``object`` (see :meth:`arg_type`).

        Parameters
        ----------
        arg : str
            Parameter name; must have been sampled by the tracer when
            type information exists.
        c_style : bool, default False
            If True, emit a C-style type (e.g. ``long long``) for .pxd
            files; otherwise a pure-Python-mode expression
            (e.g. ``_mx_cy.longlong``).
        formula : bool, default False
            True for the type in the method that holds the formula
            (see :meth:`arg_type`).

        Returns
        -------
        str
            The type expression, or ``"object"`` when no type
            information was sampled.
        """
        if self.has_typeinfo():
            return get_type_expr(self.arg_type(arg, formula=formula),
                                 c_style=c_style)
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
        Every other array return is ``"object"``: an array of strings,
        booleans or Python objects is neither its element type nor a
        numeric memoryview.  Non-array returns use the expression for
        :attr:`norm_type`, and cells without type information yield
        ``"object"``.

        Parameters
        ----------
        c_style : bool, default False
            If True, emit C-style types for .pxd files.
        """

        if self.has_typeinfo():
            typ = get_type_expr(self.norm_type, c_style=c_style)
            if self.is_array_returned:
                # A str array is not a str, a bool array does not coerce
                # to a bint memoryview (its buffer format is '?'), and a
                # 0-d array has no memoryview type
                if (not self.is_real_value or self.norm_type is bool
                        or self._rt.ret_type.ndim < 1
                        or not self.use_memoryview):
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
        """Return True if the type of parameter ``arg`` (see
        :meth:`arg_type`) is integral.
        """
        assert self.has_args() and self.has_typeinfo()
        return issubclass(self.arg_type(arg), numbers.Integral)

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
        it returns a real-valued scalar (not an array).  A parameter
        that the spec's ``"param_type"`` makes integral but that was
        not traced as one has no observed maximum to size the array
        with, so its cells is not arrayable.  Requires parameters and
        type information.
        """
        assert self.has_args() and self.has_typeinfo()
        if (self.is_int_args and self.is_real_value
                and not self.is_array_returned
                and all(p in self._rt.max_args for p in self.params)):
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
        when the class is defined in ``module`` itself (its
        fully-qualified name starts with ``module`` on a dotted
        boundary), fully qualified otherwise; ``''`` for non-space
        refs.
    is_relative : bool
        True when ``decl_type_expr`` is relative to ``module``.
    module_name : str
        The ``__name__`` of the sampled value when it is a module
        (``"math"`` for a reference to the math module), else ``''``.
    """

    module: str
    cls: str
    name: str
    type_: type = None
    mx_class: str = ''
    decl_type_expr: str = ''
    is_relative: bool = False
    module_name: str = ''

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
                if rt_info.mx_class.startswith(module + "."):
                    # Defined in the module itself; a bare prefix match
                    # would also catch a module whose name merely
                    # extends this one, so test on the dot boundary.
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
            self.module_name = getattr(rt_info, "module_name", "")
        else:
            self.module = module
            self.cls = cls
            self.name = name

    def get_type_expr(self, c_style=False):
        """Return the type expression used to declare this ref.

        Returns ``decl_type_expr`` when the ref holds a model space;
        ``"object"`` when the ref was never sampled at runtime;
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
        Maps a tuple of parameter names to the maximum argument
        values observed across all traced cells sharing that
        parameter tuple.
    _max_arg_cells : dict
        Maps a parameter tuple to ``{param: cells fqname}`` recording
        which cells produced each maximum, for logging.
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
        """Create a CombinedCellsInfo per lexical cells and track the
        maximum argument values observed per parameter tuple.
        """
        # .get: model classes and container-only spaces have no cells
        formula_defs = self.visitor.formula_defs.get(self.name, ())
        for name, lx_info in self.visitor.cells_info.get(self.name, {}).items():
            # An uncached cells has no _f_ method: its public method is
            # the formula and is what the tracer recorded.  It has no
            # cache either, so its arguments do not size any cache array.
            uncached = not lx_info.is_special() and name not in formula_defs
            trace_name = lx_info.method_fqname if uncached else lx_info.fqname
            rt_info = self.logger.cells_info.get(trace_name, None)
            self.cells[name] = CombinedCellsInfo(
                self,
                lx_info, rt_info,
                self.module.spec.get_spec(self.fqname).get(TransSpec.CELLS, {}).get(name, {})
            )
            if rt_info and not uncached:
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
    def is_locked(self):
        """Whether the space was exported with ``locked_spaces``.

        Read from the source: a locked space class assigns
        ``self._mx_lock`` in its ``__init__``
        (:attr:`~modelx_cython.parser.ModuleVisitor.locked_classes`).
        """
        return self.name in self.visitor.locked_classes

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
        plus one.
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


