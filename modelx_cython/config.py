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

"""Translation-spec handling for mx2cy.

This module defines :class:`TransSpec`, a thin wrapper around the
user-supplied spec dictionary read from the spec file passed to the
``mx2cy`` command (``--spec``).  The spec lets the user customize how
spaces and cells are translated to Cython, for example by fixing a
cells return type or declaring maximum cells parameter sizes.
"""

from typing import Union, Sequence, Mapping
from modelx_cython.consts import (
    FILE_PREF,
    GLOBAL_PREF,
    MODULE_PREF,
    SPACE_PREF,
)


class TransSpec:
    """Wrapper around the user-supplied translation-spec dictionary.

    The spec is a nested dict (read from the spec file with
    ``ast.literal_eval``) in which each nesting level describes one
    space.  A space-level dict may have these keys:

    * ``"spaces"``: dict mapping child space names (without prefixes)
      to their own space-level dicts.
    * ``"cells_param_size"``: dict mapping a cells parameter name (or
      a tuple of parameter names) to its maximum size (or a tuple of
      sizes).  Used to size the fixed-length caches generated for
      cells with parameters; sampled maximum arguments can enlarge
      but not shrink these sizes.
    * ``"cells"``: dict mapping cells names to per-cells dicts.
    * ``"cells_params"`` (deprecated): older form of
      ``"cells_param_size"``; maps a parameter name (or tuple) to a
      dict with a ``"size"`` key.  Consulted only when
      ``"cells_param_size"`` is absent.

    A per-cells dict may have:

    * ``"return_type"``: either a type name accepted by
      :data:`modelx_cython.typedefs.str_to_type` (``"bool"``,
      ``"int"``, ``"float"``, ``"str"``, ``"object"``) that overrides
      the traced return type, or ``"memoryview"``, which forces a
      typed-memoryview return type for a real-valued numpy array
      return regardless of the usage analysis.  A spec return type
      takes effect only when the sample run produced type
      information for the cells; otherwise it is ignored (with a
      warning for ``"memoryview"``) and the return type falls back
      to ``object``.
    * ``"param_type"``: dict mapping parameter names of the cells to
      type names accepted by :data:`modelx_cython.typedefs.str_to_type`
      that override the traced parameter types, e.g.
      ``{"t": "int", "rate": "float"}``.  Parameters it does not name
      keep their traced types.  Like ``"return_type"``, it takes
      effect only when the sample run produced type information for
      the cells (otherwise it is ignored with a warning).  A
      parameter name the cells does not have, or an unknown type
      name, is an error.

    The top-level dict, which describes the model, may also have:

    * ``"compiler_directives"``: dict of Cython compiler directives,
      e.g. ``{"infer_types": True}``, passed to ``cythonize`` in the
      generated ``setup.py`` next to ``freethreading_compatible``
      (see :meth:`get_compiler_directives`).
    * ``"use_libm"``: ``True`` to compile ``math.exp(x)``,
      ``math.log(x)`` and ``math.pow(x, y)`` calls, made through a
      reference to the ``math`` module on arguments that are provably
      C numbers, to calls of the C library functions (see
      :meth:`get_use_libm`).  Off by default.

    The ``"space_params"`` key is defined below but is not currently
    consumed by the pipeline.

    Attributes
    ----------
    SPACES : str
        Key ``"spaces"`` holding child space specs.
    SPACE_PARAMS : str
        Key ``"space_params"`` (currently unused).
    CELLS : str
        Key ``"cells"`` holding per-cells specs.
    CELLS_PARAM_SIZE : str
        Key ``"cells_param_size"`` holding max parameter sizes.
    CELLS_PARAMS : str
        Deprecated key ``"cells_params"``.
    SIZE : str
        Deprecated key ``"size"`` used inside ``"cells_params"``.
    RET_T : str
        Key ``"return_type"`` in a per-cells spec.
    RET_MEMORYVIEW : str
        Value ``"memoryview"`` for ``"return_type"`` forcing
        memoryview emission.
    PARAM_T : str
        Key ``"param_type"`` in a per-cells spec.
    COMPILER_DIRECTIVES : str
        Top-level key ``"compiler_directives"``.
    USE_LIBM : str
        Top-level key ``"use_libm"``.
    """

    SPACES = "spaces"
    SPACE_PARAMS = "space_params"
    CELLS = "cells"
    CELLS_PARAM_SIZE = "cells_param_size"
    CELLS_PARAMS = "cells_params"   # deprecated
    SIZE = "size"   # deprecated
    RET_T = "return_type"
    RET_MEMORYVIEW = "memoryview"   # RET_T value forcing memoryview emission
    PARAM_T = "param_type"
    COMPILER_DIRECTIVES = "compiler_directives"
    USE_LIBM = "use_libm"

    def __init__(self, data: dict) -> None:
        """Store the spec dictionary.

        Parameters
        ----------
        data : dict
            The nested spec dictionary as described in the class
            docstring.
        """
        
        self._data = data

    def get_spec(self, object_path: str):
        """Return the space-level spec dict for a dotted object path.

        The first component of *object_path* (the model package name)
        is discarded.  Each remaining component is interpreted by its
        prefix: components starting with ``_mx_`` (module file names
        such as ``_mx_classes``) are skipped; components starting
        with ``_m_`` (subpackage) or ``_c_`` (space class) have their
        prefix stripped and are looked up under the ``"spaces"`` key
        of the current level.  Components with no recognized prefix
        are ignored.

        Parameters
        ----------
        object_path : str
            Dotted path of a module, class, or space in the exported
            model package.

        Returns
        -------
        dict
            The spec dict for the addressed space, or an empty dict
            if any looked-up name is missing from the spec.

        Examples
        --------
        Paths such as the following address the ``Projection`` space
        and the ``Parent.Child`` space respectively::

            PkgName._mx_classes._c_Projection
            PkgName._m_Parent._m_Child
        """
        def get_subspace(data, name):
            """Return the child-space spec *name* of *data*, or {}."""
            spaces = data.get(self.SPACES)
            if spaces and spaces.get(name):
                return spaces.get(name)
            else:
                return {}
            
        names = object_path.split(".")[1:]  # Remove first
        data = self._data
        while names:
            name = names.pop(0)
            if name[:len(FILE_PREF)] == FILE_PREF:
                continue
            elif name[:len(MODULE_PREF)] == MODULE_PREF:
                name = name[len(MODULE_PREF):]
                data_ = get_subspace(data, name)
                if data_:
                    data = data_
                else:
                    return {}
            elif name[:len(SPACE_PREF)] == SPACE_PREF:
                name = name[len(SPACE_PREF):]
                data_ = get_subspace(data, name)
                if data_:
                    data = data_
                else:
                    return {}

        return data

    def get_compiler_directives(self) -> dict:
        """Return the Cython compiler directives given in the spec.

        Reads the top-level ``"compiler_directives"`` dict and checks
        it minimally against the installed Cython with
        :func:`check_compiler_directive`.  The values are returned as
        given, in the given order.

        The directives that would make the translated code compute
        something else than mx2cy translates it for -- ``cpow=True``,
        ``language_level=2``, ``annotation_typing=False`` and
        ``legacy_implicit_noexcept=True`` -- are rejected (see
        :data:`UNSUPPORTED_DIRECTIVES`).  Others change what the
        compiled model computes by design, and mx2cy does not guard
        against that.  ``infer_types=True`` in
        particular lets Cython type an untyped local variable as a C
        integer when every assignment to it is a C integer expression
        -- a local initialized from an integer literal becomes a C
        ``long``, only 32 bits wide on Windows -- and the C integer
        silently wraps around on overflow where the Python integer
        would have grown.  ``boundscheck=False``, ``wraparound=False``
        and ``cdivision=True`` likewise trade Python's semantics for
        speed.

        Returns
        -------
        dict
            Directive names mapped to values; empty if the key is
            absent.

        Raises
        ------
        ValueError
            If the value is not a dict, or a key or value is invalid.
        """
        directives = self._data.get(self.COMPILER_DIRECTIVES, {})
        if directives is None:
            return {}
        if not isinstance(directives, dict):
            raise ValueError(
                f"invalid value for spec '{self.COMPILER_DIRECTIVES}': "
                f"expected a dict of directive names to values, got "
                f"{directives!r}")
        for name, value in directives.items():
            check_compiler_directive(name, value)
        return dict(directives)

    def get_use_libm(self) -> bool:
        """Whether the spec turns on the C library math functions.

        With the top-level ``"use_libm": True``, a call
        ``self.<ref>.exp(x)``, ``self.<ref>.log(x)`` or
        ``self.<ref>.pow(x, y)`` in a formula, where ``<ref>`` holds
        the ``math`` module and every argument is provably a C number,
        is compiled to a C library call (see
        :func:`modelx_cython.powers.rewrite_math_calls`).  Where the
        arguments and the C result are finite, the C result is
        returned, which on the platform it was measured on (MSVC) is
        bit for bit what ``math`` returns; everything else -- infinite
        or nan arguments, overflow, ``log`` of zero or of a negative
        number, ``pow`` of zero to a negative power or of a negative
        number to a fractional power -- is computed by the ``math``
        module itself, so it returns or raises exactly what Python
        does.

        Raises
        ------
        ValueError
            If the value is not ``True`` or ``False``.
        """
        value = self._data.get(self.USE_LIBM, False)
        if not isinstance(value, bool):
            raise ValueError(
                f"invalid value for spec '{self.USE_LIBM}': expected True "
                f"or False, got {value!r}")
        return value


UNSUPPORTED_DIRECTIVES = {
    "cpow": (
        (None, False),
        "mx2cy translates '**' for cpow=False; with cpow=True, a power "
        "of two C integers stays an integer, so 2 ** -3 is 0"),
    "language_level": (
        (3, "3", "3str"),
        "the translated model is Python 3 code; under language_level=2, "
        "'/' between C integers is floor division, so 3 / 12 is 0"),
    "annotation_typing": (
        (True,),
        "mx2cy declares the C types of the translated code with "
        "annotations, which annotation_typing=False ignores"),
    "legacy_implicit_noexcept": (
        (False,),
        "with legacy_implicit_noexcept=True, an exception raised in a "
        "compiled formula is printed and swallowed, and the formula "
        "returns a meaningless value, instead of raising"),
}
"""dict: Cython directives of which mx2cy supports only some values,
mapped to those values -- the ones it translates for -- and the reason.
Any other value would silently change what the compiled model
computes."""


def check_compiler_directive(name, value) -> None:
    """Check one Cython compiler directive given in the spec.

    The directive must be one the installed Cython knows and that
    applies to a whole module, as ``cythonize`` applies it.  The value
    must have the directive's type: ``True`` or ``False`` for a
    boolean directive (``None`` too where that is the directive's
    default, as for ``infer_types``), an ``int`` for an integer one, a
    string for a string one, and one of the accepted strings for an
    enumerated one.  Directives taking other values, such as
    ``locals``, are rejected.  So are the values of the directives in
    :data:`UNSUPPORTED_DIRECTIVES` other than those mx2cy translates
    for: ``cpow=True``, ``language_level=2``,
    ``annotation_typing=False`` and ``legacy_implicit_noexcept=True``.

    Raises
    ------
    ValueError
        If the directive is unknown to the installed Cython, cannot be
        set for a whole module, ``value`` is not of its type, or mx2cy
        does not support the value.
    """
    from Cython.Compiler import Options

    key = TransSpec.COMPILER_DIRECTIVES
    if not isinstance(name, str) or name not in Options.directive_types:
        raise ValueError(
            f"invalid spec '{key}': {name!r} is not a Cython compiler "
            "directive")
    if name in UNSUPPORTED_DIRECTIVES:
        allowed, reason = UNSUPPORTED_DIRECTIVES[name]
        # by type as well, so that 1 is not taken for True
        if not any(type(value) is type(a) and value == a for a in allowed):
            raise ValueError(
                f"invalid spec '{key}': mx2cy does not support the Cython "
                f"directive {name}={value!r}: {reason}")
    scopes = Options.directive_scopes.get(name, ("module",))
    if isinstance(scopes, str):     # a 1-tuple written without its comma
        scopes = (scopes,)
    if "module" not in scopes:
        raise ValueError(
            f"invalid spec '{key}': the Cython directive {name!r} cannot "
            "be set for a whole module")

    typ = Options.directive_types[name]
    default = Options.get_directive_defaults().get(name)
    if typ is bool:
        ok = isinstance(value, bool) or (value is None and default is None)
        expected = "True or False" + (" or None" if default is None else "")
    elif typ is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
        expected = "an integer"
    elif typ is str:
        if name == "language_level":
            ok = value in (2, 3, "2", "3", "3str")
            expected = "2, 3 or '3str'"
        else:
            ok = isinstance(value, str)
            expected = "a string"
    elif callable(typ) and typ not in (dict, type):
        # an enumeration or an encoding name, which Cython parses itself
        try:
            ok = isinstance(value, str) and Options.parse_directive_value(
                name, value) is not None
        except ValueError:
            ok = False
        expected = "a value Cython accepts"
    else:
        ok = False
        expected = "a value mx2cy can pass to cythonize"
    if not ok:
        raise ValueError(
            f"invalid spec '{key}': the Cython directive {name!r} must be "
            f"{expected}, got {value!r}")
