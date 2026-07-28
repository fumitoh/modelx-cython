"""Type normalization and Cython type-expression helpers.

This module maps the concrete Python types observed by the tracer to
a small set of normalized types, and renders those normalized types
as Cython type expressions in either C style (for ``cdef``
declarations in generated .pxd files) or pure-Python style (for
annotations qualified with the ``_mx_cy`` cython alias).

Attributes
----------
CY_BOOL_T : str
    Cython boolean type name (``"bint"``).
CY_INT_T : str
    C-style integer type name (``"long long"``).
CY_INT_T_P : str
    Pure-Python-mode spelling of the integer type (``"longlong"``,
    rendered as ``_mx_cy.longlong``).
CY_INT_C_TYPE : type
    ctypes counterpart of :data:`CY_INT_T`
    (:class:`ctypes.c_longlong`).
CY_FLOAT_T : str
    Floating-point type name (``"double"``), identical in both
    styles.
str_to_type : dict
    Maps the type-name strings accepted in spec files (``"bool"``,
    ``"int"``, ``"float"``, ``"str"``, ``"object"``) to the
    normalized Python types used internally (``bool``,
    ``numbers.Integral``, ``numbers.Real``, ``str``, ``object``).
"""

import enum
import numbers
import ctypes

from modelx_cython.consts import CY_MOD

CY_BOOL_T = "bint"
CY_INT_T = "long long"
CY_INT_T_P = "longlong"     # For Pure Python syntax
CY_INT_C_TYPE = ctypes.c_longlong
CY_FLOAT_T = "double"


str_to_type = {
    "bool": bool,
    "int": numbers.Integral,
    "float": numbers.Real,
    "str": str,
    "object": object
}


def normalize_type(typ: type) -> type:
    """Collapse a concrete runtime type to a normalized type.

    Parameters
    ----------
    typ : type
        A concrete type observed at runtime.

    Returns
    -------
    type
        One of ``bool``, ``numbers.Integral``, ``numbers.Real``,
        ``str``, or ``object``.

    Notes
    -----
    ``bool`` is tested before ``numbers.Integral`` deliberately:
    ``bool`` is a subclass of ``int``, so with the opposite order
    booleans would normalize to integers.
    """
    if issubclass(typ, bool):
        return bool

    if issubclass(typ, numbers.Integral):
        return numbers.Integral

    elif issubclass(typ, numbers.Real):
        return numbers.Real

    elif issubclass(typ, str):
        return str

    else:
        return object


def get_type_expr(typ, c_style=False):
    """Return the Cython type expression for a type.

    Parameters
    ----------
    typ : type
        A type, classified with ``issubclass`` in the same order as
        :func:`normalize_type` (bool before Integral before Real).
    c_style : bool, optional
        If True, return C-syntax names (``"bint"``, ``"long long"``,
        ``"double"``) for use in ``cdef`` declarations and .pxd
        files.  If False (the default), return pure-Python-mode
        expressions qualified with the cython alias
        (``"_mx_cy.bint"``, ``"_mx_cy.longlong"``,
        ``"_mx_cy.double"``).

    Returns
    -------
    str
        The type expression.  String types yield ``"str"`` and all
        other non-numeric types yield ``"object"``, regardless of
        *c_style*.
    """
    if issubclass(typ, bool):
        if c_style:
            return CY_BOOL_T
        else:
            return f"{CY_MOD}.{CY_BOOL_T}"
    elif issubclass(typ, numbers.Integral):
        if c_style:
            return CY_INT_T
        else:
            return f"{CY_MOD}.{CY_INT_T_P}"
    elif issubclass(typ, numbers.Real):
        if c_style:
            return CY_FLOAT_T
        else:
            return f"{CY_MOD}.{CY_FLOAT_T}"

    elif issubclass(typ, str):
        return "str"
    else:
        return "object"