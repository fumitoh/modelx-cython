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