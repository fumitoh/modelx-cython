import enum
import numbers
import ctypes

from modelx_cython.consts import CY_MOD

ANY_T = "object"
FLOAT_T = "float"
STR_T = "str"
CY_BOOL_T = "bint"
CY_INT_T = "long long"
CY_INT_T_P = "longlong"     # For Pure Python syntax
CY_INT_C_TYPE = ctypes.c_longlong
CY_FLOAT_T = "double"


def get_type_expr(typ, c_style=False):

    if issubclass(typ, numbers.Integral):
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
        return STR_T
    else:
        return ANY_T


class TypeGroup(enum.IntEnum):

    C_TYPE = 1
    MEM_VIEW = 2
    OBJECT = 3



