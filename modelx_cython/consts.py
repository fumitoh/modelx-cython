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


FILE_PREF = "_mx_"
GLOBAL_PREF = "_mx_"
MODULE_PREF = "_m_"
SPACE_PREF = "_c_"
FORMULA_PREF = "_f_"
VAR_PREF = "_v_"
HAS_PREF = "_has_"
MX_SELF = "self"

MX_MODEL_MOD = FILE_PREF + "model"
MX_SPACE_MOD = FILE_PREF + "classes"
MX_SYS_MOD = FILE_PREF + "sys"

MX_ASSIGN_REFS = GLOBAL_PREF + "assign_refs"
MX_COPY_REFS = GLOBAL_PREF + "copy_refs"
CY_MOD = GLOBAL_PREF + "cy"
NP_MOD = GLOBAL_PREF + "np"
PD_MOD = GLOBAL_PREF + "pd"

ANY_T = "object"
FLOAT_T = "float"
CY_BOOL_T = "bint"
CY_INT_T = "long long"
CY_INT_T_P = "longlong"     # For Pure Python sysntax
CY_FLOAT_T = "double"


def is_user_defined(name: str):
    if name[:1] == "_":
        return False
    else:
        return True


