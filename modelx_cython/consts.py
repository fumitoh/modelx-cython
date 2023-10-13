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
CY_BOOL_T = CY_MOD + ".bint"
CY_INT_T = CY_MOD + ".int"
CY_FLOAT_T = CY_MOD + ".double"


def is_user_defined(name: str):
    if name[:1] == "_":
        return False
    else:
        return True


