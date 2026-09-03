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

"""Naming conventions shared across the mx2cy translation pipeline.

A modelx model exported with ``Model.export()`` is a pure-Python
package whose generated names follow fixed prefixes.  This module
centralizes those prefixes and the well-known names derived from
them, so that the tracer, parser, builder, and transformer all agree
on how to recognize and generate names.

Attributes
----------
FILE_PREF : str
    ``"_mx_"``.  Prefix of generated module file names
    (``_mx_model``, ``_mx_classes``, ``_mx_sys``).
GLOBAL_PREF : str
    ``"_mx_"``.  Prefix of generated system-level method, global,
    and alias names (``_mx_assign_refs``, ``_mx_cy``, ...).
MODULE_PREF : str
    ``"_m_"``.  Prefix of subpackage names holding child spaces; a
    child space ``Foo`` lives in subpackage ``_m_Foo``.
SPACE_PREF : str
    ``"_c_"``.  Prefix of the class name generated for a space;
    space ``Foo`` becomes class ``_c_Foo``.
FORMULA_PREF : str
    ``"_f_"``.  Prefix of the methods holding cells formulas;
    cells ``bar`` is implemented by method ``_f_bar``.
VAR_PREF : str
    ``"_v_"``.  Prefix of instance attributes caching cells values.
HAS_PREF : str
    ``"_has_"``.  Prefix of boolean instance attributes recording
    whether the corresponding ``_v_`` cache entry holds a value.
MX_SELF : str
    ``"self"``.  Name of the instance parameter in generated
    methods.
MX_MODEL_MOD : str
    ``"_mx_model"``.  Module defining the model class.
MX_SPACE_MOD : str
    ``"_mx_classes"``.  Module defining the space classes of a
    (sub)package.
MX_SYS_MOD : str
    ``"_mx_sys"``.  System module providing base classes.
MX_ASSIGN_REFS : str
    ``"_mx_assign_refs"``.  Name of the generated method assigning
    refs to a space.
MX_COPY_REFS : str
    ``"_mx_copy_refs"``.  Name of the generated method copying refs
    from a base space.
MX_LOCK : str
    ``"_mx_lock"``.  Attribute holding the ``threading.RLock`` shared
    by the locked spaces of a model.  Its assignment in a space's
    ``__init__`` marks the space class as locked.
BASE_MODEL : str
    ``"BaseModel"``.  Name of the model base class defined in
    ``_mx_sys``.
SPACE_PARAMS : str
    ``"_v_space_params"``.  Prefix of module-level globals listing
    the parameter names of a parametrized space
    (``_v_space_params_<SpaceName>``).
CY_MOD : str
    ``"_mx_cy"``.  Alias under which the ``cython`` module is
    imported in translated modules.
NP_MOD : str
    ``"_mx_np"``.  Reserved alias for ``numpy`` (currently unused).
PD_MOD : str
    ``"_mx_pd"``.  Reserved alias for ``pandas`` (currently unused).
"""

import ctypes

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
MX_LOCK = GLOBAL_PREF + "lock"
BASE_MODEL = "BaseModel"
SPACE_PARAMS = VAR_PREF + "space_params"
CY_MOD = GLOBAL_PREF + "cy"
NP_MOD = GLOBAL_PREF + "np"
PD_MOD = GLOBAL_PREF + "pd"



def is_user_defined(name: str):
    """Return True if *name* is a user-defined modelx name.

    Any name starting with an underscore is treated as
    system-generated (this covers all the ``_mx_``/``_m_``/``_c_``/
    ``_f_``/``_v_``/``_has_`` prefixes above) and yields False;
    every other name yields True.
    """
    if name[:1] == "_":
        return False
    else:
        return True


