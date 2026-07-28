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

    The ``"space_params"`` and ``"param_type"`` keys are defined
    below but are not currently consumed by the pipeline.

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
        Key ``"param_type"`` (currently unused).
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
