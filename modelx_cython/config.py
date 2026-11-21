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
import os
from dataclasses import dataclass
from modelx_cython.consts import (
    FILE_PREF,
    GLOBAL_PREF,
    MODULE_PREF,
    SPACE_PREF,
)


@dataclass(frozen=True)
class CompilerPolicy:
    """Resolve dependent diagnostic options once, before code generation."""
    automatic_optimizations: bool
    dense_bounds_proofs: bool
    dense_miss_binding: bool
    dense_one_axis: bool
    propagate_exact_receiver: bool
    native_cache_hits: str
    native_math: bool  # Guarded builtin rounding.
    # M1 (W1.1, default on since W1.5): '# cython: infer_types=True, overflowcheck=True'
    # on generated model modules.
    infer_types: bool = True
    # M1 (W1.3): lower pandas-spelled table reads to typed handle reads (table_reads.py).
    # On by default since the runtime handles (W1.2) landed (D45).
    typed_table_reads: bool = True
    # diag-4w (D71): the handle of a zero-argument table Cell is held per instance and read
    # borrowed (no Cell call, no reference count per read); false resolves it per call.
    held_table_handles: bool = True
    # auto-tables (D104): a plain pandas table that is not admitted, or pandas run on an admitted
    # table in a per-step cell (class C), stops the build instead of a warning.
    strict_tables: bool = False
    # integer guards (owner's rule): 'auto' (default) compiles without overflow checks when no
    # traced integer reaches UNCHECKED_LIMIT in magnitude, with them otherwise; 'checked' and
    # 'unchecked' force a mode. The build resolves 'auto' after the trace (resolve_integer_checks).
    integer_checks: str = 'auto'
    # T-3 (toolkit p2/tables/DESIGN.md): the sample runs on the user's frames (no table admission or
    # substitution); reads of a traced DataFrame or Series are lowered onto a PreparedTable bound to
    # the real frame (frame_reads.py). False restores the admission of the older table layer.
    prepared_tables: bool = True
    # T-4: opt-in check mode: each hold of a prepared table first checks that the frame still equals the copy
    # prepared when it was bound, and raises if it was edited in place without a clear
    verify_inputs: bool = False
    # T-5c: a formula whose table-read lowering raises an internal error is compiled as written (its reads on
    # pandas) and the build report says so; strict turns that internal error into a build failure (test suites)
    strict_lowering: bool = False

    @classmethod
    def resolve(cls, data):
        automatic = data.get('automatic_optimizations', True)
        miss = data.get('dense_miss_binding', automatic)
        values = dict(automatic_optimizations=automatic,
            dense_bounds_proofs=data.get('dense_bounds_proofs', automatic),
            dense_miss_binding=miss, dense_one_axis=data.get('dense_one_axis', automatic),
            propagate_exact_receiver=data.get('propagate_exact_receiver', automatic and miss),
            native_cache_hits=data.get('native_cache_hits', 'auto' if automatic and miss else 'none'),
            native_math=data.get('native_math', True),
            infer_types=data.get('infer_types', True),
            typed_table_reads=data.get('typed_table_reads', True),
            held_table_handles=data.get('held_table_handles', True),
            strict_tables=data.get('strict_tables', os.environ.get('MX_CY_STRICT_TABLES') == '1'),
            integer_checks=data.get('integer_checks', 'auto'),
            prepared_tables=data.get('prepared_tables', True),
            verify_inputs=data.get('verify_inputs', False),
            strict_lowering=data.get('strict_lowering', os.environ.get('MX_CY_STRICT_LOWERING') == '1'))
        for name, value in values.items():
            if name not in ('parallel_workers', 'native_cache_hits', 'integer_checks') and type(value) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if values['integer_checks'] not in ('auto', 'checked', 'unchecked'):
            raise ValueError('integer_checks must be auto, checked or unchecked')
        if type(values['native_cache_hits']) is not str or values['native_cache_hits'] not in (
                'none', 'scalar', 'dense', 'both', 'auto'):
            raise ValueError('native_cache_hits must be none, scalar, dense, both or auto')
        if not miss:
            if values['propagate_exact_receiver']:
                raise ValueError('propagate_exact_receiver requires dense_miss_binding')
            if values['native_cache_hits'] != 'none':
                raise ValueError('native_cache_hits requires dense_miss_binding')
        return cls(**values)


# integer guards: with every traced integer below 2**31 in magnitude, a product of two stays below
# 2**62, so the ordinary arithmetic of a formula keeps far from the 64-bit limit
UNCHECKED_LIMIT = 1 << 31
UNCHECKED_LIMIT_NOTE = (
    'Unchecked mode: intermediate values are not traced, so a product of several large terms or an '
    'integer power could exceed 64 bits and would wrap, as on upstream; set "integer_checks": "checked" '
    'in the spec to force the checked mode.')


def resolve_integer_checks(policy, largest_traced_integer):
    """The integer-check mode for this build: the spec's, or for 'auto' 'unchecked' when the
    largest traced integer is below UNCHECKED_LIMIT in magnitude and 'checked' otherwise."""
    if policy.integer_checks != 'auto':
        return policy.integer_checks
    return 'unchecked' if largest_traced_integer < UNCHECKED_LIMIT else 'checked'


class TransSpec:
    """Wrapper around the user-supplied translation-spec dictionary.

    The spec is a nested dict (read from the spec file with
    a safe literal decoder) in which each nesting level describes one
    space.  A space-level dict may have these keys:

    * ``"spaces"``: dict mapping child space names (without prefixes)
      to their own space-level dicts.
    * ``"cells_param_size"``: dict mapping a cells parameter name (or
      a tuple of parameter names) to its maximum size (or a tuple of
      sizes).  Used to size the fixed-length caches generated for
      cells with parameters; sampled maximum arguments can enlarge
      but not shrink these sizes. Also read at the model level (top of
      the spec) and in enclosing Spaces: the nearest declaration of a key
      wins. A tuple key applies to Cells with exactly that parameter
      tuple; a single-name key bounds that parameter's axis of every
      cached Cell in scope whose parameter tuple has no tuple entry.
    * ``"cells"``: dict mapping cells names to per-cells dicts.
    * ``"cells_params"`` (deprecated): older form of
      ``"cells_param_size"``; maps a parameter name (or tuple) to a
      dict with a ``"size"`` key.  Consulted only when
      ``"cells_param_size"`` is absent.

    A per-cells dict may have:

    * ``"native_numeric_cache"``: override automatic selection of native numeric
      key storage for an eligible cell. Other signatures use dictionaries.
      Both support compiled model pickling. Setting ``"numeric_cache_capacity"``
      also opts in unless ``"native_numeric_cache"`` is false.
    * ``"dense_window"``: a sequence of half-open ``(lower, stop)`` pairs
      in parameter order for a dense cache. The physical window must
      fit inside the existing logical bounds. A declared window overrides
      the automatic dense storage size ceiling for an eligible cached Cell.
      Exact spill keeps the original logical domain available.
    * ``"native_divisibility"``: enable the checked Cython zero-remainder
      comparison for proven signed-integer operands.
    * ``"memoize_power"``: enable a one-entry exact-operand memo for the
      existing typed power helper on a privately owned policy Space.
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

    The top-level ``"dense_one_axis"`` experiment separates checked and proven
    calls to an existing one-axis dense cache body.
    ``"propagate_exact_receiver"`` carries the receiver fact through those
    entries and private dense formula calls. ``"native_math"`` enables guarded
    builtin rounding by default.
    ``"infer_types"`` (default true): C type inference (``# cython: infer_types=True``) in
    generated model modules; false keeps Cython's default (no inference of C types for untyped
    locals). Overflow checks follow ``"integer_checks"`` either way.
    ``"integer_checks"`` (default ``"auto"``): ``"unchecked"`` compiles C integer arithmetic
    without overflow checks (as plain C), ``"checked"`` with them; ``"auto"`` chooses unchecked when
    no traced integer (Cell arguments and results, References, ItemSpace parameters, integer
    columns of input tables) reaches 2**31 in magnitude. ``_mx_integer_checks.json`` records the
    mode used and the largest traced integer. Limit of the unchecked mode: intermediate values are
    not traced, so a product of several large terms or an integer power could exceed 64 bits and
    would wrap, as on upstream; set ``"integer_checks": "checked"`` to force the checked mode.
    ``"typed_table_reads"`` (default true) allows inlining a row Cell into eligible field reads
    on the prepared pandas table path.
    ``"strict_tables"`` (default false, or true when the environment sets
    ``MX_CY_STRICT_TABLES=1``): a table read that remains on pandas in a cell with arguments
    stops a prepared-table build with an error instead of a warning.
    ``"held_table_handles"`` (default true): the handle of a zero-argument table Cell
    read by a lowered read is resolved once per instance into a field (reset with the
    per-policy caches) and used borrowed; false resolves it on every call.
    ``"native_cache_hits"`` defaults to ``"auto"`` when automatic optimization
    and dense miss binding are enabled. It selects scalar hits and already-proven
    dense hits; explicit alternatives are ``"none"``, ``"scalar"``, ``"dense"``
    and ``"both"``. Hits require ``"dense_miss_binding"``. Misses retain their
    original entry and Cython exception handling.
    ``"automatic_optimizations": False`` disables automatic selection and the
    default dense entry optimizations on every platform. Individual diagnostic
    options can override these defaults, subject to their dependencies.

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
    PARAM_BOUND_CHECKS = "param_bound_checks"
    SIZE = "size"   # deprecated
    RET_T = "return_type"
    DENSE_WINDOW = "dense_window"
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
        self.policy = CompilerPolicy.resolve(data)
        self.automatic_optimizations = self.policy.automatic_optimizations

    def get_declared_sizes(self, object_path: str):
        """Return ``(tuples, names)`` of the ``"cells_param_size"`` entries in
        force for a Space: ``{parameter tuple: (sizes, where)}`` and
        ``{parameter name: (size, where)}``. The model level comes first, then
        each enclosing Space down to the addressed one; a nearer declaration
        of the same key wins. A level without ``"cells_param_size"`` uses its
        deprecated ``"cells_params"`` sizes, as for one Space."""
        tuples, names = {}, {}

        def merge(data, where):
            declared = data.get(self.CELLS_PARAM_SIZE)
            if declared is None:
                declared = {keys: config[self.SIZE]
                            for keys, config in data.get(self.CELLS_PARAMS, {}).items()
                            if self.SIZE in config}
            for keys, stops in declared.items():
                if isinstance(keys, tuple):
                    tuples[keys] = tuple(_declared_stop(keys, k, v) for k, v in zip(keys, stops)), where
                else:
                    names[keys] = _declared_stop(keys, keys, stops), where

        merge(self._data, 'model')
        data, path = self._data, []
        for name in object_path.split(".")[1:]:
            for prefix in (MODULE_PREF, SPACE_PREF):
                if name[:len(prefix)] == prefix:
                    name = name[len(prefix):]
                    break
            else:
                continue
            data = (data.get(self.SPACES) or {}).get(name) or {}
            if not data:
                break
            path.append(name)
            merge(data, '.'.join(path))
        return tuples, names

    def get_declared_lowers(self, object_path: str):
        """Return ``(tuples, names)`` of the declared lower bounds in force for a Space, with the same
        precedence as :meth:`get_declared_sizes`: ``{parameter tuple: {parameter: lower}}`` and
        ``{parameter name: lower}``. A declaration is a size (lower bound: today's default) or a
        ``(lower, stop)`` pair; only pairs appear here (R-3-lower)."""
        tuples, names = {}, {}

        def merge(data):
            declared = data.get(self.CELLS_PARAM_SIZE)
            if declared is None:
                declared = {keys: config[self.SIZE]
                            for keys, config in data.get(self.CELLS_PARAMS, {}).items()
                            if self.SIZE in config}
            for keys, stops in declared.items():
                if isinstance(keys, tuple):
                    tuples[keys] = {k: _declared_lower(keys, k, v)
                                    for k, v in zip(keys, stops) if _declared_lower(keys, k, v) is not None}
                elif _declared_lower(keys, keys, stops) is not None:
                    names[keys] = _declared_lower(keys, keys, stops)
                else:
                    names.pop(keys, None)

        merge(self._data)
        data = self._data
        for name in object_path.split(".")[1:]:
            for prefix in (MODULE_PREF, SPACE_PREF):
                if name[:len(prefix)] == prefix:
                    name = name[len(prefix):]
                    break
            else:
                continue
            data = (data.get(self.SPACES) or {}).get(name) or {}
            if not data:
                break
            merge(data)
        return tuples, names

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


def _declared_pair(keys, key, value):
    """A declaration ``(lower, stop)`` of two ints with lower < stop, else None (a plain size)."""
    if isinstance(value, (tuple, list)):
        if (len(value) != 2 or not all(type(v) is int for v in value) or not value[0] < value[1]):
            raise ValueError(f'cells_param_size {keys!r}: {key} must be a size or a (lower, stop) pair of '
                             f'ints with lower < stop, got {value!r}')
        return tuple(value)
    return None


def _declared_stop(keys, key, value):
    pair = _declared_pair(keys, key, value)
    return value if pair is None else pair[1]


def _declared_lower(keys, key, value):
    pair = _declared_pair(keys, key, value)
    return None if pair is None else pair[0]
