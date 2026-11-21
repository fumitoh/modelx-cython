"""Model-independent cache policy; observations never narrow logical domains."""

from .cell_entries import DenseAxis
import ctypes
import sys

# Shared resource policies, independent of model/Cell names. At double plus
# bint-readiness storage the dense ceiling is 768 KiB per Cell on Windows. Associative
# storage retains the existing growing overflow table beyond its primary cap.
MAX_DENSE_ENTRIES = 65536
MAX_PRIMARY_ENTRIES = 4096


def enabled(cell):
    module = getattr(getattr(cell, 'parent', None), 'module', None)
    return getattr(getattr(module, 'spec', None), 'automatic_optimizations', False)


def observed_axes(cell, logical):
    runtime = getattr(cell, '_rt', None)
    minimum = getattr(runtime, 'observed_min_args', {})
    maximum = getattr(runtime, 'observed_max_args', {})
    if not logical or any(axis.key not in minimum or axis.key not in maximum for axis in logical):
        return logical
    declared = declared_axes(cell)
    # A declared size is allocated as declared (cells_param_size); only sampled axes shrink.
    axes = tuple(axis if axis.key in declared else
                 DenseAxis(axis.key, max(axis.lower, minimum[axis.key]),
                           min(axis.stop, maximum[axis.key] + 1), axis.c_type) for axis in logical)
    if not all(axis.lower < axis.stop for axis in axes):
        return logical
    full, compact = 1, 1
    for complete, observed in zip(logical, axes):
        full *= complete.stop - complete.lower
        compact *= observed.stop - observed.lower
    value_size = ctypes.sizeof(ctypes.c_int if cell.get_rettype_expr(c_style=True) == 'bint'
                               else ctypes.c_longlong)
    saved = (full - compact) * (value_size + ctypes.sizeof(ctypes.c_int))
    # A compact cache adds a pointer and a cold dictionary. Do not trade a
    # near-full native array (and its range proofs) for less than that storage.
    spill_cost = ctypes.sizeof(ctypes.c_void_p) + sys.getsizeof({})
    return axes if saved >= spill_cost else logical


def declared_axes(cell):
    """Parameters of a Cell whose size the spec declares (cells_param_size, any level)."""
    parent = getattr(cell, 'parent', None)
    if parent is None or not hasattr(parent, 'declared_axes'):
        return {}
    parent.cells_arg_sizes
    return parent.declared_axes.get(tuple(getattr(cell, 'params', ())), {})


def max_dense_entries_set(cell):
    """Whether the spec sets max_dense_entries for the Cell or its Space (it then applies whether or
    not automatic optimizations are on)."""
    if 'max_dense_entries' in getattr(cell, '_spec', {}):
        return True
    parent = getattr(cell, 'parent', None)
    spec = getattr(getattr(parent, 'module', None), 'spec', None)
    return spec is not None and hasattr(spec, 'get_spec') and 'max_dense_entries' in spec.get_spec(parent.fqname)


def max_dense_entries(cell):
    """D74: the inline dense ceiling of a Cell: spec key ``max_dense_entries`` of the Cell, else of its
    Space, else MAX_DENSE_ENTRIES. Inline dense storage is part of every instance on every route."""
    value = getattr(cell, '_spec', {}).get('max_dense_entries')
    if value is None:
        parent = getattr(cell, 'parent', None)
        module = getattr(parent, 'module', None)
        spec = getattr(module, 'spec', None)
        if spec is not None and hasattr(spec, 'get_spec'):
            value = spec.get_spec(parent.fqname).get('max_dense_entries')
    if value is None:
        return MAX_DENSE_ENTRIES
    if type(value) is not int or value <= 0:
        raise ValueError(f'{getattr(cell, "fqname", cell)}: max_dense_entries must be a positive integer')
    return value


# R-3b: above the default ceiling, storage is chosen by use. A rectangle stays dense while it is at most
# DENSE_USE_FACTOR times the entries one receiver (policy root) fills in the sample: the formula evaluations
# per receiver the tracer counts (max_evaluations_per_receiver). A hash entry costs about 550 instructions
# to fill and 119 per hit against a few for a dense slot, while a dense slot costs about 2 (collect reset)
# to 10 (loop zero-fill) instructions per policy whether used or not: dense is cheaper up to a ratio of
# about 80 on the loop route and 400 on collect.
DENSE_USE_FACTOR = 16


def filled_per_receiver(cell):
    """The most entries of the Cell one receiver filled in the sample (its formula evaluations)."""
    return getattr(getattr(cell, '_rt', None), 'max_evaluations_per_receiver', 0) or 0


def dense_by_use(cell, count):
    """R-3b: whether a rectangle of ``count`` entries above the ceiling stays dense by use. An explicit
    max_dense_entries (Cell or Space) is a hard limit and is not overridden."""
    if max_dense_entries_set(cell):
        return False
    filled = filled_per_receiver(cell)
    return filled > 0 and count <= DENSE_USE_FACTOR * filled


def dense_allowed(cell, count):
    """Dense when the rectangle is within the ceiling, or (R-3b) within DENSE_USE_FACTOR x its use."""
    return count > 0 and (count <= max_dense_entries(cell) or dense_by_use(cell, count))


def dense_count(cell):
    count = 1
    for axis in observed_axes(cell, cell.dense_logical_axes):
        count *= axis.stop - axis.lower
    return count


def dense_fits(cell):
    return dense_allowed(cell, dense_count(cell))


def primary_capacity(cell):
    # Formula evaluations are cache misses, counted per receiver rather than
    # summed across traced policies. Keep the table's existing <=50% load rule.
    evaluations = getattr(getattr(cell, '_rt', None), 'max_evaluations_per_receiver', 0)
    if evaluations <= 0:
        return 0
    return min(MAX_PRIMARY_ENTRIES, 1 << (2 * evaluations - 1).bit_length())
