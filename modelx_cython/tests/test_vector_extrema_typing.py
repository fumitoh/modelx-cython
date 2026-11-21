"""diag-vec: the element typer of vectors.py on min()/max() of typed scalars.

All integer -> integer, all float -> float, mixed -> NUM (Python returns one argument unchanged, so
the value may be an int); NUM is a float only through arithmetic with a float. A reduction over
vectors is lowered only when its element is a float, so value and type equal the Python sum.
Ties are not a typing question (Python returns the first of equal arguments; Cython's C min/max
keeps the first on equality too): integration_tests/test_vector_extrema_and_cache.py checks them
bit by bit.
"""
import libcst as cst
import pytest

from modelx_cython import vectors
from modelx_cython.vectors import FLOAT, INT, NUM, VectorLowering


class _Cls:
    cells = {}


def _typer(builtins=('sum', 'len', 'zip', 'range', 'min', 'max')):
    lowering = VectorLowering(_Cls(), None, set(builtins), [0])
    lowering.scalars = {'i': INT, 'j': INT, 'f': FLOAT, 'g': FLOAT}
    return lowering


@pytest.mark.parametrize('expr, kind', [
    ('min(i, j)', INT), ('max(i, j, 3)', INT),                 # int / int
    ('min(f, g)', FLOAT), ('max(f, 0.5)', FLOAT),              # float / float
    ('max(f, i)', NUM), ('min(i, f)', NUM), ('max(5.0, i)', NUM),   # int / float
    ('f * max(f, i)', FLOAT), ('max(f, i) * 2.0', FLOAT), ('max(f, i) / 2', FLOAT),
    ('max(f, min(i, j)) * f', FLOAT), ('i * max(f, i)', NUM), ('max(f, i) + 1', NUM),
    ('-max(f, i)', NUM), ('max(f, i) ** 2.0', None), ('max(f, i) < 1.0', None),
    ('max(f, x)', None), ('max(i)', None), ('max(*v)', None), ('max(f, i, key=k)', None),
])
def test_min_max_kinds(expr, kind):
    assert _typer().scalar(cst.parse_expression(expr)) == kind


def test_shadowed_min_max_are_not_typed():
    assert _typer(('sum', 'zip')).scalar(cst.parse_expression('min(i, j)')) is None


def test_num_never_types_a_local_or_vector():
    lowering = _typer()
    assert lowering._aug_kind((None, cst.AddAssign(), cst.parse_expression('max(f, i)'))) is None
    assert {FLOAT: 'AV', INT: 'AI'}.get(NUM, 'A?') == 'A?'


def test_multi_parameter_cache_key_is_row_major():
    class Axis:
        def __init__(self, key, lower, stop):
            self.key, self.lower, self.stop = key, lower, stop

    class Cell:
        params = ['t', 'g', 'z']
        associative_logical_axes = (Axis('t', 0, 10), Axis('g', 1, 6), Axis('z', -1, 2))

    source = vectors._cache_key(Cell(), 'k')
    body = '\n'.join(source.splitlines()[1:]) + '\n'      # the line after the declaration
    seen = set()
    for t in range(10):
        for g in range(1, 6):
            for z in range(-1, 2):
                scope = {'t': t, 'g': g, 'z': z, 'k': -1}
                exec(body, {}, scope)
                assert scope['k'] == ((t - 0) * 5 + (g - 1)) * 3 + (z + 1)
                seen.add(scope['k'])
    assert seen == set(range(10 * 5 * 3))
    for t, g, z in ((10, 1, 0), (0, 0, 0), (0, 1, 2), (-1, 1, 0)):
        scope = {'t': t, 'g': g, 'z': z, 'k': -1}
        exec(body, {}, scope)
        assert scope['k'] == -1


class _Axis:
    def __init__(self, key, lower, stop):
        self.key, self.lower, self.stop = key, lower, stop


class _VecCell:
    vector_shape = 'V'
    has_formula_def = True
    is_locked = False

    def __init__(self, name, sizes, spec=None):
        self.name = name
        self.params = [f'p{k}' for k in range(len(sizes))]
        self.associative_logical_axes = tuple(_Axis(p, 0, n) for p, n in zip(self.params, sizes))
        self._spec = spec or {}

    def get_argtype_expr(self, param, c_style=True):
        return 'long long'


class _Space:
    fqname = 'M.S'

    def __init__(self, *cells):
        self.cells = {c.name: c for c in cells}
        for c in cells:
            c.parent = self


def test_vector_cache_budget_d74():
    from modelx_cython.vectors import VECTOR_CACHE_SPACE_BYTES as REUSE_DENSE_SPACE_BYTES
    small = _VecCell('small', (1104, 6))
    wide = _VecCell('wide', (300, 300))                 # 90,000 > max_dense_entries 65,536
    allowed = _VecCell('allowed', (300, 300), spec={'max_dense_entries': 100_000})
    space = _Space(small, wide, allowed)
    chosen, refused = vectors.vector_cache_plan(space)
    assert chosen['small'][1:] == (6624, 6624 * vectors.CACHE_SLOT_BYTES)
    assert 'max_dense_entries 65,536' in refused['wide']
    assert 'allowed' in chosen and vectors.uses_vector_cache(small) and not vectors.uses_vector_cache(wide)
    # the Space budget: largest first, the rest keep the dictionary cache with the reason
    per_cell = (1 << 20)                                # MAX_CACHE_SLOTS slots = 8 MiB each
    many = [_VecCell(f'c{k}', (1024, 1024), spec={'max_dense_entries': per_cell}) for k in range(9)]
    space = _Space(*many)
    chosen, refused = vectors.vector_cache_plan(space)
    assert len(chosen) == REUSE_DENSE_SPACE_BYTES // (per_cell * vectors.CACHE_SLOT_BYTES) == 8
    assert list(refused) == ['c8'] and 'Space budget' in refused['c8']
