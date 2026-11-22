"""D86 (diag-vec): a dense product index over a multi-level key of small-range plain integers.

A HASHED multi-level table whose key parts are all plain ints in a box of at most GRID_MAX_SLOTS
positions finds rows by position arithmetic (row + 1 per position, 0 absent) instead of hashing.
Every lookup must give what the hash index gave: hit, miss inside the box, miss outside the box,
over-budget tables keep the hash index, duplicate keys find the same row, tables with float or
string levels are unchanged. Driven through the W1.2 handle driver with C-integer and object keys.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pdn
from table_matrix_cases_w1_2 import build


@pytest.fixture(scope='module')
def b(tmp_path_factory):
    return build(tmp_path_factory)


def _handle(b, frame):
    provider = importlib.import_module(b.provider)
    return provider, provider.TableHandle(pdn.table(frame))


def _frame(keys, seed=3):
    rng = np.random.default_rng(seed)
    keys = list(keys)
    rng.shuffle(keys)                       # any row order
    index = pd.MultiIndex.from_tuples(keys, names=['sex', 'age'])
    return pd.DataFrame({'q': np.arange(len(keys), dtype=float) * 0.5 + 1.0}, index=index)


def _lookups(b, h, frame, keys):
    d = b.driver
    for cint in (True, False):
        for key in keys:
            if key in frame.index:
                assert d.loc(h, key, 'q', cint) == frame.loc[key, 'q'], (key, cint)
                assert d.contains(h, key, cint)
            else:
                with pytest.raises(KeyError) as error:
                    d.loc(h, key, 'q', cint)
                assert error.value.args == (key,), (key, error.value.args)
                assert not d.contains(h, key, cint)


def test_grid_hit_and_misses_inside_and_outside_the_box(b):
    keys = [(s, a) for s in (0, 1) for a in range(20, 61) if not (s == 0 and a == 33)]
    frame = _frame(keys)
    provider, h = _handle(b, frame)
    assert h.scheme_name == 'hashed' and h.key_index == 'grid'
    probes = keys + [(0, 33),                                    # inside the box, absent
                     (2, 25), (-1, 25), (0, 19), (1, 61), (0, -5), (1, 10 ** 12)]   # outside
    _lookups(b, h, frame, probes)


def test_over_budget_keeps_the_hash_index(b):
    keys = [(0, 0), (1, 5), (1, 200_000)]                        # box 2 x 200,001 > GRID_MAX_SLOTS
    frame = _frame(keys)
    provider, h = _handle(b, frame)
    assert h.key_index.startswith('hashed: key box above'), h.key_index
    assert provider.GRID_MAX_SLOTS == 65536
    _lookups(b, h, frame, keys + [(0, 5), (2, 0)])


def test_duplicate_keys_are_refused_before_any_index(b):
    # as before: the helper Table refuses a duplicate key, so neither index ever holds one (the grid
    # builder keeps the first row of a position all the same, as the hash index would find it)
    frame = pd.DataFrame({'q': [10.0, 20.0, 30.0]}, index=pd.MultiIndex.from_tuples([(0, 1), (0, 2), (0, 2)]))
    with pytest.raises(ValueError, match='duplicate table key'):
        _handle(b, frame)


@pytest.mark.parametrize('levels', [
    [(0, 'x'), (0, 'y'), (1, 'x')],                               # a string level
    [(0, 1.5), (0, 2.5), (1, 1.5)],                               # a float level
    [(True, 1), (False, 1), (True, 2)],                           # a bool level
])
def test_non_integer_levels_are_unchanged(b, levels):
    frame = pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=pd.MultiIndex.from_tuples(levels))
    provider, h = _handle(b, frame)
    assert h.key_index.startswith('hashed'), h.key_index
    d = b.driver
    for key in levels:
        assert d.loc(h, key, 'q', False) == frame.loc[key, 'q']
