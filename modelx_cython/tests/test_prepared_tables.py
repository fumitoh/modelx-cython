"""PreparedTable (runtime/_tables.pyx) answers keyed reads exactly as pandas does (T-2).

Random keys, misses included, on every index kind the preparation accepts: dense and sparse int, text,
IntEnum, float, unsorted, two and three levels, a Series. Each read is compared with ``.loc[key, col]`` in value
and type, a miss with pandas' KeyError; ``in index`` and ``index.asof`` likewise. A typed probe module
exercises the C-level reads compiled code uses (code_ll, row_codes, prefix_range, f64_at, i64_at).
"""
import enum
import zlib
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'

PROBE = textwrap.dedent('''\
    # cython: freethreading_compatible=True
    from _tables cimport PreparedTable

    def flat_f64(PreparedTable table, long long key, Py_ssize_t slot):
        cdef Py_ssize_t row = table.code_ll(0, key) if table.nlevels == 1 else -1
        if row < 0:
            raise KeyError(key)
        return table.f64_at(row, slot)

    def codes_row(PreparedTable table, codes):
        cdef Py_ssize_t[8] buffer
        for i, c in enumerate(codes):
            buffer[i] = c
        return table.row_codes(buffer, len(codes))

    def prefix_count(PreparedTable table, codes):
        cdef Py_ssize_t[8] buffer
        cdef Py_ssize_t start = 0, stop = 0
        for i, c in enumerate(codes):
            buffer[i] = c
        if not table.prefix_range(buffer, len(codes), &start, &stop):
            return 0
        return stop - start

    def i64_value(PreparedTable table, key, Py_ssize_t slot):
        cdef Py_ssize_t row = table.row(key)
        if row < 0:
            raise KeyError(key)
        return table.i64_at(row, slot)
''')


@pytest.fixture(scope='module')
def tables(tmp_path_factory):
    folder = tmp_path_factory.mktemp('prepared')
    shutil.copy(RUNTIME / '_tables.pyx', folder / '_tables.pyx')
    shutil.copy(RUNTIME / '_tables.pxd', folder / '_tables.pxd')
    (folder / 'tables_probe.pyx').write_text(PROBE, encoding='utf-8')
    (folder / 'setup.py').write_text(
        'from setuptools import setup\nfrom Cython.Build import cythonize\n'
        'setup(ext_modules=cythonize(["_tables.pyx", "tables_probe.pyx"], '
        'compiler_directives={"freethreading_compatible": True, "language_level": 3}))\n', encoding='utf-8')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=folder, check=True,
                   stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(folder))
    try:
        import _tables
        import tables_probe
        yield _tables, tables_probe
    finally:
        sys.path.remove(str(folder))


class Plan(enum.IntEnum):
    BASIC = 1
    PLUS = 2
    MAX = 5


def frames():
    rng = np.random.default_rng(7)
    n = 40
    values = dict(q=rng.random(n), k=rng.integers(-5, 5, n).astype(np.int64),
                  s=[f'v{i}' for i in range(n)], b=rng.random(n) > 0.5)
    yield 'dense int', pd.DataFrame(values, index=pd.Index(range(20, 20 + n), name='age'))
    yield 'sparse int', pd.DataFrame(values, index=pd.Index(rng.permutation(np.arange(0, 3 * n, 3)), name='k'))
    yield 'text', pd.DataFrame(values, index=pd.Index([f'key{i}' for i in rng.permutation(n)]))
    yield 'float', pd.DataFrame(values, index=pd.Index(np.arange(n) * 0.5))
    yield 'two levels', pd.DataFrame(values, index=pd.MultiIndex.from_product([['F', 'M'], range(20)]))
    yield 'three levels enum', pd.DataFrame(
        {'q': rng.random(3 * 2 * 4), 'k': np.arange(24, dtype=np.int64)},
        index=pd.MultiIndex.from_product([list(Plan), [1, 2], [30, 31, 35, 40]]))
    yield 'unsorted two levels', pd.DataFrame(values, index=pd.MultiIndex.from_tuples(
        [(('C', 'A', 'B')[i % 3], int(x)) for i, x in enumerate(rng.permutation(n))]))


def keys_of(frame, rng):
    present = list(frame.index)
    if frame.index.nlevels == 1:
        label = present[0]
        if isinstance(label, str):
            absent = ['nokey', 'key999']
        elif isinstance(label, float):
            absent = [-1.0, 1e9, 0.25]
        else:
            absent = [-999, 10 ** 6, int(present[0]) + 10 ** 5]
    else:
        absent = [tuple('Z' if isinstance(p, str) else 999 for p in present[0])]
    return [present[int(i)] for i in rng.integers(0, len(present), 60)] + absent


def outcome(function):
    try:
        return 'ok', function()
    except KeyError as error:
        return 'KeyError', error.args
    except Exception as error:
        return type(error).__name__, str(error)


@pytest.mark.parametrize('name,frame', list(frames()), ids=[n for n, _ in frames()])
def test_value_equals_loc(tables, name, frame):
    _tables, _ = tables
    table = _tables.prepare(frame)
    assert table is not None, _tables.unprepared_reason(frame)
    assert table.source is frame
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    for key in keys_of(frame, rng):
        for column in frame.columns:
            want, got = outcome(lambda: frame.loc[key, column]), outcome(lambda: table.value(key, column))
            assert want[0] == got[0], (name, key, column, want, got)
            if want[0] == 'ok':
                assert type(want[1]) is type(got[1]) and want[1] == got[1], (name, key, column, want, got)
        assert (key in frame.index) == table.contains(key), (name, key)


def test_series(tables):
    _tables, _ = tables
    series = pd.Series(np.arange(5) * 0.1, index=pd.Index([3, 1, 4, 9, 5]), name='q')
    table = _tables.prepare(series)
    for key in [1, 3, 9, 2, 100]:
        want, got = outcome(lambda: series.loc[key]), outcome(lambda: table.value(key))
        assert want[0] == got[0] and (want[0] != 'ok' or (type(want[1]) is type(got[1]) and want[1] == got[1]))


def test_typed_reads(tables):
    _tables, probe = tables
    frame = pd.DataFrame({'q': np.arange(30) * 0.25, 'k': np.arange(30, dtype=np.int64) * 3},
                         index=pd.Index(np.arange(30) * 2 + 7))
    table = _tables.prepare(frame)
    q, k = table.column('q'), table.column('k')
    for key in list(frame.index) + [8, -1, 10 ** 9]:
        want = outcome(lambda: float(frame.loc[key, 'q']))
        got = outcome(lambda: probe.flat_f64(table, key, 0))
        assert want == got, (key, want, got)
        want = outcome(lambda: int(frame.loc[key, 'k']))
        assert want == outcome(lambda: probe.i64_value(table, key, 0)), key
    multi = pd.DataFrame({'r': np.arange(8) * 1.0},
                         index=pd.MultiIndex.from_product([['A', 'B'], [1, 2, 3, 4]]))
    table = _tables.prepare(multi)
    a = table.level_labels[0].index("A")
    assert probe.prefix_count(table, [a]) == len(multi.loc['A'])
    for key in multi.index:
        assert probe.codes_row(table, table.key_codes(key)) >= 0
    assert probe.codes_row(table, [a, 9]) == -1


def test_asof(tables):
    _tables, _ = tables
    for index in (pd.Index([20, 21, 25, 30]), pd.Index([0.5, 1.5, 3.0]), pd.Index([5, 1, 3])):
        frame = pd.DataFrame({'q': np.arange(len(index)) * 1.0}, index=index)
        table = _tables.prepare(frame)
        sorted_index = index.sort_values()
        # NaN is left out: pandas' asof of NaN differs by version (NaN on pandas 3, the last label on 2.2), so the
        # compiled reads take it to pandas (test_table_probe's asof cells check that against live)
        for x in [-10, 0.5, 1, 1.49, 2, 20, 22.5, 30, 31]:
            want = sorted_index.asof(x)
            got = table.asof(x)
            assert (want != want and got != got) or (type(want) is type(got) and want == got), (list(index), x, want, got)


def test_extremes(tables):
    _tables, _ = tables
    for frame in (pd.DataFrame({'q': [1.0, 2.0]}, index=pd.Index([3, 9])),
                  pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=pd.MultiIndex.from_tuples([('A', 1), ('B', 2), ('B', 5)]))):
        table = _tables.prepare(frame)
        assert type(table.max_label) is type(frame.index.max()) and table.max_label == frame.index.max()
        assert type(table.min_label) is type(frame.index.min()) and table.min_label == frame.index.min()


@pytest.mark.parametrize('frame,reason', [
    (pd.DataFrame({'q': [1.0, 2.0]}, index=pd.Index([1, 1])), 'not unique'),
    (pd.DataFrame({'q': [1.0, 2.0]}, index=pd.Index([1.0, np.nan])), 'NaN'),
    (pd.DataFrame([[1.0, 2.0]], columns=['q', 'q']), 'duplicate column'),
    (pd.DataFrame({'q': [1.0, 2.0]}, index=pd.MultiIndex.from_tuples([('A', 1), ('A', 1)])), 'not unique'),
])
def test_left_to_pandas(tables, frame, reason):
    _tables, _ = tables
    assert _tables.prepare(frame) is None
    assert reason in _tables.unprepared_reason(frame)


def test_copy_at_bind(tables):
    _tables, _ = tables
    frame = pd.DataFrame({'q': [1.0, 2.0, 3.0]}, index=pd.Index([10, 11, 12]))
    table = _tables.prepare(frame)
    frame.loc[11, 'q'] = 99.0                   # an in-place edit after binding: the prepared copy is unchanged
    assert table.value(11, 'q') == 2.0 and table.source is frame


@pytest.mark.parametrize('name,frame', list(frames()), ids=[n for n, _ in frames()])
def test_hundred_thousand_reads(tables, name, frame):
    """10^5 random keys per index kind (about 10% absent), one float and one int column: 0 differences."""
    _tables, _ = tables
    table = _tables.prepare(frame)
    rng = np.random.default_rng(zlib.crc32(name.encode()) + 1)
    present = list(frame.index)
    absent = keys_of(frame, rng)[-1]
    expected = {key: (float(frame.loc[key, 'q']), int(frame.loc[key, 'k'])) for key in present}
    picks = rng.integers(0, len(present) + max(1, len(present) // 9), 100_000)
    for pick in picks:
        key = present[pick] if pick < len(present) else absent
        want = expected.get(key)
        row = table.row(key)
        if want is None:
            assert row < 0, (name, key)
            continue
        assert (table.value(key, 'q'), table.value(key, 'k')) == want, (name, key)


def test_check_mode_detects_an_edit(tables):
    """T-4 verify_inputs: a frame edited in place after it was bound is detected; an unchanged one passes."""
    _tables, _ = tables
    frame = pd.DataFrame({'q': [1.0, 2.0, 3.0], 'k': np.arange(3, dtype=np.int64)}, index=pd.Index([10, 11, 12]))
    _tables.check_unchanged(frame)          # binds it, then checks: unchanged
    _tables.check_unchanged(frame)
    frame.loc[11, 'q'] = 9.0
    with pytest.raises(RuntimeError, match='edited in place'):
        _tables.check_unchanged(frame)
