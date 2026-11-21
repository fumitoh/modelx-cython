"""Exact public contracts and native row/column selection over existing owners."""
import importlib
import math
from pathlib import Path
import pickle
import subprocess
import sys

from modelx_cython import runtime as native_runtime
import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pandas_inputs as inputs, typed_inputs as helpers
from modelx_helpers.immutable_inputs import Record, Table
from modelx_helpers.table_keys import keys
from modelx_helpers.table_operations import row, column, to_pandas


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp('native-table-operations')
    package = Path(native_runtime.__file__).parent
    for suffix in ['pyx', 'pxd']:
        (root / ('provider.' + suffix)).write_bytes((package / ('_native_inputs.' + suffix)).read_bytes())
    (root / 'probe.pyx').write_text('''cimport provider as p

def select(function, source, key):
    return p.row(function, (), source, key)

def select_keyword(function, source, key):
    return p.row(function, ('key', 'table'), key=key, table=source)

def wrap_table(source):
    return p.NativeTable(source)
''')
    (root / 'setup.py').write_text('''from pathlib import Path
from setuptools import setup
from Cython.Build import cythonize
import modelx_cython
setup(ext_modules=cythonize(['provider.pyx','probe.pyx'],include_path=[str(Path(modelx_cython.__file__).resolve().parent.parent)],compiler_directives={'language_level':3,'freethreading_compatible':True}))
''')
    run = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=root,
                         capture_output=True, text=True)
    (root / 'build.log').write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    sys.path.insert(0, str(root))
    try:
        yield importlib.import_module('probe')
    finally:
        sys.path.remove(str(root))
        sys.modules.pop('probe', None); sys.modules.pop('provider', None)


def frames():
    yield pd.DataFrame({'age': [2**53+1, 42], 'rate': [0.1, np.nan]}, index=pd.Index([9, 2], name='policy'))
    yield pd.DataFrame({'age': [42, 43], 'label': ['M', 'F'], 'flag': [True, False]}, index=['p', 'q'])
    yield pd.DataFrame({'age': pd.array([42, pd.NA], dtype='Int64'),
                        'rate': pd.array([0.1, pd.NA], dtype='Float64')}, index=['p', 'q'])
    yield pd.DataFrame({'label': pd.array(['M', pd.NA], dtype='string'),
                        'flag': pd.array([True, pd.NA], dtype='boolean')})
    yield pd.DataFrame({'x': [1.0, 2.0]}, index=pd.MultiIndex.from_tuples([('M', 40), ('F', 41)], names=['sex', 'age']))
    yield pd.DataFrame({'x': pd.Series([], dtype='Int64'), 'y': pd.Series([], dtype='string')})
    yield pd.DataFrame({'x': pd.Series([], dtype='float64')}, index=pd.RangeIndex(7, 7, name='t'))


@pytest.mark.parametrize('frame', list(frames()))
def test_exact_output_rows_values_and_pickle(native, frame):
    frame.attrs = {'scenario': ['original']}
    frame.columns.name = 'cashflow'
    frame.flags.allows_duplicate_labels = False
    owned = inputs.table(frame, rows=True)
    for source in [owned, pickle.loads(pickle.dumps(owned))]:
        pd.testing.assert_frame_equal(to_pandas(source), frame, check_exact=True)
        assert to_pandas(source).attrs == frame.attrs
        assert to_pandas(source).flags.allows_duplicate_labels is False
        for key in frame.index:
            expected = frame.loc[key]
            for selected in [row(source, key), native.select(row, source, key),
                             native.select_keyword(row, source, key)]:
                assert type(selected) is Record
                assert keys(selected) == tuple(frame.columns)
                actual = to_pandas(selected)
                pd.testing.assert_series_equal(actual, expected, check_exact=True)
                assert actual.attrs == expected.attrs
                assert actual.flags.allows_duplicate_labels == expected.flags.allows_duplicate_labels
                pd.testing.assert_series_equal(to_pandas(pickle.loads(pickle.dumps(selected))), expected, check_exact=True)
                for name in frame.columns:
                    a, b = selected[name], expected[name]
                    assert type(a) is type(b)
                    if b is pd.NA: assert a is b
                    elif isinstance(b, float) and np.isnan(b): assert np.isnan(a)
                    else: assert a == b


def test_legacy_row_metadata_remains_readable(native):
    frame = pd.DataFrame({'age': [42], 'rate': [0.1]}, index=['policy'])
    source = inputs.table(frame, rows=True)
    object.__setattr__(source, '_metadata', source._metadata[:10])
    legacy = pickle.loads(pickle.dumps(source))
    for selected in (row(legacy, 'policy'), native.select(row, legacy, 'policy')):
        pd.testing.assert_series_equal(to_pandas(selected), frame.loc['policy'], check_exact=True)


def test_integer_row_lookup_does_not_alias_a_boolean_label(native):
    source = inputs.table(pd.DataFrame({'rate': [0.1, 0.2]}, index=[False, True]), rows=True)
    with pytest.raises(KeyError):
        native.select(row, source, 1)


def test_compiled_selection_has_no_python_helper_pandas_or_owner_calls(native):
    source = inputs.table(pd.DataFrame({'age': [42, 43], 'rate': [0.1, 0.2]}), rows=True)
    forbidden = {row.__code__, column.__code__, Table.lookup.__code__,
                 Table.__getitem__.__code__, Record.__init__.__code__, Record.__getitem__.__code__}
    calls = []
    def profile(frame, event, arg):
        if event == 'call' and (frame.f_code in forbidden or
                              frame.f_globals.get('__name__', '').startswith('pandas')):
            calls.append((frame.f_globals.get('__name__'), frame.f_code.co_name))
    sys.setprofile(profile)
    try:
        for _ in range(2):
            selected = native.select(row, source, 0)
    finally:
        sys.setprofile(None)
    assert calls == []
    assert selected['age'] == 42.0


def test_replacement_layout_and_errors(native):
    source = inputs.table(pd.DataFrame({'rate': [0.1]}, index=[3]), rows=True)
    observed = []
    def current(*, key, table):
        observed.append((key, table))
        return 'replacement'
    assert native.select_keyword(current, source, 3) == 'replacement'
    assert observed == [(3, source)]
    with pytest.raises(TypeError): native.select(current, source, 3)
    for operation in [row, lambda table, key: native.select(row, table, key)]:
        with pytest.raises(KeyError) as error: operation(source, 99)
        assert error.value.args == (99,)
    for operation in [column]:
        with pytest.raises(KeyError) as error: operation(source, 'absent')
        assert error.value.args == ('absent',)
    with pytest.raises(ValueError, match='metadata'): row(Table([3], [[0.1]], ['rate']), 3)
    expected = pd.Series([0.1], index=pd.Index(['rate'], dtype=object), dtype=object)
    pd.testing.assert_series_equal(to_pandas(Record({'rate': 0.1})), expected)
    with pytest.raises(TypeError): to_pandas({'rate': 0.1})


def test_output_mutation_does_not_change_owned_snapshot():
    source = inputs.table(pd.DataFrame({'x': [1.0]}, index=pd.Index([3], name='t')), report=True)
    public = to_pandas(source)
    public.iloc[0, 0] = 9.0
    public.index.name = 'changed'
    assert to_pandas(source).iloc[0, 0] == 1.0
    assert to_pandas(source).index.name == 't'
    assert to_pandas(source, index_name=None).index.name is None


def test_row_snapshots_are_explicit_and_follow_selected_columns(native):
    frame = pd.DataFrame({'age': [42], 'rate': [0.1]})
    lookup_only = inputs.table(frame)
    pd.testing.assert_frame_equal(to_pandas(lookup_only), frame, check_exact=True)
    report_only = inputs.table(frame, report=True)
    assert report_only._metadata[4] is None
    pd.testing.assert_frame_equal(to_pandas(report_only), frame, check_exact=True)
    for operation in [row, lambda table, key: native.select(row, table, key)]:
        with pytest.raises(ValueError, match='rows=True'): operation(lookup_only, 0)
    selected = inputs.table(frame, columns=('age',), rows=True)
    expected = frame[['age']].loc[0]
    pd.testing.assert_series_equal(to_pandas(native.select(row, selected, 0)), expected, check_exact=True)
    assert type(row(selected, 0)['age']) is np.int64
    with pytest.raises(ValueError, match='leave at least'):
        inputs.table(frame.rename_axis('t'), groups=('t',), rows=True)


@pytest.mark.parametrize('index', [pd.Index([1, 2]), pd.Index([False, True]),
    pd.Index([False, True], dtype=object), pd.Index([False, 'a'], dtype=object)])
@pytest.mark.parametrize('key', [False, True, np.bool_(False), np.bool_(True)])
def test_boolean_row_key_error_matches_original_loc(native, index, key):
    frame = pd.DataFrame({'rate': [0.1, 0.2]}, index=index)
    source = inputs.table(frame, rows=True)
    try:
        expected = frame.loc[key]
    except KeyError as error:
        expected_args = error.args
        for operation in [lambda: row(source, key), lambda: native.select(row, source, key)]:
            with pytest.raises(KeyError) as actual:
                operation()
            assert actual.value.args == expected_args
    else:
        pd.testing.assert_series_equal(to_pandas(native.select(row, source, key)), expected, check_exact=True)


@pytest.mark.parametrize('fault', ['hash', 'ordering', 'type comparison'])
@pytest.mark.parametrize('indexed', [False, True])
def test_native_views_leave_unread_object_keys_untouched(native, fault, indexed):
    armed = False

    class LabelType(type):
        def __eq__(self, other):
            if armed and fault == 'type comparison':
                raise AssertionError('native admission compared the unread key type')
            return super().__eq__(other)

    class UnreadKey(int, metaclass=LabelType):
        def __hash__(self):
            if armed and fault == 'hash':
                raise AssertionError('native admission rehashed an unread key')
            return super().__hash__()

        def __lt__(self, other):
            if armed and fault == 'ordering':
                raise AssertionError('native admission ordered an unread key')
            return super().__lt__(other)

        def __gt__(self, other):
            if armed and fault == 'ordering':
                raise AssertionError('native admission ordered an unread key')
            return super().__gt__(other)

    spec = (inputs.table(pd.DataFrame({'rate': [1.5, 2.5]}, index=[1, 23]), indexing='at')._index_error
            if indexed else None)
    cases = [
        (lambda table: 1 in native.wrap_table(table).index, True),
    ]
    for operation, expected in cases:
        armed = False
        table = Table((1, UnreadKey(23)), ((1.5,), (2.5,)), ('rate',), spec)
        armed = True
        assert helpers.lookup(table, 1, 'rate') == 1.5
        assert operation(table) == expected


def test_rectangular_storage_rejects_duplicate_axis_labels():
    index = pd.MultiIndex(levels=[['F', 'F'], [0, 1]],
                          codes=[[0, 0, 1, 1], [0, 1, 0, 1]], verify_integrity=False)
    frame = pd.DataFrame({'rate': [1.0, 2.0, 3.0, 4.0]}, index=index)
    with pytest.raises(ValueError, match='duplicate table key'):
        inputs.table(frame, columns=('rate',))
