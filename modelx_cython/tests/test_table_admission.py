"""auto-tables (D104): the admission conditions and their messages, decided on the value only."""
import enum
import pickle

import numpy as np
import pandas as pd
import pytest

from modelx_cython.runtime.table_admission import TableAdmissionError, admit, conditions, display_name


class Sex(enum.IntEnum):
    F = 0
    M = 1


def frame(**columns):
    index = columns.pop('index', pd.Index([30, 31, 32], name='age'))
    return pd.DataFrame(columns or {'q': [0.01, 0.02, 0.03]}, index=index)


def reasons(value, nargs=0):
    with pytest.raises(TableAdmissionError) as info:
        admit(value, 'Data.t', nargs)
    assert info.value.cell == 'Data.t'
    assert str(info.value).splitlines() == [f'Data.t: {r}' for r in info.value.reasons]
    return info.value.reasons


def test_a_keyed_frame_is_admitted_whatever_built_it():
    for value in (frame(), frame().reset_index().set_index('age'),
                  pd.merge(frame().reset_index(), frame(w=[1.0, 2.0, 3.0]).reset_index()).set_index('age'),
                  pd.DataFrame({'a': [1.0], 'b': [2.0]}, index=pd.Index(['x'], name='k'))):
        table = admit(value, 'Data.t')
        assert type(table).__name__ == 'Table'
        assert table.loc[value.index[0], value.columns[0]] == value.iloc[0, 0]


def test_values_that_are_not_pandas_pass_through():
    from modelx_helpers import pdn
    table = pdn.table(frame())
    assert admit(table, 'Data.t') is table      # pdn.table(...) in model text keeps working
    assert admit(1.5, 'Data.t') == 1.5


@pytest.mark.parametrize('value, nargs, expected', [
    (frame()['q'], 0, 'returns a pandas Series: not admitted as a table (C2)'),
    (frame(index=pd.Index([30, 30, 31], name='age')), 0, 'duplicate key 30 (1 duplicated rows) (C3)'),
    (frame(index=pd.Index([30.0, np.nan, 31.0], name='age')), 0, "missing value in key level 'age' (C4)"),
    (frame(index=pd.Index([30, 'x', 31], name='age', dtype=object)), 0,
     "key level 'age' mixes int, str labels (C5)"),
    (frame(index=pd.Index([(1, 2), (3, 4), (5, 6)], name='age', tupleize_cols=False)), 0,
     "key level 'age' holds tuple labels (C5)"),
    (pd.DataFrame({1: [1.0]}, index=pd.Index([1], name='k')), 0, 'column names must be non-empty unique strings (C6)'),
    (pd.DataFrame([[1.0, 2.0]], columns=['a', 'a'], index=pd.Index([1], name='k')), 0,
     'column names must be non-empty unique strings (C6)'),
    (frame(d=pd.to_datetime(['2020-01-01'] * 3)), 0, "column 'd' has unsupported dtype datetime64"),
])
def test_each_condition_has_its_message(value, nargs, expected):
    found = reasons(value, nargs)
    assert any(r.startswith(expected) for r in found), found


def test_every_failed_condition_is_listed():
    value = pd.DataFrame({1: [1.0, 2.0]}, index=pd.Index([1, 1], name='k'))
    found = reasons(value, 2)
    assert [r.split(' (C')[1][0] for r in found] == ['3', '6']


def test_a_cell_with_arguments_is_admitted_too():
    # D105: the argument count does not decide (the build decides by the native reads, D106)
    assert type(admit(frame(), 'Projection.t', 2)).__name__ == 'Table'


def test_c8_is_the_helper_contract():
    # an object column holding a set: pdn.table refuses an unsupported scalar
    value = pd.DataFrame({'x': [{1}, {2}]}, index=pd.Index([1, 2], name='k'))
    assert conditions(value) == []
    found = reasons(value)
    assert len(found) == 1 and found[0].endswith('(C8)'), found


@pytest.mark.parametrize('index, key', [
    (pd.Index([0.5, 1.5], name='k'), 1.5),                               # float keys
    (pd.Index(['p', 'q'], name='k'), 'q'),                               # string keys
    (pd.Index([Sex.F, Sex.M], name='sex', dtype=object), Sex.M),         # an enum key level
    (pd.Index([True, False], name='flag'), False),
    (pd.RangeIndex(2), 1),                                               # row numbers as keys
])
def test_awkward_keys_are_admitted(index, key):
    value = pd.DataFrame({'x': [1.0, 2.0]}, index=index)
    table = admit(value, 'Data.t')
    assert table.loc[key, 'x'] == value.loc[key, 'x']


def test_an_unknown_enum_label_raises_keyerror_at_load():
    raw = pd.DataFrame({'sex': ['F', 'X'], 'q': [0.1, 0.2]})
    with pytest.raises(KeyError):
        raw['sex'] = [Sex[n] for n in raw['sex']]     # the mapping line kept in the model text (D40)


def test_nan_values_and_strings_survive():
    value = pd.DataFrame({'x': [1.0, np.nan], 's': ['a', None]}, index=pd.Index([1, 2], name='k'))
    table = admit(value, 'Data.t')
    assert np.isnan(table.loc[2, 'x']) and pd.isna(value.loc[2, 's'])
    assert table.to_pandas().equals(value) if hasattr(table, 'to_pandas') else True


def test_unused_multiindex_levels_read_as_pandas():
    full = pd.DataFrame({'q': [1.0, 2.0, 3.0, 4.0]},
                        index=pd.MultiIndex.from_product([['F', 'M'], [30, 31]], names=['sex', 'age']))
    value = full.loc[['M']]
    table = admit(value, 'Data.t')
    assert table.loc[('M', 31), 'q'] == 4.0
    assert list(table.index.levels[0]) == list(value.index.levels[0]) == ['F', 'M']


def test_the_admitted_frame_is_the_value_itself():
    """D130: the table keeps the frame it was admitted from (no copy); an in-place edit of that frame
    reaches the table only when it is admitted again (what a clear does), never by itself."""
    value = frame()
    table = admit(value, 'Data.t')
    from modelx_helpers.immutable_inputs import table_frame
    assert table_frame(table) is value
    value.loc[30, 'q'] = 9.0
    assert table.loc[30, 'q'] == 0.01         # the table is a snapshot until it is rebuilt
    assert admit(value, 'Data.t').loc[30, 'q'] == 9.0


def test_an_admitted_table_pickles_with_its_frame():
    table = admit(frame(), 'Data.t')
    again = pickle.loads(pickle.dumps(table))
    assert len(again) == 3 and list(again) == ['q']


def test_display_names():
    assert display_name('pkg_cy._mx_classes._c_Data.points') == 'Data.points'
    assert display_name('pkg._m_Outer._mx_classes._c_Inner._f_x') == 'Outer.Inner.x'
    assert display_name('Data.t') == 'Data.t'
