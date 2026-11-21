"""Grouped leaf reports retain pandas partial-selection axes and Series schema."""
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pandas_inputs as inputs
from modelx_helpers.table_operations import row, column, to_pandas
from modelx_helpers.immutable_inputs import Table, Record
from modelx_cython.tests.helper_contracts.test_table_operations import native


@pytest.mark.parametrize('rows', [False, True])
def test_grouped_leaf_keeps_remaining_levels_order_and_public_schema(rows):
    index = pd.MultiIndex.from_tuples([('M', 'N', 42), ('F', 'Y', 40),
                                      ('M', 'N', 40), ('M', 'Y', 41)],
                                     names=['sex', 'smoker', 'age'])
    frame = pd.DataFrame({'mort_rate': [0.2, 0.3, np.nan, 0.4],
                          'count': [1, 2, 3, 4]}, index=index)
    frame.columns.name = 'basis'
    frame.attrs = {'source': ['original']}
    frame.flags.allows_duplicate_labels = False
    prepared = inputs.table(frame, groups=('sex',), report=True, rows=rows)
    for owned in [prepared, pickle.loads(pickle.dumps(prepared))]:
        for sex in ['M', 'F']:
            expected = frame.loc[sex]
            leaf = owned[sex]
            pd.testing.assert_frame_equal(to_pandas(leaf), expected, check_exact=True)
            assert to_pandas(leaf).flags.allows_duplicate_labels is False
            for name in frame.columns:
                pd.testing.assert_series_equal(to_pandas(column(leaf, name, report=True)), expected[name], check_exact=True)
            if rows:
                for key in expected.index:
                    pd.testing.assert_series_equal(to_pandas(row(leaf, key)), expected.loc[key], check_exact=True)


def test_nonleading_group_order_retains_exact_remaining_index():
    index = pd.MultiIndex.from_tuples([('M', 42, 'N'), ('F', 40, 'Y'), ('M', 40, 'N')],
                                     names=['sex', 'age', 'smoker'])
    frame = pd.DataFrame({'rate': [0.2, 0.3, 0.1]}, index=index)
    prepared = inputs.table(frame, groups=('smoker', 'sex'), report=True)
    expected = frame.xs(('N', 'M'), level=('smoker', 'sex'))
    pd.testing.assert_frame_equal(to_pandas(prepared['N']['M']), expected, check_exact=True)
    pd.testing.assert_series_equal(to_pandas(column(prepared['N']['M'], 'rate', report=True)), expected['rate'], check_exact=True)


@pytest.mark.parametrize('dtype,values', [('int64', [42, 43]), ('float64', [0.1, np.nan]),
    ('bool', [True, False]), ('Int64', [42, pd.NA]), ('object', [None, pd.NaT])])
@pytest.mark.parametrize('empty', [False, True])
def test_series_output_retains_dtype_missingness_name_and_empty_axes(dtype, values, empty):
    frame = pd.DataFrame({'rate': pd.Series(values, dtype=dtype)})
    frame.index = pd.Index([9, 2], name='policy_year')
    if empty:
        frame = frame.iloc[:0]
    prepared = inputs.table(frame, report=True)
    for owned in [prepared, pickle.loads(pickle.dumps(prepared))]:
        pd.testing.assert_series_equal(to_pandas(column(owned, 'rate', report=True)), frame['rate'], check_exact=True)
        assert to_pandas(column(owned, 'rate', report=True), index_name=None).index.name is None
        with pytest.raises(KeyError) as error:
            to_pandas(column(owned, 'absent', report=True))
        assert error.value.args == ('absent',)


def test_grouped_rows_raw_opaque_values_and_output_independence():
    opaque = ['original']
    frame = pd.DataFrame({'rate': pd.Series([np.float64(0.1), object()], dtype=object).array,
                          'value': pd.Series([opaque, object()], dtype=object).array},
        index=pd.MultiIndex.from_tuples([('M', 1), ('F', 2)], names=['sex', 'age']))
    prepared = inputs.table(frame, groups=('sex',), rows=True, preserve_scalar_types=True)
    selected = row(prepared['M'], 1)
    assert selected['value'] is opaque and type(selected['rate']) is np.float64
    result = to_pandas(column(prepared['M'], 'rate', report=True))
    result.iloc[0] = 9.0
    assert to_pandas(column(prepared['M'], 'rate', report=True)).iloc[0] == 0.1
    with pytest.raises(TypeError, match='owned Table'):
        column(selected, 'rate', report=True)


def test_grouped_missing_leaf_label_uses_owned_index_missing_semantics():
    frame = pd.DataFrame({'rate': [0.1, 0.2]}, index=pd.MultiIndex.from_tuples(
        [('M', np.nan), ('M', 1.0)], names=['sex', 'age']))
    leaf = inputs.table(frame, groups=('sex',), report=True, rows=True)['M']
    pd.testing.assert_series_equal(to_pandas(row(leaf, np.nan)), frame.loc['M'].loc[np.nan], check_exact=True)
