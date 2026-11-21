"""Raw scalar snapshots defer conversion and preserve pandas object identity."""
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

from modelx_helpers import pandas_inputs as inputs, typed_inputs as h
from modelx_helpers.immutable_inputs import Table, Record
from modelx_helpers.table_operations import row, to_pandas
from modelx_cython.tests.helper_contracts.test_table_operations import native


class UnexpectedConversion:
    def __float__(self):
        raise AssertionError('unused value converted')


@pytest.mark.parametrize('dtype,values', [('int64', [2**53+1, 4]),
    ('float64', [0.1, np.nan]), ('bool', [True, False]),
    ('Int64', [42, pd.NA]), ('object', [None, pd.NaT])])
def test_column_scalar_types_and_missing_identity(dtype, values):
    frame = pd.DataFrame({'value': pd.Series(values, dtype=dtype)})
    prepared = inputs.table(frame, preserve_scalar_types=True, report=True)
    for restored in [prepared, pickle.loads(pickle.dumps(prepared))]:
        pd.testing.assert_frame_equal(to_pandas(restored), frame, check_exact=True)
        for key in frame.index:
            actual, expected = h.lookup_object(restored, key, 'value'), frame.loc[key, 'value']
            assert type(actual) is type(expected)
            if expected is pd.NA or expected is pd.NaT or expected is None:
                assert actual is expected
            else:
                assert actual == expected or np.isnan(actual) and np.isnan(expected)


def test_unused_opaque_values_do_not_raise_and_selected_conversion_does(native):
    unused = UnexpectedConversion()
    frame = pd.DataFrame({'value': pd.Series([np.float64(0.1), unused], dtype=object),
                          'other': pd.Series([object(), object()], dtype=object)})
    prepared = inputs.table(frame, preserve_scalar_types=True, rows=True)
    forbidden = {h.lookup_object.__code__, h.lookup_float.__code__,
                 Table.lookup.__code__, Table.__init__.__code__, Record.__init__.__code__,
                 row.__code__, UnexpectedConversion.__float__.__code__}
    calls = []
    def profile(frame, event, arg):
        if event == 'call' and (frame.f_code in forbidden or
                frame.f_globals.get('__name__', '').startswith('pandas')):
            calls.append(frame.f_code.co_name)
    sys.setprofile(profile)
    try:
        selected = native.select(row, prepared, 0)
    finally:
        sys.setprofile(None)
    assert calls == []
    assert type(selected['value']) is np.float64 and selected['value'] == 0.1
    with pytest.raises(TypeError, match='unsupported immutable input'):
        inputs.table(frame)
    for restored in [pickle.loads(pickle.dumps(prepared)),
                     pickle.loads(pickle.dumps(native.select(row, prepared, 1)))]:
        value = h.lookup_object(restored, 1, 'value') if isinstance(restored, Table) else restored['value']
        assert type(value) is UnexpectedConversion


def test_record_raw_option_is_consistent_and_container_snapshot_is_shallow():
    mutable = ['original']
    series = pd.Series([np.float64(0.1), mutable], index=['rate', 'opaque'], dtype=object)
    prepared = inputs.record(series, preserve_scalar_types=True)
    assert prepared['opaque'] is mutable
    assert type(prepared['rate']) is np.float64
    series['rate'] = 0.9
    mutable.append('edited')
    assert prepared['rate'] == 0.1 and prepared['opaque'] == ['original', 'edited']
    restored = pickle.loads(pickle.dumps(prepared))
    assert restored['opaque'] == mutable
    assert type(restored['rate']) is np.float64


def test_empty_and_grouped_raw_storage():
    frame = pd.DataFrame({'rate': pd.Series([], dtype='float64')})
    pd.testing.assert_frame_equal(to_pandas(inputs.table(frame, report=True,
        preserve_scalar_types=True)), frame, check_exact=True)
    frame = pd.DataFrame({'rate': pd.Series([np.float64(0.1), object()], dtype=object).array},
        index=pd.MultiIndex.from_tuples([('M', 1), ('F', 2)], names=['sex', 'age']))
    prepared = inputs.table(frame, groups=('sex',), preserve_scalar_types=True)
    assert type(h.lookup_object(prepared['M'], 1, 'rate')) is np.float64
    assert type(h.lookup_object(pickle.loads(pickle.dumps(prepared))['F'], 2, 'rate')) is object
