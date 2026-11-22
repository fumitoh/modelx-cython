import pandas as pd
import pytest

from modelx_cython import read_notes


def test_mixed_subscripts_retain_exact_counts_and_veto_evidence():
    class _c_Data:
        def table(self):
            return pd.DataFrame({'rate': [0.1, 0.2]}, index=[10, 20])

    site = ('_c_Projection', '_f_get_rate', 'receiver[key]', 'sub')
    raised_site = ('_c_Projection', '_f_missing', 'receiver[key]', 'sub')
    with read_notes.recording('unused'):
        frame = read_notes.r_call(('_c_Projection', '_f_get_rate', 'self.data.table()', 'call'),
                                  _c_Data(), 'table')
        assert read_notes.r_sub(site, [0.1], 0) == 0.1
        assert read_notes.r_sub(site, [0.2], 0) == 0.2
        assert read_notes.r_sub(site, ['x'], 0) == 'x'
        with pytest.raises(IndexError):
            read_notes.r_sub(site, [], 0)
        assert read_notes.r_sub(site, frame.loc, ([10, 20], 'rate')).tolist() == [0.1, 0.2]
        with pytest.raises(KeyError):
            read_notes.r_sub(raised_site, frame.loc, (30, 'rate'))
        notes = read_notes.by_text()
        mixed = notes[site[:2]][site[2]]
        raised = notes[raised_site[:2]][raised_site[2]]
        assert mixed['count'] == 5
        assert mixed['table_count'] == 1
        assert raised['table_count'] == 1
        assert mixed['seen'] == {
            ('list', 'num', 'float', None), ('list', 'num', 'str', None),
            ('list', 'num', read_notes.RAISES, None), ('loc:table:Data.table', ('list', 'str'), 'series', 'rate'),
        }
        assert any(read_notes.selects_rows(observation[1]) for observation in mixed['seen'])
        assert not read_notes.only_raised(mixed)
        assert read_notes.only_raised(raised)


def test_cached_cell_call_still_follows_pandas_result_and_records_exceptions():
    class _c_Data:
        def values(self):
            if isinstance(self.value, Exception):
                raise self.value
            return self.value

    site = ('_c_Projection', '_f_values', 'self.data.values()', 'call')
    receiver = _c_Data()
    with read_notes.recording('unused'):
        receiver.value = [42]
        for _ in range(3):
            assert read_notes.r_call(site, receiver, 'values') == [42]
        receiver.value = pd.Series([0.1, 0.2])
        assert read_notes.r_call(site, receiver, 'values') is receiver.value
        receiver.value = ValueError('missing')
        with pytest.raises(ValueError, match='missing'):
            read_notes.r_call(site, receiver, 'values')
        assert read_notes.COUNTS[site] == 5
        assert read_notes.NOTES[site] == {
            ('_c_Data', 'values', 'list'), ('_c_Data', 'values', 'col:Data.values'),
            ('_c_Data', 'values', read_notes.RAISES),
        }
