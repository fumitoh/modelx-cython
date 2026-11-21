"""M1 (W1.1): the tracer reads a traced Table's kinds through the helper's table_schema(t)."""

import types

import pytest

from modelx_cython.tracer import table_schema

def _traces(*values):
    return [types.SimpleNamespace(ret_val=value) for value in values]


def test_non_tables_have_no_schema():
    assert table_schema(_traces(1.0)) is None
    assert table_schema([]) is None


def test_schema_comes_from_the_helper_and_merges_row_counts():
    immutable_inputs = pytest.importorskip('modelx_helpers.immutable_inputs')
    if not hasattr(immutable_inputs, 'table_schema'):
        pytest.skip('helper without the table_schema contract (round-2 helper)')
    pd = pytest.importorskip('pandas')
    from modelx_helpers import pdn
    short = pdn.table(pd.DataFrame({'q': [1., 2.]}, index=[5, 6]))
    long = pdn.table(pd.DataFrame({'q': [1., 2., 3.]}, index=[5, 6, 7]))
    assert table_schema(_traces(long)) == immutable_inputs.table_schema(long)
    merged = table_schema(_traces(long, short, long))
    assert merged['rows'] is None and merged['columns'] == {'q': 'float'}
    mixed = pdn.table(pd.DataFrame({'q': ['a', 'b']}, index=[5, 6]))
    assert table_schema(_traces(long, mixed)) is None
