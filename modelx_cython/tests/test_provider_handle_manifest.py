"""M1 (W1.1): the optional ``handle`` declaration of a provider type."""

import pytest

from modelx_cython.provider_manifest import TypedHandle, _read_type, read_provider_manifest

TABLE = 'modelx_helpers.immutable_inputs.Table'
RECORD = 'modelx_helpers.immutable_inputs.Record'


def test_native_inputs_declares_table_and_record_handles():
    manifest = read_provider_manifest('modelx_cython.runtime._native_inputs', 'owned-table-1')
    assert manifest.types[TABLE].handle == TypedHandle('TableHandle', 'table_handle')
    assert manifest.types[RECORD].handle == TypedHandle('RecordHandle', 'record_handle')


def test_handle_is_optional_and_defaults_to_none():
    assert _read_type('pkg.mod.Plain', {}).handle is None


@pytest.mark.parametrize('handle', [
    {'type': 'TableHandle'},                                     # build missing
    {'type': 'TableHandle', 'build': 'table_handle', 'x': 'y'},  # unknown field
    {'type': 'Table Handle', 'build': 'table_handle'},           # not an identifier
    'TableHandle',
])
def test_invalid_handle_declarations_are_rejected(handle):
    with pytest.raises(ValueError):
        _read_type('pkg.mod.Plain', {'handle': handle})
