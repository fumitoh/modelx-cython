import json
import sys

import pytest

from modelx_cython.helpers import registered_helper
from modelx_cython.provider_manifest import owner_provider_binding, read_provider_manifest


PROVIDER = 'rates_demo.compiled.curve_kernel'
ABI = 'curve-2026-01'
TYPE = 'rates_demo.curves.Curve'


@pytest.fixture
def declaration(tmp_path, monkeypatch):
    package = tmp_path / 'rates_demo' / 'compiled'
    package.mkdir(parents=True)
    (package.parent / '__init__.py').touch()
    (package / '__init__.py').touch()
    (package / 'curve_kernel.py').write_text(
        'raise AssertionError("the provider module should not be imported")',
        encoding='utf-8',
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    document = {
        'protocol': 1,
        'provider': PROVIDER,
        'abi': ABI,
        'types': {
            TYPE: {
                'view_type': 'CurveReceiver',
                'bind': 'as_receiver',
            },
            'rates_demo.curves.Labels': {},
        },
    }
    try:
        yield package / 'curve_kernel.provider.json', document
    finally:
        for name in ('rates_demo.compiled', 'rates_demo'):
            sys.modules.pop(name, None)


def test_read_independent_provider_manifest(declaration):
    resource, document = declaration
    resource.write_text(json.dumps(document), encoding='utf-8')

    manifest = read_provider_manifest(PROVIDER, ABI)

    assert (manifest.protocol, manifest.provider, manifest.abi) == (1, PROVIDER, ABI)
    curve = manifest.types[TYPE]
    assert (curve.view_type, curve.bind) == ('CurveReceiver', 'as_receiver')
    assert (curve.admission, curve.bind_cell) == ('observed', False)
    assert manifest.types['rates_demo.curves.Labels'].bind is None
    assert PROVIDER not in sys.modules
    with pytest.raises(TypeError):
        manifest.types['another'] = curve


def test_function_provider_does_not_need_a_manifest(declaration):
    def plain_lookup(value):
        return value

    plain_lookup.__modelx_cython_helper__ = (PROVIDER, 'lookup')
    assert registered_helper(plain_lookup) == (PROVIDER, 'lookup')
    assert read_provider_manifest(PROVIDER, ABI) is None


def test_missing_provider_package():
    assert read_provider_manifest('uninstalled_rates_fixture.kernel', ABI) is None


@pytest.mark.parametrize('provider', ['kernel', 'rates_demo..kernel', 'rates_demo.import',
                                      'rates_demo/compiled.kernel'])
def test_invalid_lookup_identity(provider):
    with pytest.raises(ValueError, match='invalid packaged provider identity'):
        read_provider_manifest(provider, ABI)


@pytest.mark.parametrize(('field', 'value', 'error'), [
    ('protocol', True, 'unsupported protocol'),
    ('protocol', 2, 'unsupported protocol'),
    ('provider', 'other_demo.compiled.curve_kernel', 'declares provider'),
    ('abi', 'unknown-abi', 'declares ABI.*expected'),
    ('abi', 'bad abi', 'invalid ABI identity'),
    ('types', {'not-a-qualified-type': {}}, 'invalid qualified type identity'),
    ('types', {TYPE: {'view_type': 'CurveReceiver'}}, 'view_type and bind'),
    ('types', {TYPE: {'members': {}}}, 'invalid fields'),
])
def test_invalid_manifest_is_distinct_from_missing(declaration, field, value, error):
    resource, document = declaration
    document[field] = value
    resource.write_text(json.dumps(document), encoding='utf-8')

    with pytest.raises(ValueError, match=error):
        read_provider_manifest(PROVIDER, ABI)


def test_duplicate_fields_are_rejected(declaration):
    resource, document = declaration
    resource.write_text(json.dumps(document).replace('{', '{"protocol": 2, ', 1),
                        encoding='utf-8')

    with pytest.raises(ValueError, match="duplicate field 'protocol'"):
        read_provider_manifest(PROVIDER, ABI)


def test_binding_on_an_exact_owner_without_a_view(declaration):
    resource, document = declaration
    document['types'][TYPE] = {}
    resource.write_text(json.dumps(document), encoding='utf-8')
    owner = type('Curve', (), {'__modelx_cython_type__': (1, PROVIDER, ABI, TYPE)})

    assert owner_provider_binding(owner).declaration.view_type is None
    assert owner_provider_binding(type('Derived', (owner,), {})) is None


def test_broken_package_dependency_is_reported(declaration):
    resource, _ = declaration
    (resource.parent / '__init__.py').write_text(
        'import missing_rates_fixture_dependency', encoding='utf-8',
    )

    with pytest.raises(ModuleNotFoundError, match='missing_rates_fixture_dependency'):
        read_provider_manifest(PROVIDER, ABI)
