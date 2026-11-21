"""An independent provider's view type binds through forwarded Cells, behind its ABI guard."""

from importlib import import_module
import json
import shutil
import subprocess
import sys
from textwrap import dedent

import modelx as mx

from modelx_cython.helpers import module_name



PROVIDER = 'curve_example.kernel'
ABI = 'curve-v1'
TYPE = 'curve_example.Curve'
OWNER = '''\
class Curve:
    __modelx_cython_type__ = (1, 'curve_example.kernel', 'curve-v1', 'curve_example.Curve')

    def __init__(self, level):
        self.level = level

    def rate_at(self, key):
        if type(key) is not int:
            raise TypeError('curve key must be an integer')
        return self.level + key

    def price(self, key, column, default):
        return self.rate_at(key) if column == 'rate' else default
'''
DECLARATIONS = '''\
cdef class CurveView:
    cdef object owner
    cdef double level
cdef class LookupContext:
    cdef bint _private_owner
    cdef object owner, column
cdef bint prices_ready(LookupContext, const void*, const void*) noexcept
cdef double bound_price(LookupContext, long long, long long, bint, bint, double) except? -1
cdef double prime_price(LookupContext, CurveView, long long, object, object) except? -1
cdef CurveView make_view(object value)
cdef double calculate_rate(CurveView view, long long key) except? -1
'''
IMPLEMENTATION = '''\
# cython: freethreading_compatible=True
from curve_example import Curve

__modelx_cython_provider__ = (1, 'curve_example.kernel', 'curve-v1')
preparations = 0
reads = 0
price_reads = 0

cdef class LookupContext:
    def _mx_enable_private_lookup(self):
        self._private_owner = True

cdef bint prices_ready(LookupContext context, const void* owner, const void* column) noexcept:
    return <const void*>context.owner == owner and <const void*>context.column == column

cdef double bound_price(LookupContext context, long long key, long long column, bint named,
                        bint has_default, double default) except? -1:
    global price_reads
    price_reads += 1
    return context.owner.price(key, context.column if named else column, default)

cdef double prime_price(LookupContext context, CurveView view, long long key,
                        object column, object default) except? -1:
    context.owner = view.owner
    context.column = column
    return view.price(key, column, default)

cdef class CurveView:
    def rate_at(self, key):
        return self.owner.rate_at(key)

    def __modelx_cython_owner__(self):
        return self.owner

    def price(self, key, column, default):
        return self.owner.price(key, column, default)

cdef CurveView make_view(object value):
    global preparations
    if type(value) is CurveView:
        return value
    convert = getattr(value, '__modelx_cython_owner__', None)
    if convert is not None:
        value = convert()
    if type(value) is not Curve:
        raise TypeError('curve_example.Curve owner required')
    cdef CurveView result = CurveView.__new__(CurveView)
    result.owner = value
    result.level = value.level
    preparations += 1
    return result

cdef double calculate_rate(CurveView view, long long key) except? -1:
    global reads
    reads += 1
    return view.level + key
'''


def test_independent_provider_and_forwarding(tmp_path, monkeypatch):
    package = tmp_path / 'curve_example'
    package.mkdir()
    (package / '__init__.py').write_text(OWNER, encoding='utf-8')
    (package / 'kernel.pxd').write_text(DECLARATIONS, encoding='utf-8')
    implementation = package / 'kernel.pyx'
    implementation.write_text(IMPLEMENTATION, encoding='utf-8')
    manifest = {
        'protocol': 1, 'provider': PROVIDER, 'abi': ABI,
        'types': {TYPE: {'view_type': 'CurveView', 'bind': 'make_view'}},
    }
    (package / 'kernel.provider.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    curves = import_module('curve_example')

    model = mx.new_model('IndependentCurve')
    try:
        data = model.new_space('Data')
        data.curves = curves
        data.level = 2.5
        data.source = curves.Curve(9.5)
        data.new_cells('primary', 'lambda: curves.Curve(level)')
        data.new_cells('forward', 'lambda: primary()')
        data.new_cells('again', 'lambda: forward()')
        data.new_cells('opaque', 'def opaque() -> object:\n    return primary()\n')
        data.new_cells('bound', 'lambda: source')
        data.new_cells('direct_rate', 'lambda: primary().rate_at(4)')
        data.new_cells('forwarded_rate', 'lambda: again().rate_at(4)')
        data.new_cells('opaque_rate', 'lambda: opaque().rate_at(4)')
        data.new_cells('variable_rate', 'lambda key: again().rate_at(key)')
        projection = model.new_space('Projection')
        projection.data = data
        projection.new_cells('rate_from_data', 'lambda: data.again().rate_at(4)')
        keys = (1, 2)
        pricing = model.new_space('Pricing', formula='lambda key: None', refs={'data': data, 'keys': keys},
                                  collect_over={'key': keys},
                                  collect_outputs='lambda: {"sum": {"total": total()}}')
        pricing.new_cells('schedule', 'lambda: data.again()')
        pricing.new_cells('quote', 'lambda month: schedule().price(month, "rate", 0.0)')
        pricing.new_cells('total', 'lambda: quote(key) + quote(key + 1)')
        assert (data.direct_rate(), data.forwarded_rate(), data.opaque_rate()) == (6.5, 6.5, 6.5)
        assert projection.rate_from_data() == 6.5
        assert 'curve_example.kernel' not in sys.modules
        for name in ('first', 'second'):
            model.export(tmp_path / name, locked_spaces=['Data'])
            (tmp_path / f'{name}_sample.py').write_text(
                f'from {name} import mx_model\n'
                'data = mx_model.Data\n'
                'assert data.direct_rate() == data.forwarded_rate() == data.opaque_rate() == 6.5\n'
                'assert data.variable_rate(4) == 6.5\n'
                'assert data.bound().rate_at(0) == 9.5\n'
                'assert mx_model.Projection.rate_from_data() == 6.5\n'
                'assert mx_model.Pricing.collect().sum.total == 18.0\n', encoding='utf-8')
    finally:
        model.close()

    (tmp_path / 'spec.py').write_text(
        repr({'spaces': {'Pricing': {'serial_collect': 'reuse'}}}), encoding='utf-8')
    for name in ('first', 'second'):
        subprocess.run([sys.executable, '-m', 'modelx_cython', name,
                        '--sample', f'{name}_sample.py', '--spec', 'spec.py'], cwd=tmp_path, check=True)
        implementation.write_text(IMPLEMENTATION + '\n# New build identity.\n', encoding='utf-8')

    alias = module_name(PROVIDER)
    probe = dedent(f'''\
        from importlib import import_module
        from importlib.machinery import ExtensionFileLoader
        from first_cy import mx_model as first
        from second_cy import mx_model as second

        f, s = first.Data, second.Data
        first_provider = import_module('first_cy.{alias}')
        second_provider = import_module('second_cy.{alias}')
        assert isinstance(first_provider.__loader__, ExtensionFileLoader)
        assert f.direct_rate() == f.forwarded_rate() == f.opaque_rate() == 6.5
        assert f.primary() is f.forward() is f.again()
        assert f.primary().__modelx_cython_owner__() is f.opaque().__modelx_cython_owner__()
        assert first_provider.preparations == 1
        assert first.Projection.rate_from_data() == 6.5
        assert first_provider.preparations == 1
        assert type(first)().Pricing[1].quote(1) == 3.5
        assert first.Pricing.collect().sum.total == 18.0
        assert type(f.primary()) is not type(s.primary())
        assert second_provider.preparations == 1
        s.source = f.primary()
        assert type(s.bound()) is type(s.primary())
        assert s.bound().__modelx_cython_owner__() is f.primary().__modelx_cython_owner__()
        assert s.bound().rate_at(4) == 6.5
        fresh = type(first)()
        fresh.Data.level = 4.5
        assert fresh.Data.forwarded_rate() == 8.5 and f.forwarded_rate() == 6.5
        fresh.Data.source = object()
        try:
            fresh.Data.bound()
        except TypeError as error:
            assert 'curve_example.Curve' in str(error)
        else:
            raise AssertionError('bound accepted an unregistered owner')
    ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)

    first_binary, = (tmp_path / 'first_cy').glob(f'{alias}.*{_extension_suffix()}')
    second_binary, = (tmp_path / 'second_cy').glob(f'{alias}.*{_extension_suffix()}')
    shutil.copyfile(second_binary, first_binary)
    result = subprocess.run([sys.executable, '-c', 'import first_cy'], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert f'Native provider {PROVIDER} requires ABI {ABI}' in result.stderr


def test_removed_private_lookup_spec_key(tmp_path):
    """audit-routes P6: the removed spec key is refused with its reason, not silently ignored."""
    model = mx.new_model('RemovedKey')
    try:
        keys = (1, 2)
        pricing = model.new_space('Pricing', formula='lambda key: None', refs={'keys': keys},
                                  collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"total": total()}}')
        pricing.new_cells('total', 'lambda: 1.5 * key')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\nmx_model.Pricing.collect()\n',
                                        encoding='utf-8')
    (tmp_path / 'spec.py').write_text(
        repr({'spaces': {'Pricing': {'serial_collect': 'reuse', 'serial_private_lookups': True}}}), encoding='utf-8')
    run = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                          '--spec', 'spec.py', '--translate-only'], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode != 0
    assert 'Pricing: the spec key serial_private_lookups was removed' in run.stdout + run.stderr


def _extension_suffix():
    return '.pyd' if sys.platform == 'win32' else '.so'
