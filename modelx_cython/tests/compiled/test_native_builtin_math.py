"""Guarded native rounding preserves exact scalar and callable semantics."""

import json
import math
import subprocess
import sys
import textwrap

import modelx as mx
from modelx_cython.helpers import module_name
from modelx_cython.rounding import NATIVE_ROUNDING_PROVIDER


def test_native_builtin_math(tmp_path):
    model = mx.new_model('NativeBuiltinMath')
    try:
        space = model.new_space('Values', refs={'math': math, 'numbers': math,
                                              'downward': math.floor})
        for name, formula in (
            ('one', 'lambda x: round(x)'),
            ('digits', 'lambda x, n: round(x, n)'),
            ('down', 'lambda x: math.floor(x)'),
            ('up', 'lambda x: math.ceil(x)'),
            ('module_down', 'lambda x: numbers.floor(x)'),
            ('module_up', 'lambda x: numbers.ceil(x)'),
            ('callable_down', 'lambda x: downward(x)'),
        ):
            space.new_cells(name, formula).is_cached = False
        space.new_cells('cached_digits', '''\
def cached_digits(x, n):
    amount = x * 0.5
    adjustment = 0.125
    total = amount + adjustment
    return round(total, min(n, 22))
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    # The uncached cells also accept arbitrary Python objects and return unbounded ints.
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'v.cached_digits(5.1, 2)\n',
        encoding='utf-8')
    (tmp_path / 'spec.py').write_text("{'native_math': True}\n", encoding='utf-8')
    # The CLI, its setup child, and the imported artifact all run here.
    (tmp_path / 'pandas.py').write_text(
        'raise AssertionError("a scalar-only model attempted to import pandas")\n',
        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported'],
                   cwd=tmp_path, check=True)
    package = tmp_path / 'exported_cy'
    report = json.loads((package / '_mx_admission.json').read_text(encoding='utf-8'))
    # the build report: the repeated rounding of cached_digits is the native rounding operation
    assert (f'{NATIVE_ROUNDING_PROVIDER}.repeated_round_float'
            in report['exported_cy._mx_classes._c_Values._f_cached_digits']['native_helpers'])
    assert not (package / f"{module_name('modelx_cython.runtime._native_inputs')}.pyx").exists()
    probe = textwrap.dedent('''\
        import math
        import random
        import struct
        import sys
        from exported_cy import mx_model
        v = mx_model.Values
        assert v.cached_digits(5.1, 2) == round(5.1 * 0.5 + 0.125, 2)
        assert 'pandas' not in sys.modules
        rng = random.Random(107)
        edge = [0., -0., 0.5, -0.5, 2.5, -2.5, 2.675, 1.005, 1.015,
                0.15, -0.15, 0.125, 125., -125., 2.**52, 2.**63, -2.**63,
                5e-324, -5e-324, float('inf'), float('-inf'), float('nan')]
        random_values = [struct.unpack('=d', rng.randbytes(8))[0] for _ in range(4000)]
        decimal_edges = [math.nextafter(x, direction) for n in range(-22, 23)
                         for k in (0.5, 1.5, 2.5, 128.5, 1e7+0.5)
                         for x in (k * 10.0**(-n), -k * 10.0**(-n))
                         for direction in (-math.inf, x, math.inf)]
        for x in edge + random_values + decimal_edges:
            for n in range(-24, 25):
                actual, expected = v.digits(x, n), round(x, n)
                assert (math.isnan(actual) and math.isnan(expected)
                        or struct.pack('=d', actual) == struct.pack('=d', expected)), (x, n, actual, expected)
            for actual, expected in ((v.one, round), (v.down, math.floor), (v.up, math.ceil),
                                     (v.module_down, math.floor), (v.module_up, math.ceil),
                                     (v.callable_down, math.floor)):
                try:
                    target = expected(x)
                except (ValueError, OverflowError) as error:
                    try:
                        actual(x)
                    except type(error):
                        pass
                    else:
                        raise AssertionError(x)
                else:
                    result = actual(x)
                    assert type(result) is int and result == target, (x, result, target)
        class Hook:
            def __round__(self, *args): return ('round', args)
            def __floor__(self): return 'floor'
            def __ceil__(self): return 'ceil'
        assert v.one(Hook()) == ('round', ())
        assert v.digits(Hook(), 2) == ('round', (2,))
        assert v.down(Hook()) == 'floor'
        assert v.up(Hook()) == 'ceil'
        assert v.module_down(Hook()) == v.callable_down(Hook()) == 'floor'
        assert v.module_up(Hook()) == 'ceil'
        from types import SimpleNamespace
        v.math = SimpleNamespace(floor=lambda x: x + 10, ceil=lambda x: x + 20)
        assert v.down(1.5) == 11.5 and v.up(1.5) == 21.5
        v.numbers = SimpleNamespace(floor=lambda x: x + 30, ceil=lambda x: x + 40)
        v.downward = lambda x: x + 50
        assert v.module_down(1.5) == 31.5 and v.module_up(1.5) == 41.5
        assert v.callable_down(1.5) == 51.5
    ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
