"""W1.4: ``math.<fn>`` on a module Reference bound at build time (D25 Q4).

Typed arguments call libm through ``_native_math.m_<fn>(_mx_cy.cast(_mx_cy.p_void, self.math), x)``: no per-call
attribute read. Results and errors equal the module's; floor/ceil return ints; a
replaced Reference keeps its own member; members with no native form are listed.
"""
import json
import math
import subprocess
import sys
import textwrap

import modelx as mx


def test_module_math_bound_at_build_time(tmp_path):
    model = mx.new_model('ModuleMath')
    try:
        s = model.new_space('Values', refs={'math': math, 'scale': 0.5,
            'edges': (-1.0, 0.0, -0.0, math.inf, -math.inf, math.nan, 1e308, -5e-324, 2.0 ** 63, 0.5)})
        s.new_cells('x_of', 'lambda i: (i - 500) * 0.37')
        for name in ('erf', 'exp', 'log', 'sqrt', 'isinf'):
            s.new_cells(name, f'lambda i: math.{name}(x_of(i))')
        s.new_cells('down', 'lambda i: math.floor(x_of(i))')
        s.new_cells('up', 'lambda i: math.ceil(x_of(i) * scale)')
        s.new_cells('down_int', 'lambda i: math.floor(i)')
        s.new_cells('rounded', 'lambda i: math.floor(x_of(i) / 0.25 + 0.5) * 0.25')
        s.new_cells('years', 'lambda i: min(10, int(math.ceil(x_of(i))))')
        s.new_cells('unbounded', 'lambda i: math.inf if i > 3 else x_of(i)')
        s.new_cells('loglin', 'lambda i: math.exp(math.log(1.5) + 0.25 * (math.log(2.5) - math.log(1.5)))')
        s.new_cells('raw', 'lambda x: math.floor(x)').is_cached = False
        s.new_cells('loose', 'lambda x: math.floor(x)').is_cached = False
        s.new_cells('exp_raw', 'lambda x: math.exp(x)').is_cached = False
        s.new_cells('inf_raw', 'lambda: math.inf').is_cached = False
        s.new_cells('edge', 'lambda k: edges[k]')
        for name in ('erf', 'exp', 'log', 'sqrt', 'isinf', 'floor', 'ceil'):
            s.new_cells('e_' + name, f'lambda k: math.{name}(edge(k))')
        s.new_cells('close', 'lambda i: math.isclose(x_of(i), 1.0)')
        s.new_cells('base_two', 'lambda i: math.log(x_of(i) + 200.0, 2.0)')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'v = m.Values\n'
        'for i in range(1000):\n'
        '    v.x_of(i); v.exp(i); v.erf(i); v.isinf(i); v.down(i); v.up(i); v.down_int(i)\n'
        '    v.rounded(i); v.years(i); v.unbounded(i); v.loglin(i); v.close(i); v.base_two(i)\n'
        '    if i > 500: v.log(i); v.sqrt(i)\n'
        'v.raw(1.5); v.exp_raw(0.5); v.inf_raw()\n'
        'for k in range(10):\n'
        '    v.edge(k)\n'
        '    for name in ("erf", "exp", "log", "sqrt", "isinf", "floor", "ceil"):\n'
        '        try: getattr(v, "e_" + name)(k)\n'
        '        except (ValueError, OverflowError): pass\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())

    def entry(name):
        return report[f'exported_cy._mx_classes._c_Values._f_{name}']

    # the build report: the math calls are native provider operations
    def helpers(name):
        return {h.rsplit('.', 1)[-1] for h in entry(name)['native_helpers']}

    for name in ('erf', 'exp', 'log', 'sqrt', 'isinf'):
        assert f'm_{name}' in helpers(name), (name, helpers(name))
    assert 'm_floor_ll' in helpers('down') and 'm_ceil_ll' in helpers('up'), (helpers('down'), helpers('up'))
    used = set().union(*(helpers(name.rsplit('._f_', 1)[-1]) for name in report))
    assert {'m_floor_ll', 'm_constant', 'm_floor'} <= used, used   # m_floor: the unsampled (object) argument
    assert entry('down')['return_type'] == 'long long' and entry('loglin')['native_expressions'] == 4
    # W1.8: math.isclose compiles (m_isclose); test_native_builtins_w18 compares it
    assert entry('close')['expressions_untyped'] == [] and 'm_isclose' in helpers('close')
    assert '2 arguments' in entry('base_two')['expressions_untyped'][0]['reason']
    probe = textwrap.dedent('''\
        import math, random, struct, sys
        from importlib import import_module
        from types import SimpleNamespace
        v = import_module(sys.argv[1]).mx_model.Values

        def same(left, right):
            if isinstance(right, float):
                return type(left) is float and (math.isnan(left) and math.isnan(right)
                       or struct.pack('<d', left) == struct.pack('<d', right))
            return type(left) is type(right) and left == right

        def outcome(function, *args):
            try:
                return ('ok', function(*args))
            except Exception as error:
                return ('raise', type(error))

        for i in range(1000):
            x = (i - 500) * 0.37
            for name in ('erf', 'exp', 'isinf'):
                assert same(getattr(v, name)(i), getattr(math, name)(x)), (name, i)
            if i > 500:
                assert same(v.log(i), math.log(x)) and same(v.sqrt(i), math.sqrt(x)), i
            assert same(v.down(i), math.floor(x)) and same(v.up(i), math.ceil(x * 0.5)), i
            assert same(v.down_int(i), i)
            assert same(v.rounded(i), math.floor(x / 0.25 + 0.5) * 0.25), i
            assert same(v.years(i), min(10, int(math.ceil(x)))), i
            assert same(v.unbounded(i), math.inf if i > 3 else x)
        assert same(v.loglin(0), math.exp(math.log(1.5) + 0.25 * (math.log(2.5) - math.log(1.5))))
        compiled = sys.argv[1] == 'exported_cy'
        rng = random.Random(4)
        values = [struct.unpack('<d', rng.randbytes(8))[0] for _ in range(20000)]
        values += [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0 ** 52 + 0.5, 2.0 ** 63, -2.0 ** 63, 1e308,
                   -1e308, 5e-324, math.inf, -math.inf, math.nan, 709.78, 709.79, -745.2]
        for x in values:
            expected = outcome(math.floor, x)
            for typed, cell in ((True, v.raw), (False, v.loose)):
                actual = outcome(cell, x)
                if compiled and typed and math.isfinite(x) and not -2.0 ** 63 <= x < 2.0 ** 63:
                    expected_here = ('raise', OverflowError)   # C integer result domain (listed)
                else:
                    expected_here = expected
                assert actual[0] == expected_here[0] and (
                    actual[1] is expected_here[1] if actual[0] == 'raise' else same(actual[1], expected_here[1])), (x, actual)
        edges = (-1.0, 0.0, -0.0, math.inf, -math.inf, math.nan, 1e308, -5e-324, 2.0 ** 63, 0.5)
        for k, x in enumerate(edges):
            for name in ('erf', 'exp', 'log', 'sqrt', 'isinf', 'floor', 'ceil'):
                left, right = outcome(getattr(v, 'e_' + name), k), outcome(getattr(math, name), x)
                if compiled and name in ('floor', 'ceil') and math.isfinite(x) and abs(x) >= 2.0 ** 63:
                    right = ('raise', OverflowError)
                assert left[0] == right[0] and (left[1] is right[1] if left[0] == 'raise'
                                                else same(left[1], right[1])), (name, x, left, right)
        v.math = SimpleNamespace(floor=lambda x: 7, ceil=lambda x: 8, exp=lambda x: 1.25,
                                 log=lambda x: 2.5, inf=-1.0, isinf=math.isinf, erf=math.erf,
                                 sqrt=math.sqrt, isclose=math.isclose)
        assert v.raw(1.5) == 7 and v.loose(1.5) == 7 and v.exp_raw(3.0) == 1.25 and v.inf_raw() == -1.0
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
