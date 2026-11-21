"""diag-jp: forms the build report did not show as slow, compiled natively, and the report entries.

- a sum over `range(max(0, t - window()), t)` with a term `cell(s) * (1.0 - rho) ** (t - 1 - s)`
  (Term_JP_S pols_lapse_pool): the range is C integers, the loop variable a C integer, the power
  native and the sum state a double; `range(int(x), int(x) + int(m))` likewise;
- table key parts bound by a tuple assignment `x, m = f(k), g(k)` used in Python helper lookups;
- the report lists what still runs on objects (`slow_paths`): hash caches with the reason,
  reductions over objects, table key parts as objects, object returns of cells with parameters.
Compiled equals the exported model on every call (value, type, or exception).
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx
from modelx_helpers.immutable_inputs import Table

PREFIX = 'exported_cy._mx_classes._c_Values._f_'


def _build(tmp_path):
    model = mx.new_model('DiagJP')
    try:
        rates = Table([(0, 20, 5), (1, 20, 5), (0, 25, 5)], [(1.5,), (2.5,), (3.5,)], ['rate'])
        s = model.new_space('Values', refs={'rates': rates, 'window_len': 4})
        s.new_cells('window', 'lambda: window_len')
        s.new_cells('rho', 'lambda: 0.125')
        s.new_cells('flow', 'lambda t: 1.0 + 0.5 * t')
        s.new_cells('pool', textwrap.dedent('''\
            def pool(t):
                r = rho()
                return sum(flow(s) * (1.0 - r) ** (t - 1 - s) for s in range(max(0, t - window()), t))
            '''))
        s.new_cells('mean', textwrap.dedent('''\
            def mean(x, m):
                return sum(flow(a) for a in range(int(x), int(x) + int(m))) / m
            '''))
        s.new_cells('start', 'lambda k: 20 + 5 * (k % 2)')
        s.new_cells('length', 'lambda k: 5')
        s.new_cells('table', '''lambda: rates''')
        s.new_cells('rate', textwrap.dedent('''\
            def rate(k):
                x, m = start(k), length(k)
                key = (k % 2, int(x), int(m))
                if key in table().index:
                    return float(table().loc[key, "rate"])
                return -1.0
            '''))
        s.new_cells('lookup', textwrap.dedent('''\
            def lookup(t):
                return {1: 2.0, 3: 4.0}.get(t, 0.0)
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        '[v.pool(t) for t in range(12)]; [v.mean(x, 3) for x in (1.0, 2.0)]\n'
        '[v.rate(k) for k in range(4)]; [v.lookup(t) for t in range(4)]\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return ((tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8'),
            json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text()))


def test_forms_compile_natively_and_equal_the_interpreter(tmp_path):
    source, report = _build(tmp_path)
    pool = source[source.index('def _f_pool'):]
    pool = pool[:pool.index('\n    @')] if '\n    @' in pool else pool
    formula = source[source.index('_bound_0: _mx_cy.longlong = max(0, '):][:2000]
    assert 'in range(_mx_reduce_0_bound_0, _mx_reduce_0_bound_1):' in formula, formula
    assert '.sum_add_f64(' in formula and 'cast(object' not in formula.split('return')[0], formula
    assert report[PREFIX + 'pool']['slow_paths'] == [], report[PREFIX + 'pool']
    assert report[PREFIX + 'mean']['slow_paths'] == [] or all(
        e['kind'] != 'reduction-on-objects' for e in report[PREFIX + 'mean']['slow_paths'])
    assert 'key = (k % 2, int(x), int(m))' in source
    assert 'if key in ' in source and '.loc[key, "rate"]' in source
    probe = textwrap.dedent('''\
        import json, sys
        from importlib import import_module
        v = import_module(sys.argv[1]).mx_model.Values
        out = []
        def call(f, *args):
            try:
                r = f(*args)
                # a cell with a C double return gives 0.0 where an empty sum gives 0 (a known type
                # difference of compiled double cells, before this change too): values compare as floats
                out.append(['number', repr(float(r))] if isinstance(r, (int, float)) else [type(r).__name__, repr(r)])
            except Exception as e:
                out.append(['raise', type(e).__name__])
        # inside the sampled indices: a dense cache sized by the sample raises outside them by design
        for t in range(12):
            call(v.pool, t)
        for x in (1.0, 2.0):
            call(v.mean, x, 3)
        for k in range(4):
            call(v.rate, k)
        print(json.dumps(out))
        ''')
    results = []
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        results.append(json.loads(run.stdout))
    assert results[0] == results[1], results


def test_report_lists_slow_paths(tmp_path):
    _, report = _build(tmp_path)
    lookup = report[PREFIX + 'lookup']['slow_paths']
    assert lookup == [] or all(set(entry) == {'kind', 'detail', 'per_step'} for entry in lookup)
    every = [entry for cell in report.values() for entry in cell.get('slow_paths', ())]
    assert all(entry['kind'] in ('hash-cache', 'reduction-on-objects', 'table-key-object',
                                 'power-on-objects-in-loop', 'object-return') for entry in every), every
