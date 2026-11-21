"""diag-mt2: a module math member call passes the math Reference borrowed (no reference per step).

``self.math.log(x)`` compiles to ``_native_math.m_log(<the Reference>, x)``. The Reference was passed
as an ``object`` argument, so Cython took a reference on the module around every call (an atomic
increment and decrement on the one ``math`` module that every worker shares; Pflege_DE_S at four
workers was slower than at one). The provider now takes the Reference as a borrowed ``void*``
(``_mx_cy.cast(_mx_cy.p_void, self.math)``): identity checks compare the pointer, the Python fallback
of another namespace is unchanged. Results must not change: compiled at one and at four workers
equals the exported model.
"""
import math
import json
import subprocess
import sys
import textwrap

import modelx as mx


def _model(tmp_path):
    keys = tuple(range(1, 61))
    model = mx.new_model('MathBorrowed')
    try:
        portfolio = model.new_space('Portfolio')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'math': math, 'keys': keys},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('rate', '''\
def rate(t):
    return 0.001 * (t + key % 7 + 1)
''')
        projection.new_cells('surv', '''\
def surv(t):
    if t == 0:
        return 1.0
    mu = -math.log(1.0 - rate(t - 1))
    return surv(t - 1) * math.exp(-mu / 12.0) * math.sqrt(1.0 - rate(t) / 4.0)
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(48):
        total += surv(t) * math.floor(t / 12.0 + 1.0) * math.pi
    return total
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    spec = {'parallel_workers': 4, 'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'parallel_collect_reuse': True, 'parallel_collect_order': 'domain'}}}}}
    (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'p.collect()\n'
        'for key in (1, 2, 3):\n'
        '    [p[key].surv(t) for t in range(48)]; p[key].value()\n',
        encoding='utf-8')


def test_math_reference_is_borrowed_and_results_stay(tmp_path):
    _model(tmp_path)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    # the build report: the math calls are native provider operations
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    helpers = {h.rsplit('.', 1)[-1] for entry in admission.values() for h in entry['native_helpers']}
    assert {'m_log', 'm_exp', 'm_sqrt', 'm_floor', 'm_constant'} <= helpers, helpers
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        m = import_module(sys.argv[1]).mx_model
        p = m.Portfolio.Projection
        if sys.argv[1] == 'exported_cy':
            serial = type(m)()
            serial.collect_workers = 1
            one = serial.Portfolio.Projection.collect().sum.value
            four = p.collect().sum.value
            again = p.collect().sum.value
            assert again == four
            print(repr(one), repr(four))
        else:
            print(repr(p.collect().sum.value), repr(p.collect().sum.value))
        item = p[7]
        print(repr([item.surv(t) for t in range(48)] + [item.value()]))
        ''')
    out = {}
    for name in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, name], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, name + run.stdout + run.stderr
        out[name] = run.stdout.splitlines()
    one, four = out['exported_cy'][0].split()
    assert one == out['exported'][0].split()[0], (out, 'one worker equals the exported model')
    assert abs(float(four) - float(one)) <= 1e-12 * abs(float(one)), out
    assert out['exported_cy'][1:] == out['exported'][1:], out
