"""diag-scale P1: a formula-local ``import math`` binds the native math provider.

A formula that writes ``import math`` (or ``import math as m``) as a top-level statement and never
rebinds the name gets its ``math.<fn>(x)`` calls and ``math.<constant>`` reads lowered as a math
Reference's are (W1.4, diag-mt2): the provider receives the ``math`` module itself, so it takes its
scalar path and keeps the Python fallback outside the C domain (errors as ``math`` raises them). The
import is dropped only when no read of the name is left. Declined, and run as written: a member with
no native form (the import stays), an import inside a branch, a name bound twice. Compiled equals
the exported model in values and in errors.
"""
import json
import os
import subprocess
import sys
import textwrap

import modelx as mx


CELLS = {
    'lowered': '''\
def lowered(t):
    import math
    x = 1.0 + 0.01 * t
    return math.exp(math.log(x) * 0.5) + math.sqrt(t + 1.0) + math.floor(t / 3.0) + math.pi
''',
    'aliased': '''\
def aliased(t):
    import math as m
    return m.exp(-0.1 * t) * m.e
''',
    'in_generator': '''\
def in_generator(t):
    import math
    return sum(math.sqrt(i + t) for i in range(4))
''',
    'not_native': '''\
def not_native(t):
    import math
    return math.exp(0.01 * t) + math.fsum([t, 0.5])
''',
    'conditional': '''\
def conditional(t):
    if t >= 0:
        import math
        return math.exp(0.02 * t)
    return 0.0
''',
    'rebound': '''\
def rebound(t):
    import math
    x = math.exp(0.03 * t)
    if t < 0:
        math = None
    return x
''',
    'domain': '''\
def domain(t):
    import math
    return math.log(t - 3.0)
''',
}


def _model(tmp_path):
    model = mx.new_model('LocalMath')
    try:
        s = model.new_space('Projection')
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'p = m.Projection\n'
        'for t in range(4, 9):\n'
        + ''.join(f'    p.{name}(t)\n' for name in CELLS) , encoding='utf-8')


PROBE = textwrap.dedent('''\
    import sys
    from importlib import import_module
    p = import_module(sys.argv[1]).mx_model.Projection
    names = sys.argv[2].split(',')
    rows = []
    for t in range(0, 9):
        row = []
        for name in names:
            try:
                row.append(repr(getattr(p, name)(t)))
            except Exception as error:
                row.append(f'{type(error).__name__}: {error}')
        rows.append(row)
    print(repr(rows))
    ''')


def _admission(tmp_path):
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    return {key.rsplit('._f_', 1)[-1]: value for key, value in report.items()
            if isinstance(value, dict) and '._f_' in key}


def test_formula_local_math_import_binds_the_provider(tmp_path):
    _model(tmp_path)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', PROBE, package, ','.join(CELLS)], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout
    # values, and the errors outside the domain (math.log of a non-positive number), are those of math
    assert out['exported_cy'] == out['exported'], out
    assert 'ValueError' in out['exported']

    cells = _admission(tmp_path)
    native = {name: [h.rsplit('.', 1)[-1] for h in cells[name]['native_helpers'] if '_native_math' in h]
              for name in CELLS}
    assert {'m_exp', 'm_log', 'm_sqrt', 'm_floor_ll', 'm_constant'} <= set(native['lowered']), native
    assert {'m_exp', 'm_constant'} <= set(native['aliased']), native
    assert 'm_sqrt' in native['in_generator'], native
    assert 'm_log' in native['domain'], native
    # a member with no native form keeps the Python call (and its import); the others are lowered
    assert native['not_native'] == ['m_exp'], native
    assert any('fsum' in json.dumps(entry) for entry in cells['not_native']['expressions_untyped']), cells['not_native']
    # not lowered: an import inside a branch, a name bound twice
    assert native['conditional'] == [] and native['rebound'] == [], native
