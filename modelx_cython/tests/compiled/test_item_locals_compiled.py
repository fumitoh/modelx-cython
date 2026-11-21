"""D110: a compiled Cell that loops over the items of an ItemSpace Reference computes the live values,
whether its item local is typed (bound once) or not (rebound), and integer arguments keep the
checks and errors of the Python gate (P1); Python callers keep the gate's checks and errors (P0)."""

import json
import subprocess
import sys
import textwrap

import modelx as mx

FORMULAS = {
    'direct': '''\
        def direct():
            total = 0.0
            for key in projection.policy_keys():
                item = projection[key]
                for t in range(item.n()):
                    total += item.f(t)
            return total
        ''',
    'direct_call': '''\
        def direct_call():
            total = 0.0
            for key in projection.policy_keys():
                item = projection(key)
                for t in range(item.n()):
                    total += item.f(t) * item.g(t, 2)
            return total
        ''',
    'rebound': '''\
        def rebound():
            total = 0.0
            item = None
            for key in projection.policy_keys():
                item = projection[key]
                total += item.f(item.n() - 1)
            return total
        ''',
    'shifted': '''\
        def shifted(shift):
            item = projection[2]
            return item.f(shift + 1)
        ''',
}


def build(tmp_path, mode='checked'):
    # the checks below are the checked mode's (owner's rule): the small sample resolves unchecked
    model = mx.new_model('ItemLocals')
    try:
        projection = model.new_space('Projection', formula=lambda point_id: None)
        projection.new_cells('policy_keys', 'lambda: [1, 2, 3]')
        projection.new_cells('n', 'lambda: 2 + point_id')
        projection.new_cells('f', textwrap.dedent('''\
            def f(t):
                return 1.5 * point_id + (f(t - 1) if t > 0 else 0.25)
            '''))
        projection.new_cells('g', 'lambda t, k: float(t * k + point_id)')
        portfolio = model.new_space('Portfolio', refs={'projection': projection})
        for name, formula in FORMULAS.items():
            # uncached: an argument of any 64-bit value reaches the formula
            portfolio.new_cells(name, textwrap.dedent(formula), is_cached=name != 'shifted')
        live = {name: getattr(portfolio, name)() for name in FORMULAS if name != 'shifted'}
        live['shifted'] = portfolio.shifted(1)
        live['item_f_3'] = projection[2].f(3)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio\n'
        'p.direct(); p.direct_call(); p.rebound(); p.shifted(1)\n'
        'mx_model.Projection[2].f(3)\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'integer_checks': mode}), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                    '--spec', 'spec.py'], cwd=tmp_path, check=True)
    return live


PROBE = '''\
import json
from exported_cy import mx_model
p, P = mx_model.Portfolio, mx_model.Projection
out = {name: getattr(p, name)() for name in ('direct', 'direct_call', 'rebound')}
out['shifted'] = p.shifted(1)
item = P[2]
out['item_f_3'] = item.f(3)
errors = {}
for label, call in (('big', lambda: item.f(2 ** 70)), ('float', lambda: item.f(1.5)), ('text', lambda: item.f('a')),
                    ('keyword', lambda: item.f(t=2 ** 64)), ('arity', lambda: item.f(1, 2)),
                    ('cell_big', lambda: p.shifted((1 << 63) - 1))):
    try:
        call()
        errors[label] = None
    except Exception as error:
        errors[label] = [type(error).__name__, str(error)]
gate = item.f
out['errors'] = errors
out['doc'] = [gate.__doc__ == type(item).f.__doc__, type(gate).__name__]
# items are cached and stay the same object; deleting one frees it without the cyclic GC
import gc
out['same_item'] = P[3] is P[3] and P(3) is P[3]
gc.disable()
def live():
    return sum(1 for o in gc.get_objects() if type(o) is type(item))
before = live()
del P[3]
out['freed_on_delete'] = before - live()
try:
    del P[3]
    out['missing_key'] = None
except KeyError as error:
    out['missing_key'] = ['KeyError', str(error)]
out['recreated'] = P[3].f(2) == p.direct() * 0 + P[3].f(2)
gc.enable()
out['gate'] = [gate.__name__, gate.__wrapped__.__name__, callable(gate)]
print(json.dumps(out))
'''


def test_item_locals_values_and_checks(tmp_path):
    live = build(tmp_path)
    (tmp_path / 'probe.py').write_text(PROBE, encoding='utf-8')
    result = json.loads(subprocess.run([sys.executable, 'probe.py'], cwd=tmp_path, check=True,
                                       capture_output=True, text=True).stdout)
    for name, value in live.items():
        assert abs(result[name] - value) <= 1e-7 + 1e-10 * abs(value), name
    message = 'Projection.f requires an exact signed 64-bit integer. Rebuild with a wider sample or spec.'
    errors = result['errors']
    assert errors['big'] == ['OverflowError', message]
    assert errors['float'] == ['TypeError', message]
    assert errors['text'] == ['TypeError', message]
    assert errors['keyword'] == ['OverflowError', message]
    assert errors['arity'][0] == 'TypeError'
    # an argument computed in the compiled Cell that leaves the 64-bit range raises, never wraps
    assert errors['cell_big'] is not None and errors['cell_big'][0] == 'OverflowError'
    assert result['gate'] == ['f', 'f', True]
    assert result['doc'][0]
    assert result['same_item'] and result['freed_on_delete'] == 1
    assert result['missing_key'] == ['KeyError', '3']
    assert result['recreated']


def test_item_locals_unchecked_mode(tmp_path):
    """Unchecked mode: the values and the checked argument conversions are as in the checked mode;
    an argument computed in the Cell beyond 64 bits is plain C arithmetic, and the wrapped value
    then fails the argument's range guard instead of raising OverflowError."""
    live = build(tmp_path, 'unchecked')
    (tmp_path / 'probe.py').write_text(PROBE, encoding='utf-8')
    result = json.loads(subprocess.run([sys.executable, 'probe.py'], cwd=tmp_path, check=True,
                                       capture_output=True, text=True).stdout)
    for name, value in live.items():
        assert abs(result[name] - value) <= 1e-7 + 1e-10 * abs(value), name
    message = 'Projection.f requires an exact signed 64-bit integer. Rebuild with a wider sample or spec.'
    assert result['errors']['big'] == ['OverflowError', message]
    assert result['errors']['cell_big'] is not None and result['errors']['cell_big'][0] == 'IndexError'
