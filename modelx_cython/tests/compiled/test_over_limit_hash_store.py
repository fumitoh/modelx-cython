"""D135: a Cell above the dense limit is held in the same hash store on both routes.

`surv(s, t)` has a 2-D integer window. With ``max_dense_entries`` below its size it is not inline dense,
and since D135 no route gives it another storage (the D74 window of collect's reused root is removed):
collect's per-policy reset empties the hash store and a direct item keeps its own. Values differ by
policy, so a stale entry would change the total. The build warns, naming the cell, its traced range and
the grid-key rule (R9); the report states the cache kind and why it is not dense. Compiled at one and four
workers equals the exported model; an item evaluated outside collect equals it too. The removed spec key
``reuse_dense_windows`` is refused with the reason.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx

PREFIX = 'exported_cy._m_Portfolio._mx_classes._c_Projection._f_'


def _model(tmp_path, spec):
    keys = tuple(range(1, 31))
    model = mx.new_model('ReuseDense')
    try:
        portfolio = model.new_space('Portfolio')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'keys': keys},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('rate', 'lambda t: 0.01 * ((t + key) % 7)')
        projection.new_cells('surv', '''\
def surv(s, t):
    if t <= s:
        return 1.0
    return surv(s, t - 1) * (1.0 - rate(t - 1))
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(12):
        for s in range(0, t + 1, 1 + key % 3):
            total += surv(s, t) * (t + 1)
    return total
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'p.collect()\n'
        'for key in (1, 2, 3):\n'
        '    p[key].value(); [p[key].surv(s, t) for s in range(12) for t in range(12)]\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    if not done.returncode == 0:
        return done
    return (done.stdout + done.stderr,
            json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8')))


PROBE = textwrap.dedent('''\
    import sys
    from importlib import import_module
    m = import_module(sys.argv[1]).mx_model
    p = m.Portfolio.Projection
    if sys.argv[1] == 'exported_cy':
        serial = type(m)()
        serial.collect_workers = 1
        one = serial.Portfolio.Projection.collect().sum.value
        again = serial.Portfolio.Projection.collect().sum.value
        four = p.collect().sum.value
        print(repr(one), repr(again), repr(four))
    else:
        v = p.collect().sum.value
        print(repr(v), repr(v), repr(v))
    item = p[7]
    print(repr([item.surv(s, t) for s in range(12) for t in range(12)] + [item.value()]))
    ''')


def _run(tmp_path):
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', PROBE, package], cwd=tmp_path, capture_output=True,
                             text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    return out


def test_over_limit_cell_is_one_hash_store_on_both_routes(tmp_path):
    spec = {'parallel_workers': 4, 'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'parallel_collect_reuse': True, 'parallel_collect_order': 'domain',
        'cells': {'surv': {'max_dense_entries': 16}}}}}}}
    built = _model(tmp_path, spec)
    assert isinstance(built, tuple), built.stdout[-3000:] + built.stderr[-3000:]
    log, report = built
    detail = report[PREFIX + 'surv']['cache_detail']
    assert detail['kind'] == 'native-numeric-keys', detail
    assert 'exceeds max_dense_entries 16' in detail['hash_reason'], detail
    assert 'reused_root' not in detail, detail                       # no collect-only storage (D135)
    assert any(e['kind'] == 'hash-cache' for e in report[PREFIX + 'surv']['slow_paths'])
    warning = [line for line in log.splitlines() if 'Projection.surv: window' in line]
    assert warning and 'exceeds max_dense_entries 16' in warning[0] and 'every route' in warning[0], log[-3000:]
    assert 's in [' in warning[0] and 't in [' in warning[0] and 'R9' in warning[0], warning
    out = _run(tmp_path)
    one, again, four = out['exported_cy'][0].split()
    assert one == again == out['exported'][0].split()[0], out        # reset leaves nothing stale
    # four chunk totals merged in domain order can differ from the serial sum in the last bits (config
    # docs: parallel_collect_order); every policy value is exact (one worker and the item read)
    assert abs(float(four) - float(one)) <= 1e-12 * abs(float(one)), out
    assert out['exported_cy'][1] == out['exported'][1], out


def test_removed_spec_key_reuse_dense_windows_is_refused(tmp_path):
    spec = {'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'reuse_dense_windows': False, 'max_dense_entries': 16}}}}}
    done = _model(tmp_path, spec)
    assert not isinstance(done, tuple), 'the removed spec key was accepted'
    text = done.stdout + done.stderr
    assert 'Projection: the spec key reuse_dense_windows was removed (D135)' in text and 'R9' in text, text[-3000:]
