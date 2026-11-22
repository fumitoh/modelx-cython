"""Reads through a helper Record keep their values and error classes at one and four workers."""
import json
import subprocess
import sys
import textwrap

import modelx as mx
from modelx_helpers.immutable_inputs import Table

KEYS = tuple(range(1, 41))


def _points():
    rows = []
    for key in KEYS:
        maybe = float('nan') if key % 7 == 0 else (None if key % 11 == 0 else 2.5 + key)
        rows.append((key % 5 + 1, 0.01 * key, 40.0 + key / 3.0, 'p' * (key % 4 + 1), maybe))
    return Table(list(KEYS), rows, ['n', 'rate', 'age_f', 'label', 'maybe'])


def _build(tmp_path):
    model = mx.new_model('HeldRecords')
    try:
        portfolio = model.new_space('Portfolio')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'points': _points(), 'keys': KEYS},
            collect_over={'key': KEYS}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('model_point', 'lambda: points.loc[key]')
        projection.new_cells('term', textwrap.dedent('''\
            def term(t):
                mp = model_point()
                n = int(mp["n"])
                return 1.0 if t < 12 * n else 0.0
            '''))
        projection.new_cells('rate', 'lambda t: float(model_point()["rate"]) * (t + 1)')
        projection.new_cells('age', 'lambda t: int(model_point()["age_f"]) + t // 12')
        projection.new_cells('label_len', 'lambda t: len(model_point()["label"]) + t')
        projection.new_cells('bad', 'lambda t: int(model_point()["maybe"]) + t')
        projection.new_cells('badf', 'lambda t: float(model_point()["maybe"]) * t')
        projection.new_cells('missing', 'lambda t: model_point()["nope"]')
        projection.new_cells('branchy', textwrap.dedent('''\
            def branchy(t):
                if t % 2 == 0:
                    mp = model_point()
                    return float(mp["rate"]) + t
                return 0.0
            '''))
        projection.new_cells('value', textwrap.dedent('''\
            def value():
                total = 0.0
                for t in range(48):
                    total += rate(t) * term(t) + age(t) + label_len(t) + branchy(t)
                return total
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    spec = {'parallel_workers': 4, 'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'parallel_collect_reuse': True, 'parallel_collect_order': 'domain'}}}}}
    (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(textwrap.dedent('''\
        from exported import mx_model
        p = mx_model.Portfolio.Projection
        p.collect()
        for key in (1, 2, 3, 14, 22):
            item = p[key]
            [item.term(t) for t in range(48)]; item.value()
            for t in range(4):
                for f in (item.bad, item.badf, item.missing):
                    try:
                        f(t)
                    except (KeyError, ValueError, TypeError):
                        pass
        '''), encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))


def test_held_record_reads_equal_live_at_one_and_four_workers(tmp_path):
    report = _build(tmp_path)
    point = next(value for key, value in report.items() if key.endswith('_f_model_point'))
    helper_reads = point['table_reads_untyped']
    assert helper_reads and all('helper Table' in read['reason'] for read in helper_reads)
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
            print(repr(one), repr(four))
        else:
            print(repr(p.collect().sum.value), repr(p.collect().sum.value))
        out = []
        for key in KEYS:
            item = p[key]
            for t in (0, 1, 13, 47):
                for f in (item.term, item.rate, item.age, item.label_len, item.bad, item.badf,
                          item.missing, item.branchy):
                    try:
                        r = f(t)
                        out.append(repr(float(r)))
                    except Exception as e:
                        out.append(type(e).__name__)
        print(repr(out))
        ''').replace('KEYS', repr(KEYS))
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    one, four = out['exported_cy'][0].split()
    assert one == out['exported'][0].split()[0], out
    assert abs(float(four) - float(one)) <= 1e-12 * abs(float(one)), out
    assert out['exported_cy'][1] == out['exported'][1], [
        (a, b) for a, b in zip(eval(out['exported'][1]), eval(out['exported_cy'][1])) if a != b][:10]
    errors = eval(out['exported'][1])
    assert 'ValueError' in errors and 'TypeError' in errors and 'KeyError' in errors
