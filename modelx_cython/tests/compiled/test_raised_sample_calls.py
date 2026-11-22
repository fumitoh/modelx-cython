"""D111: a sampled call that raised contributes its argument ranges but no return type.

A Cell that raises for some sampled items (its own guard) keeps the type of the calls that
returned; a Cell whose every sampled call raised keeps the previous behaviour (its result is
traced as None, so it returns a Python object). Compiled values and raised exceptions equal live.
"""
import json
import os
import subprocess
import sys
import textwrap

import modelx as mx

KEYS = (1, 2, 3, 4)


def _build(tmp_path):
    model = mx.new_model('RaisedSample')
    try:
        p = model.new_space('Projection', formula='lambda policy_id: None',
                            refs={'keys': KEYS},
                            collect_over={'policy_id': KEYS},
                            collect_outputs='lambda: {"sum": {"total": total()}}')
        p.new_cells('life', 'lambda: policy_id % 2 == 0')
        p.new_cells('rate', 'lambda: 0.01 * policy_id')
        p.new_cells('proj_len', 'lambda: 5 + policy_id')
        p.new_cells('factor', textwrap.dedent('''\
            def factor():
                if not life():
                    raise ValueError("factor is defined on the life shape only")
                v = 1.0 / (1.0 + rate())
                return sum(v ** (t + 1) for t in range(proj_len()))
            '''))
        p.new_cells('never', textwrap.dedent('''\
            def never():
                raise LookupError("never defined")
            '''))
        p.new_cells('pay', 'lambda t: 100.0 / factor() if life() else 10.0 * t')
        p.new_cells('total', 'lambda: sum(pay(t) for t in range(proj_len()))')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    # The shipped-sample pattern (D102): project, then call every zero-argument Cell, recording raises.
    (tmp_path / 'sample.py').write_text(textwrap.dedent('''\
        from exported import mx_model as m
        for key in (1, 2, 3, 4):
            item = m.Projection[key]
            item.total()
            for name in ('factor', 'never'):
                try:
                    getattr(item, name)()
                except (ValueError, LookupError):
                    pass
        '''))
    (tmp_path / 'spec.py').write_text(repr({'parallel_workers': 1,
                                            'spaces': {'Projection': {'serial_collect': 'reuse'}}}))
    build = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported',
                            '--sample', 'sample.py', '--spec', 'spec.py'],
                           cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True)
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    return json.loads((tmp_path / 'exported_cy' / '_mx_admission.json').read_text())


PROBE = textwrap.dedent('''\
    import importlib, json, sys
    m = importlib.import_module(sys.argv[1]).mx_model
    out = {}
    for key in (1, 2, 3, 4):
        item = m.Projection[key]
        row = {'total': item.total(), 'pay': [item.pay(t) for t in range(3)]}
        for name in ('factor', 'never'):
            try:
                value = getattr(item, name)()
                row[name] = ['value', type(value).__name__, value]
            except Exception as error:
                row[name] = ['raised', type(error).__name__, str(error)]
        out[key] = row
    print(json.dumps(out, default=repr))
    ''')


def _run(tmp_path, package):
    (tmp_path / 'probe.py').write_text(PROBE)
    run = subprocess.run([sys.executable, '-B', 'probe.py', package], cwd=tmp_path,
                         env=dict(os.environ), capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout)


def _cell(report, name):
    return next(v for k, v in report.items() if k.endswith('._f_' + name))


def test_raised_sample_calls_keep_the_returned_type(tmp_path):
    report = _build(tmp_path)
    # The returned calls decide the type: the two raising items do not widen it to object.
    assert _cell(report, 'factor')['return_type'] == 'double'
    assert _cell(report, 'pay')['return_type'] == 'double'
    # Every sampled call raised: unchanged behaviour, the result stays a Python object.
    assert _cell(report, 'never')['return_type'] == 'object'
    live, compiled = _run(tmp_path, 'exported'), _run(tmp_path, 'exported_cy')
    assert compiled == live
    # Typed result and values on the returning items, the model's own error on the others.
    for key in ('2', '4'):
        kind, tp, value = compiled[key]['factor']
        assert (kind, tp) == ('value', 'float') and value > 0
    for key in ('1', '3'):
        assert compiled[key]['factor'] == ['raised', 'ValueError',
                                           'factor is defined on the life shape only']
        assert compiled[key]['never'][:2] == ['raised', 'LookupError']
