"""diag-mt: per-step code of a collected Projection takes no reference on objects every worker shares.

Two forms took a reference per call on one shared object, which four workers contended on
(free-threaded CPython: atomic reference counts on the same cache line):

- a table bound to a local, ``tbl = market.curve()``, uses the per-item prepared-table hold and
  the admitted column slots;
- a lowered reduction over the builtin ``sum`` / ``min``: the builtin was bound to an ``object``
  local per call (a reference on the one builtin function); the builtin name is now read directly.

Results must not change: compiled at one and at four workers equals the exported model.
"""
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd


def _model(tmp_path):
    keys = tuple(range(1, 61))
    model = mx.new_model('SharedRefs')
    try:
        portfolio = model.new_space('Portfolio')
        rates = pd.DataFrame({'rate': [0.05, 0.06, 0.07], 'load': [2.0, 3.0, 5.0]}, index=[1, 2, 3])
        data = portfolio.new_space('Inputs', refs={'rates': rates})
        data.new_cells('curve', 'lambda: rates')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'market': data, 'keys': keys},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('rate', '''\
def rate(t):
    tbl = market.curve()
    k = (t + key) % 3 + 1
    return float(tbl.loc[k, "rate"]) * float(tbl.loc[min(k, tbl.index.max()), "load"])
''')
        projection.new_cells('claims', '''\
def claims(t, kind):
    return rate(t) * (kind + 1) / (t + key)
''')
        projection.new_cells('claims_all', '''\
def claims_all(t):
    return sum(claims(t, kind) for kind in range(3))
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(36):
        total += claims_all(t) * key
    return total
''')
        model.export(tmp_path / 'exported', locked_spaces=[data])
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
        '    [p[key].claims_all(t) for t in range(36)]; p[key].value()\n',
        encoding='utf-8')


def test_held_table_local_and_builtin_reduction_take_no_shared_reference(tmp_path):
    _model(tmp_path)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    generated = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    # the table local: resolved once per instance, read borrowed, its columns held
    assert '.PreparedRef, self._mx_tp_0)' in generated and 'if self._mx_tq_0 == 0:' in generated
    assert generated.count('self._mx_tp_0 = self.market._mx_prepared_curve()') == 1
    assert 'root._mx_tp_0 = None' in generated and 'root._mx_tq_0 = 0' in generated
    # the builtin reduction: no object local bound to the builtin
    assert '.f64_accepts(sum)' in generated, generated
    assert '_function = sum' not in generated and '_function: object' not in generated
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
            print(float(one), float(four))
        else:
            print(float(p.collect().sum.value), float(p.collect().sum.value))
        item = p[7]
        values = [item.claims_all(t) for t in range(36)] + [item.rate(t) for t in range(36)] + [item.value()]
        print(repr([float(value) for value in values]))
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    one, four = out['exported_cy'][0].split()
    assert one == out['exported'][0].split()[0], (out, 'one worker equals the exported model')
    # four chunks merge in domain order: equal to the one-worker total within its rounding
    assert abs(float(four) - float(one)) <= 1e-12 * abs(float(one)), out
    assert out['exported_cy'][1:] == out['exported'][1:], out


def test_enum_member_local_is_a_c_integer_and_results_stay(tmp_path, monkeypatch):
    """D83: ``kind = Kind.A if ... else Kind.B`` read only as a compiled Cell's integer argument and in
    comparisons with members is a C integer bound to the module constants (no member object read per
    call); both branches give the exported model's values; four workers = one worker."""
    (tmp_path / 'diag_mt_kinds.py').write_text(
        'import enum\n\nclass Kind(enum.IntEnum):\n    A = 1\n    B = 2\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    from diag_mt_kinds import Kind
    keys = tuple(range(1, 41))
    model = mx.new_model('EnumLocal')
    try:
        portfolio = model.new_space('Portfolio')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'keys': keys, 'Kind': Kind},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('pay', '''\
def pay(t, kind):
    return (t + key) * (1.5 if kind == Kind.A else 0.25)
''')
        projection.new_cells('chosen', '''\
def chosen(t):
    kind = Kind.A if (t + key) % 3 == 0 else Kind.B
    if kind == Kind.B:
        return pay(t, kind) + 1.0
    return pay(t, kind)
''')
        projection.new_cells('named', '''\
def named(t):
    kind = Kind.A if t % 2 == 0 else Kind.B
    return kind.name
''')
        projection.new_cells('value', '''\
def value():
    total = 0.0
    for t in range(30):
        total += chosen(t)
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
        '    [p[key].chosen(t) for t in range(30)]; p[key].value(); [p[key].named(t) for t in range(3)]\n',
        encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    generated = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    assert generated.count('\n    kind: _mx_cy.longlong\n') == 1, generated   # chosen only; named keeps the member
    assert 'kind = _mx_enum_const_0 if (t + self.key) % 3 == 0 else _mx_enum_const_1' in generated
    assert '_mx_definition_Kind.A if' in generated                           # named: an object, as before
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        m = import_module(sys.argv[1]).mx_model
        p = m.Portfolio.Projection
        if sys.argv[1] == 'exported_cy':
            serial = type(m)()
            serial.collect_workers = 1
            one = serial.Portfolio.Projection.collect().sum.value
            print(repr(one), repr(p.collect().sum.value))
        else:
            print(repr(p.collect().sum.value), repr(p.collect().sum.value))
        item = p[5]
        print(repr([item.chosen(t) for t in range(30)]), [item.named(t) for t in range(4)])
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    one, four = out['exported_cy'][0].split()
    assert one == out['exported'][0].split()[0], out
    assert abs(float(four) - float(one)) <= 1e-12 * abs(float(one)), out
    assert out['exported_cy'][1:] == out['exported'][1:], out
