"""Reads through a pandas object are reported (diag-uc-fr).

A zero-argument cell that returns a pandas DataFrame (here a list selection ``.loc[[k]]`` on a
frame with repeated keys, which a helper Table refuses) and a per-step reader that iterates its
columns run pandas on every step. No mechanism lowers them; before this report they were also
invisible in ``table_reads_untyped``, so such a formula read as clean while it cost ~90k
instructions per step (UC_FR_S ``uc_return_mth``).
"""

import json
import subprocess
import sys

import modelx as mx
import pandas as pd

PREFIX = 'exported_cy._m_Portfolio._mx_classes._c_Projection._f_'


def _model(tmp_path):
    frame = pd.DataFrame({'scenario': ['a', 'a', 'b'], 'lo': [0, 6, 0], 'hi': [5, 99, 99],
                          'rate': [0.01, 0.02, 0.03]}).set_index('scenario')
    model = mx.new_model('Segments')
    try:
        portfolio = model.new_space('Portfolio')
        projection = portfolio.new_space('Projection', refs={'segments_frame': frame})
        projection.new_cells('scenario', 'lambda: "a"')
        projection.new_cells('segments', '''\
def segments():
    return segments_frame.loc[[scenario()]]
''')
        projection.new_cells('rate', '''\
def rate(t):
    tbl = segments()
    for lo, hi, r in zip(tbl["lo"], tbl["hi"], tbl["rate"]):
        if int(lo) <= t <= int(hi):
            return float(r)
    raise ValueError("no segment")
''')
        projection.new_cells('labels', '''\
def labels():
    return segments_frame.index
''')
        projection.new_cells('uses', '''\
def uses(t):
    n = len(segments()) + segments().reset_index().shape[0]
    for _, row in segments().iterrows():
        n += 0 * row["lo"]
    return t * n if "a" in labels() else 0
''')
        projection.new_cells('total', '''\
def total(t):
    return rate(t) + (total(t - 1) if t > 0 else 0.0)
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'assert [p.rate(t) for t in (0, 5, 6, 9)] == [0.01, 0.01, 0.02, 0.02]\n'
        'p.total(9)\n'
        'assert [p.uses(t) for t in range(10)] == [4 * t for t in range(10)]\n', encoding='utf-8')


def _translate(tmp_path, translate_only=True):
    args = [sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec']
    if translate_only:
        args.append('--translate-only')
    done = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    return report, done.stdout + done.stderr


def test_per_step_reads_through_a_pandas_cell_are_reported_and_warned(tmp_path):
    _model(tmp_path)
    report, output = _translate(tmp_path)
    rate = report[PREFIX + 'rate']
    assert rate['table_reads_typed'] == 0
    assert [(e['read'], e['kind'], e['line'], e['per_step']) for e in rate['table_reads_untyped']] == [
        ('tbl["lo"]', 'column', 3, True), ('tbl["hi"]', 'column', 3, True), ('tbl["rate"]', 'column', 3, True)]
    assert all(e['reason'].startswith('receiver is a pandas DataFrame, Series or Index, not a helper Table')
               for e in rate['table_reads_untyped'])
    uses = report[PREFIX + 'uses']['table_reads_untyped']      # methods, bare uses, an Index cell
    assert sorted((e['read'], e['kind'], e['line'], e['per_step']) for e in uses) == [
        ('row["lo"]', 'row', 4, True), ('self.labels()', 'use', 5, True), ('self.segments()', 'use', 2, True),
        ('self.segments().iterrows()', 'iterrows', 3, True),
        ('self.segments().reset_index().shape', 'attribute', 2, True)], uses     # the loop row is no 'use'
    segments = report[PREFIX + 'segments']['table_reads_untyped']     # once per policy: listed, not warned
    assert [(e['read'], e['kind'], e['per_step']) for e in segments] == [
        ('self.segments_frame.loc[[self.scenario()]]', 'loc', False)]
    warned = [line for line in output.splitlines() if 'per-step table read is not typed:' in line]
    assert len(warned) == 8 and sum(f'{PREFIX}rate:' in line for line in warned) == 3, output
    assert report[PREFIX + 'total']['table_reads_untyped'] == []


def test_pandas_report_leaves_the_generated_formula_and_values_alone(tmp_path):
    _model(tmp_path)
    _translate(tmp_path, translate_only=False)
    classes = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    assert 'zip(tbl["lo"], tbl["hi"], tbl["rate"])' in classes      # kept as written
    check = subprocess.run([sys.executable, '-c', '''
from exported_cy import mx_model
p = mx_model.Portfolio.Projection
assert [p.rate(t) for t in (0, 5, 6, 9)] == [0.01, 0.01, 0.02, 0.02]
rates = [0.01] * 6 + [0.02] * 4
expected = 0.0
for t in range(10):
    expected = rates[t] + expected
assert p.total(9) == expected, p.total(9)
'''], cwd=tmp_path, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
