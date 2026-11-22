"""A built-in reduction nested in a table read (original FIA_US_S, RILA_US_S, VA_US_S,
Annuity_JP_S, Pension_KR_S: ``float(sub.loc[max(keys), "rollup_rate"])``).

The lift computes a reduction result for the helper table key. The model builds
without hoisting compiler-only syntax, and the resulting values and exceptions
match the exported model.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx
from modelx_helpers.immutable_inputs import Table


def test_reduction_key_of_a_lowered_table_read_builds_and_matches(tmp_path):
    model = mx.new_model('LiftKey')
    try:
        data = model.new_space('Inputs', refs={
            'rates': Table([1, 3, 5, 8], [(0.01, 2), (0.02, 3), (0.03, 5), (0.04, 7)], ['rate', 'n']),
            'grid': Table([(1, 1), (1, 4), (2, 2), (2, 6)], [(1.5,), (2.5,), (3.5,), (4.5,)], ['f'])})
        data.new_cells('rate_table', 'def rate_table() -> "modelx_helpers.immutable_inputs.Table":\n    return rates\n')
        data.new_cells('grid_table', 'def grid_table() -> "modelx_helpers.immutable_inputs.Table":\n    return grid\n')
        s = model.new_space('Values', refs={'data': data})
        s.new_cells('lo', 'lambda y: 1 + y % 2')
        s.new_cells('rollup', textwrap.dedent('''\
            def rollup(y):
                sub = data.rate_table()
                keys = [k for k in sub.index if k <= y]
                return float(sub.loc[max(keys), "rate"])
            '''))
        s.new_cells('count', textwrap.dedent('''\
            def count(y):
                sub = data.rate_table()
                return lo(y) * int(sub.loc[min(k for k in sub.index if k >= y), "n"]) + 1
            '''))
        s.new_cells('cell', textwrap.dedent('''\
            def cell(y):
                sub = data.grid_table().loc[lo(y)]
                keys = [k for k in sub.index if k <= y]
                rates = data.rate_table()
                return float(sub.loc[max(keys), "f"]) + float(rates.loc[max(k for k in rates.index if k <= y), "rate"])
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'for y in range(4, 9):\n'
        '    m.Values.rollup(y); m.Values.count(y - 3); m.Values.cell(y)\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert '= (_mx_cy.cast)' not in source and '= (object)' not in source
    assert '_mx_hoist = (sub.loc)' in source
    assert '_mx_hoist_ = _mx_reduce_0_best' in source
    assert 'return float(_mx_hoist[_mx_hoist_, "rate"])' in source
    probe = textwrap.dedent('''\
        import json, sys
        from importlib import import_module
        w = import_module(sys.argv[1]).mx_model.Values
        out = []
        for f, ys in ((w.rollup, range(0, 9)), (w.count, range(0, 6)), (w.cell, range(0, 9))):  # sampled ranges
            for y in ys:
                try:
                    out.append([repr(f(y))])
                except Exception as e:
                    out.append(['raise', type(e).__name__, str(e)])
        json.dump(out, open(sys.argv[2], 'w'))
        ''')
    results = []
    for package in ('exported', 'exported_cy'):
        out = tmp_path / f'{package}.json'
        subprocess.run([sys.executable, '-c', probe, package, str(out)], cwd=tmp_path, check=True)
        results.append(json.loads(out.read_text()))
    assert results[0] == results[1]
    assert sum(r[0] != "raise" for r in results[0]) >= 15 and ["raise", "ValueError"] == results[0][0][:2]
