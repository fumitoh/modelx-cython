"""W1.4: n-argument min/max of typed operands outside native-result scopes (D29a)."""
import json
import math
import subprocess
import sys
import textwrap

import modelx as mx


def test_typed_extrema_semantics(tmp_path):
    model = mx.new_model('TypedExtrema')
    try:
        s = model.new_space('Values', refs={
            'values': (0.0, -0.0, 1.5, math.nan, -math.inf, 3.0, 2.0 ** 62, 2.0 ** 62 + 2 ** 10),
            'ints': (3, -7, 2 ** 62 + 1, 0, 2 ** 62)})
        s.new_cells('v', 'lambda i: values[i]')
        s.new_cells('n', 'lambda i: ints[i]')
        s.new_cells('lo2', 'def lo2(i, j) -> object:\n    return min(v(i), v(j))\n')
        s.new_cells('hi3', 'def hi3(i, j) -> object:\n    return max(v(i), v(j), 0.5)\n')
        s.new_cells('ilo', 'def ilo(i, j) -> object:\n    return min(n(i), n(j))\n')
        s.new_cells('mixed', 'def mixed(i, j) -> object:\n    return max(n(i), v(j))\n')
        s.new_cells('inner', 'lambda i, j: [max(v(i), v(j)) for _ in range(1)][0]')
        s.new_cells('mixed_lo', 'def mixed_lo(i, j) -> object:\n    return min(v(j), n(i))\n')
        s.new_cells('three', 'def three(i, j) -> object:\n    return max(n(i), v(j), n(j))\n')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'w = m.Values\n'
        'for i in range(8):\n'
        '    for j in range(8):\n'
        '        w.lo2(i, j); w.hi3(i, j); w.inner(i, j)\n'
        '        if i < 5: w.mixed(i, j); w.mixed_lo(i, j)\n'
        '        if i < 5 and j < 5: w.three(i, j)\n'
        '        if i < 5 and j < 5: w.ilo(i, j)\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name in ('lo2', 'hi3', 'ilo', 'inner', 'mixed', 'mixed_lo'):
        assert report[f'exported_cy._mx_classes._c_Values._f_{name}']['native_expressions'] == 1, name
    three = report['exported_cy._mx_classes._c_Values._f_three']
    # W1.8: three mixed operands compile as comparisons of Python objects, no built-in call
    assert three['expressions_untyped'] == []
    assert [e['form'] for e in three['expressions_object']] == ['max']
    probe = textwrap.dedent('''\
        import math, struct, sys
        from importlib import import_module
        w = import_module(sys.argv[1]).mx_model.Values
        values = (0.0, -0.0, 1.5, math.nan, -math.inf, 3.0, 2.0 ** 62, 2.0 ** 62 + 2 ** 10)
        ints = (3, -7, 2 ** 62 + 1, 0, 2 ** 62)
        def same(a, b):
            return type(a) is type(b) and (repr(a) == repr(b))
        for i in range(8):
            for j in range(8):
                assert same(w.lo2(i, j), min(values[i], values[j])), (i, j)
                assert same(w.hi3(i, j), max(values[i], values[j], 0.5)), (i, j)
                assert same(w.inner(i, j), max(values[i], values[j])), (i, j)
                if i < 5:
                    assert same(w.mixed(i, j), max(ints[i], values[j])), (i, j)
                    assert same(w.mixed_lo(i, j), min(values[j], ints[i])), (i, j)
                if i < 5 and j < 5:
                    assert same(w.ilo(i, j), min(ints[i], ints[j])), (i, j)
                    assert same(w.three(i, j), max(ints[i], values[j], ints[j])), (i, j)
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
