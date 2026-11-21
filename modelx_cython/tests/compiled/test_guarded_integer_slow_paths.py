"""Integer locals whose exact values leave the 64-bit range, compiled.

Formerly the slow paths of the D76 guards (an exact Python fallback). The guards are gone
(owner's rule): in the checked mode each of these formulas raises OverflowError naming the
Cell where the exported model carries on with a big int; a small value is exact. In the
unchecked mode a small value is exact and the large ones are plain C arithmetic.
"""

import json
import subprocess
import sys

import modelx as mx

RUN = '''
import json
from {package} import mx_model as m
out = []
for name, base in (('cmp', (1 << 62) + 7), ('div', -(1 << 63) + 1), ('flt', (1 << 62) + 7),
                   ('loc', (1 << 62) + 7), ('loc', 5)):
    v = type(m)().Values
    v.base = base
    try:
        out.append([name, repr(getattr(v, name)(0, 2))])
    except OverflowError as error:
        out.append([name, 'OverflowError: ' + str(error)])
print(json.dumps(out))
'''


def test_guarded_slow_paths_are_exact_or_name_the_cell(tmp_path):
    exported, compiled = _slow_paths(tmp_path, 'checked')
    assert compiled[4] == exported[4]                   # a small value: exact
    for (name, value), (_, reference) in zip(compiled[:4], exported[:4]):
        assert not reference.startswith('OverflowError')
        assert value.startswith(f'OverflowError: Values.{name}'), (name, value)


def test_slow_paths_unchecked_mode(tmp_path):
    exported, compiled = _slow_paths(tmp_path, 'unchecked')
    assert compiled[4] == exported[4]                   # a small value: exact
    assert all(not value.startswith('OverflowError') for _, value in compiled[:4])   # plain C


def _slow_paths(tmp_path, mode):
    model = mx.new_model('GuardSlowPaths')
    try:
        space = model.new_space('Values', refs={'base': 3})
        space.new_cells('cmp', 'def cmp(a, b):\n    k = base + a\n    return 1.0 if k * b <= 0.5 else 0.0\n')
        space.new_cells('div', 'def div(a, b):\n    k = base + a\n    return (k - b) / 12.0\n')
        space.new_cells('flt', 'def flt(a, b):\n    k = base + a\n    return float(k * b)\n')
        space.new_cells('loc', 'def loc(a, b):\n    k = base + a\n    total = 0.0\n'
                               '    for j in range(1, 3):\n        n = k * b + j\n        total += n / 2.0\n'
                               '    return total\n')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nv = m.Values\nfor a in range(5):\n'
        '    v.cmp(a, 2); v.div(a, 2); v.flt(a, 2); v.loc(a, 2)\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'integer_checks': mode}), encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    results = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', RUN.format(package=package)], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        results[package] = json.loads(run.stdout.strip().splitlines()[-1])
    return results['exported'], results['exported_cy']
