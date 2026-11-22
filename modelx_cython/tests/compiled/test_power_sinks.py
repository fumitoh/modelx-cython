"""Powers whose value flows into a C double are native (D59, D62).

``total = 0.0; total += c * v ** e`` in a loop: the accumulator is a double without the
power, so the power is computed by ``_mx_sys._mx_pow`` (C pow with Python's real-domain
checks) and the loop runs on C numbers; compiled equals the interpreter bit for bit. A
negative base under a fractional exponent raises the native path's ValueError where the
interpreter gives a complex number, as a native double return already does. A power that
stays on Python's semantics inside a loop is listed in the admission report.
"""

import json
import subprocess
import sys

import modelx as mx

ACCUMULATED = '''\
def accumulated(t):
    v = 1.0 / (1.0 + rate())
    total = 0.0
    k = 1
    while k <= 12:
        total += (1.0 / 4) * v ** ((k - t) / 12.0) * 0.5
        k += 1
    for j in range(t):
        total = total + v ** (j / 12.0)
    return total
'''
KEPT = '''\
def kept(t):
    v = rate()
    best = 0.0
    for j in range(3):
        root = v ** (j / 2.0)
        best = max(best, root)
    return best
'''
NEGATIVE = '''\
def negative(t):
    v = -0.5 - rate()
    total = 0.0
    for j in range(1, 3):
        total += v ** (j / 3.0)
    return total
'''
RUN = '''\
import json
from {package} import mx_model as m
p = m.Projection
out = [p.accumulated(t).hex() for t in range(6)] + [repr(p.kept(1))]
try:
    out.append(repr(p.negative(1)))
except ValueError as error:
    out.append('ValueError')
print(json.dumps(out))
'''


def test_powers_into_double_accumulators_are_native(tmp_path):
    model = mx.new_model('PowerSinks')
    try:
        space = model.new_space('Projection')
        space.new_cells('rate', 'lambda: 0.03')
        space.new_cells('accumulated', ACCUMULATED)
        space.new_cells('kept', KEPT)
        space.new_cells('negative', NEGATIVE)
        live = [space.accumulated(t).hex() for t in range(6)] + [repr(space.kept(1)), repr(space.negative(1))]
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    assert 'j' in live[-1]                      # the interpreter gives a complex number
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nfor t in range(6):\n    m.Projection.accumulated(t)\n'
        'm.Projection.kept(1)\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    cells = {key.rsplit('._f_', 1)[-1]: row for key, row in report.items()}
    assert cells['kept']['powers_in_loops'] == ['v ** (j / 2.0)'], cells['kept']
    assert cells['accumulated']['powers_in_loops'] == [], cells['accumulated']
    out = subprocess.run([sys.executable, '-c', RUN.format(package='exported_cy')], cwd=tmp_path,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    compiled = json.loads(out.stdout.strip().splitlines()[-1])
    assert compiled[:7] == live[:7]
    assert compiled[7] == 'ValueError'          # as the native path raises for a native double return
