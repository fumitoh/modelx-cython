"""Locals read from typed Cells are C numbers (D59).

A same-Space call to an admitted Cell compiles to ``(entry(self, ...) if exact else
self.cell(...))``. The virtual branch converts to the Cell's C return type, so the whole
expression has that type and ``infer_types`` declares the local assigned from it as a C
number instead of an object. The values are compared with the interpreter, also through a
``while`` loop whose counter and bound come from such locals.
"""

import json
import subprocess
import sys

import modelx as mx

FORMULA = '''\
def factor(t):
    m = freq()
    step = 12 // m
    last = horizon()
    v = 1.0 / (1.0 + rate())
    total = 0.0
    k = 1
    while True:
        s = t + (k - 1) * step
        if s > last:
            break
        total += (1.0 / m) * v ** ((s - t) / 12.0) * surv(s)
        k += 1
    return total
'''
RUN = ('import json\nfrom {package} import mx_model as m\n'
       'print(json.dumps([m.Projection.factor(t).hex() for t in range(0, 30, 3)]))\n')


def test_locals_from_typed_cells_are_c_numbers_and_values_match(tmp_path):
    model = mx.new_model('TypedDispatch')
    try:
        space = model.new_space('Projection')
        space.new_cells('freq', 'lambda: 4')
        space.new_cells('horizon', 'lambda: 40')
        space.new_cells('rate', 'lambda: 0.03')
        space.new_cells('surv', 'lambda s: 0.99 ** s')
        space.new_cells('factor', FORMULA)
        expected = [space.factor(t).hex() for t in range(0, 30, 3)]
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nfor t in range(40):\n    m.Projection.factor(t)\n',
        encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    factor = next(entry for name, entry in admission.items() if name.endswith('._f_factor'))
    # the build report: m and step are C locals of the compiled formula, not Python objects
    assert not {'m', 'step'} & {local.split(' ')[0].split(':')[0] for local in factor['object_locals']}, factor
    for package in ('exported', 'exported_cy'):
        out = subprocess.run([sys.executable, '-c', RUN.format(package=package)], cwd=tmp_path,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
        assert json.loads(out.stdout.strip().splitlines()[-1]) == expected, package
