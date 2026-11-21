"""Sample and spec bounds govern both public and private dense calls."""
import json
import os
import subprocess
import sys
import modelx as mx
import pytest


@pytest.mark.parametrize('explicit', [False, True])
def test_dense_bounds_preserve_cache_errors_and_overrides(tmp_path, explicit):
    model=mx.new_model('InferredDomain')
    try:
        p=model.new_space('Projection', refs={'factor': 2., 'fail': True})
        p.new_cells('value', '''def value(t):
    if t == 2 and fail:
        raise ValueError('retry')
    return factor*t
''')
        p.new_cells('forward', 'lambda t: value(t+1)')
        p.new_cells('tupled', 'lambda t, k: value(t) + k')
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model as m
m.Projection.value(0)
m.Projection.forward(0)
m.Projection.forward(2)
m.Projection.tupled(0, 0)
''')
    args=['--no-spec']
    if explicit:
        (tmp_path/'spec.py').write_text(repr({'spaces':{'Projection':{
            'cells_param_size':{'t':6,('t','k'):(4,1)},
            'cells':{'value':{'dense_window':[(0,4)]}}
        }}}))
        args=['--spec','spec.py']
    build=subprocess.run([sys.executable,'-B','-m','modelx_cython','exported','--sample','sample.py',*args],
                         cwd=tmp_path,env=dict(os.environ),capture_output=True,text=True)
    (tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stdout+build.stderr
    admission = json.loads((tmp_path/'exported_cy'/'_mx_admission.json').read_text())
    assert admission['exported_cy._mx_classes._c_Projection._f_value']['exact_spill'] is explicit
    probe='''import pickle,sys
from exported_cy import mx_model as m
p=m.Projection
stop = 6 if sys.argv[1] == 'True' else 4

def rejects_index(call, cell, bounds):
    try:
        call()
    except IndexError as error:
        assert cell in str(error) and bounds in str(error) and 'cells_param_size' in str(error)
    else:
        raise AssertionError('out-of-range index was evaluated')

assert p.value(stop - 1) == 2.0 * (stop - 1)
for key in (-1, stop, 1 << 60):
    rejects_index(lambda: p.value(key), 'Projection.value', f't in [0, {stop})')
rejects_index(lambda: p.forward(stop - 1), 'Projection.value', f't in [0, {stop})')
rejects_index(lambda: p.tupled(0, 1), 'Projection.tupled', 'k in [0, 1)')
assert p.value(1) == 2.0
p.factor = 3.0
assert p.value(1) == 2.0
try:
    p.value(2)
except ValueError as error:
    assert error.args == ('retry',)
else:
    raise AssertionError('formula error was swallowed')
p.fail = False
assert p.value(2) == 6.0
q = pickle.loads(pickle.dumps(p))
assert q.value(1) == 2.0 and q.value(2) == 6.0
assert q.value(stop - 1) == 2.0 * (stop - 1)

class Derived(type(p)):
    def value(self, t):
        return 7.0 * t

d = Derived(m)
d._mx_copy_refs(p, m)
assert d.forward(2) == 21.0
'''
    result=subprocess.run([sys.executable,'-B','-c',probe,str(explicit)],cwd=tmp_path,
                          env=dict(os.environ),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
