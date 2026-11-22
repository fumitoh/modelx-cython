"""A private receiver fact never follows an optional private argument."""
import subprocess
import sys
import modelx as mx


def test_receiver_propagation_preserves_public_defaults(tmp_path):
    model=mx.new_model('DenseDefaults')
    try:
        s=model.new_space('Projection')
        s.new_cells('value','lambda t, offset=2: t * 1.0 + offset')
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model\nfor t in range(4):mx_model.Projection.value(t,2)\n')
    (tmp_path/'spec.py').write_text(repr(dict(dense_miss_binding=True,propagate_exact_receiver=True)))
    subprocess.run([sys.executable,'-B','-m','modelx_cython','exported','--sample','sample.py','--spec','spec.py'],cwd=tmp_path,check=True)
    subprocess.run([sys.executable,'-B','-c',
        'from exported_cy import mx_model\ns=mx_model.Projection\n'
        'assert s.value(0)==2.0\nassert s.value(1)==3.0\n'
        'assert s.value(2,1)==3.0\nassert s.value(3,0)==3.0\n'],cwd=tmp_path,check=True)
