"""Declared object sums retain identity; ordinary numeric Cells may widen."""
import json
import os
import subprocess
import sys

import modelx as mx


def test_scalar_sum_variant_cache_modes(tmp_path):
    model = mx.new_model('ScalarVariants')
    try:
        space = model.new_space('Values', refs={'large':2**53+1})
        space.new_cells('term', 'lambda t: 1.5*t')
        space.new_cells('total', 'def total(n) -> object:\n    return sum(term(t) for t in range(n))\n')
        space.new_cells('twice', 'def twice(n) -> object:\n    return 2*total(n)\n')
        space.new_cells('numeric_total', 'lambda n: sum(term(t) for t in range(n))')
        space.new_cells('precise', '''def precise(n):
    if n == 0:
        return large
    return 1.5
''')
        space.new_cells('precise_expression', 'lambda n: large if n == 0 else 1.5')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\nfor n in [0,1,3]:\n    mx_model.Values.total(n);mx_model.Values.twice(n);mx_model.Values.numeric_total(n);mx_model.Values.precise(n);mx_model.Values.precise_expression(n)\n')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
        '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, capture_output=True,
        text=True, env=dict(os.environ))
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    report = json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    assert report['exported_cy._mx_classes._c_Values._f_numeric_total']['return_type'] == 'double'
    probe = '''import importlib,pickle,sys
import pandas as pd
m=importlib.import_module(sys.argv[1]).mx_model
for root in [type(m)(),pickle.loads(pickle.dumps(type(m)()))]:
    s=root.Values
    for _ in range(2):
        assert s.total(0)==0 and type(s.total(0)) is int
        assert s.twice(0)==0 and type(s.twice(0)) is int
        assert s.total(3)==4.5 and type(s.total(3)) is float
        assert s.twice(3)==9.0 and type(s.twice(3)) is float
        assert str(pd.DataFrame({'amount':[s.total(0)]}).dtypes.iloc[0])=='int64'
        assert s.numeric_total(0)==0 and s.numeric_total(3)==4.5
        if sys.argv[1].endswith('_cy'):assert type(s.numeric_total(0)) is float
        assert s.precise(0)==2**53+1 and s.precise(1)==1.5
        assert s.precise_expression(0)==2**53+1 and s.precise_expression(1)==1.5
'''
    for package in ['exported', 'exported_cy']:
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             capture_output=True, text=True, env=dict(os.environ))
        (tmp_path / (package + '-validation.log')).write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
