"""Existing public object-return specifications preserve raw scalar identities.

No protected compiler changes or automatic source-annotation inference. Explicit
object declarations on forwarding Cells remain part of the model ABI.
"""
import os
import json
import subprocess
import sys

import modelx as mx
import numpy as np
import pytest

from modelx_helpers import pandas_inputs, typed_inputs


@pytest.mark.parametrize('use_spec', [False, True])
def test_public_object_return_specs_all_modes(tmp_path, use_spec):
    import pandas as pd
    frame = pd.DataFrame({'rate': pd.Series([np.float64(.1), np.float64(.2), np.float64(0.)], dtype=object)})
    model = mx.new_model('DeclaredObjectResults')
    try:
        p = model.new_space('Projection', refs={'source': frame, 'prepare': pandas_inputs.table, 'lookup_object': typed_inputs.lookup_object,
            'lookup_float': typed_inputs.lookup_float, 'array_source': np.array([1., 2.])})
        p.new_cells('owned', 'lambda: prepare(source, preserve_scalar_types=True)')
        p.new_cells('raw', '''def raw(key) -> object:
    return lookup_object(owned(), key, "rate")
''')
        p.new_cells('local', '''def local(key) -> object:
    value = lookup_object(owned(), key, "rate")
    return value
''')
        p.new_cells('forward', '''def forward(key) -> object:
    return raw(key)
''')
        p.new_cells('scalar', '''def scalar() -> object:
    return raw(0)
''')
        p.new_cells('retained', '''def retained(kind=None) -> object:
    return next(map(lambda key: raw(key) if kind is None else local(key), (0,)))
''')
        p.new_cells('uncached', '''def uncached(key) -> object:
    return raw(key)
''')
        p.uncached.is_cached = False
        p.new_cells('array', '''def array() -> object:
    return array_source
''')
        p.new_cells('arithmetic', '''def arithmetic(key) -> object:
    return raw(key) / raw(key)
''')
        p.new_cells('ordinary', 'lambda key: lookup_float(owned(), key, "rate")')
        assert type(p.raw(0)) is np.float64 and p.array() is p.array_source
        model.export(tmp_path/'exported')
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model as m
p=m.Projection
for key in (0,1):
 p.raw(key);p.local(key);p.forward(key);p.uncached(key);p.ordinary(key)
p.scalar();p.retained();p.retained("local");p.array();p.arithmetic(0);p.arithmetic(2)
''')
    object_cells = ['raw','local','forward','scalar','retained','uncached','array','arithmetic']
    spec = {'spaces': {'Projection': {'cells': {name: {'return_type': 'object'}
                                              for name in object_cells}}}}
    (tmp_path/'return-types.py').write_text(json.dumps(spec,indent=2)+'\n')
    environment = dict(os.environ)
    run = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
        '--sample', 'sample.py', *(['--spec', 'return-types.py'] if use_spec else ['--no-spec'])], cwd=tmp_path, env=environment,
        capture_output=True, text=True)
    (tmp_path/'build.log').write_text(run.stdout+run.stderr)
    assert run.returncode == 0, run.stdout+run.stderr
    report = json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    types = {name.rsplit('._f_', 1)[-1]: entry['return_type'] for name, entry in report.items()}
    assert types['raw'] == 'object' and types['ordinary'] == 'double', types
    probe = '''import importlib,pickle,sys,warnings,numpy as np
m=importlib.import_module(sys.argv[1]).mx_model
p=type(m)().Projection
for key in (0,1):
 expected=p.source.loc[key,"rate"]
 for name in ("raw","local","forward","uncached"):
  function=getattr(p,name)
  assert function(key)is expected,(name,type(function(key)))
  assert function(key)is expected
 assert type(p.ordinary(key))is float
assert p.scalar()is p.raw(0)
assert p.retained()is p.raw(0) and p.retained("local")is p.raw(0)
assert p.array()is p.array_source
with warnings.catch_warnings(record=True) as seen:
 warnings.simplefilter("always")
 result=p.arithmetic(2)
assert type(result)is np.float64 and np.isnan(result)
assert [(type(w.message),str(w.message)) for w in seen]==[(RuntimeWarning,"invalid value encountered in scalar divide")]
restored=pickle.loads(pickle.dumps(p))
assert type(restored.raw(0))is np.float64
assert restored.scalar()is restored.raw(0)
assert restored.array()is restored.array_source
'''
    for package in ['exported', 'exported_cy']:
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
            env=environment, capture_output=True, text=True)
        (tmp_path/(package+'-validation.log')).write_text(run.stdout+run.stderr)
        assert run.returncode == 0, run.stdout+run.stderr
