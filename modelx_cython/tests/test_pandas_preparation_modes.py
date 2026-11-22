"""Explicit preparation obeys live dependencies and exported cache lifetimes."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd

from modelx_helpers import pandas_inputs
from modelx_helpers.typed_inputs import lookup


def test_preparation_live_exported_compiled_and_cleanup(tmp_path):
    model = mx.new_model('PandasPreparation')
    try:
        s = model.new_space('Inputs', refs={'source_frame': pd.DataFrame({'rate': [1.0, 2.0]}, index=[0, 1]),
                                         'table_ctor': pandas_inputs.table, 'lookup': lookup})
        s.new_cells('raw', 'lambda: source_frame')
        s.new_cells('prepared', 'lambda: table_ctor(raw())')
        s.new_cells('value', 'lambda key: lookup(prepared(), key, "rate")')
        assert s.value(0) == 1.0
        original_table = s.prepared()
        s.source_frame = pd.DataFrame({'rate': [3.0, 4.0]}, index=[0, 1])
        assert s.value(0) == 3.0 and s.prepared() is not original_table
        s.raw[()] = pd.DataFrame({'rate': [5.0, 6.0]}, index=[0, 1])
        assert s.value(0) == 5.0
        assert isinstance(s.raw(), pd.DataFrame)
        s.raw.clear()
        assert s.value(0) == 5.0  # clear() preserves explicitly assigned inputs
        s.raw.clear_all()
        assert s.value(0) == 3.0
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\nmx_model.Inputs.value(0)\nmx_model.Inputs.value(1)\n')
    run = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                         cwd=tmp_path, capture_output=True, text=True, env=dict(os.environ))
    (tmp_path / 'build.log').write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    probe = textwrap.dedent('''
        import importlib, pickle, sys
        import pandas as pd
        m = importlib.import_module(sys.argv[1]).mx_model
        s = m.Inputs
        assert isinstance(s.raw(), pd.DataFrame)
        assert s.value(0) == 3.0
        s.source_frame = pd.DataFrame({'rate': [7.0, 8.0]}, index=[0, 1])
        # Exported/compiled execution performs no automatic dependency tracing.
        assert s.value(0) == 3.0
        assert s.value(1) == 4.0
        restored = pickle.loads(pickle.dumps(m))
        assert restored.Inputs.value(0) == 3.0
        # A fresh model instance is explicit full cache/resource cleanup.
        fresh = type(m)()
        fresh.Inputs.source_frame = s.source_frame
        assert fresh.Inputs.value(0) == 7.0
        assert fresh.Inputs.value(1) == 8.0
        assert isinstance(fresh.Inputs.raw(), pd.DataFrame)
    ''')
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             capture_output=True, text=True, env=dict(os.environ))
        (tmp_path / f'{package}-validation.log').write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
