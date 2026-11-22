"""Public ordered-key helper parity in live/exported/compiled execution."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import modelx as mx
from modelx_helpers.immutable_inputs import Table
from modelx_helpers.table_keys import keys


def test_keys_helper_modes_and_current_reference(tmp_path):
    model=mx.new_model('OrderedKeys')
    try:
        table=Table((9,-3,5),((1.0,),)*3,('rate',))
        s=model.new_space('Inputs',refs={'keys':keys,'table':table})
        s.new_cells('ordered','lambda: keys(table)')
        s.new_cells('fresh','lambda: keys(table)').is_cached=False
        assert s.ordered()==(9,-3,5) and s.ordered() is keys(table)
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model\nmx_model.Inputs.ordered();mx_model.Inputs.fresh()\n')
    build=subprocess.run([sys.executable,'-m','modelx_cython','exported','--sample','sample.py','--no-spec'],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ))
    (tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stdout+build.stderr
    probe=textwrap.dedent('''
        import importlib,pickle,sys
        from modelx_helpers.immutable_inputs import Table
        m=importlib.import_module(sys.argv[1]).mx_model
        s=m.Inputs
        assert s.ordered()==(9,-3,5)
        s.table=Table((1,2),((1.0,),)*2,('rate',))
        assert s.ordered()==(9,-3,5)
        assert s.fresh()==(1,2)
        restored=pickle.loads(pickle.dumps(m))
        assert restored.Inputs.ordered()==(9,-3,5)
        assert restored.Inputs.fresh()==(1,2)
        calls=[]
        def replacement(table):calls.append(table);return ('current',)
        s.keys=replacement
        assert s.fresh()==('current',)
        assert calls==[s.table]
        assert s.ordered()==(9,-3,5)
    ''')
    for package in ('exported','exported_cy'):
        run=subprocess.run([sys.executable,'-c',probe,package],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ))
        (tmp_path/(package+'-validation.log')).write_text(run.stdout+run.stderr)
        assert run.returncode==0,run.stdout+run.stderr
