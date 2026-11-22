"""Standard compilation of whole-key expressions and their live/Python contracts."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import modelx as mx
from modelx_helpers.immutable_inputs import Table
from modelx_helpers import typed_inputs as tables


def test_complete_key_expressions_modes_and_native_path(tmp_path):
    model=mx.new_model('CompleteKeys')
    try:
        table=Table((('M',42),('F',43)),((1.5,),('2.5',)),('rate',))
        boolean=Table((('M',True),('M',False)),((3.5,),(4.5,)),('rate',))
        space=model.new_space('Rates',refs={'lookup':tables.lookup,'table':table,'boolean':boolean})
        space.new_cells('amount','''def amount(sex, age):
    value = float(lookup(table, (sex, age), "rate"))
    return value * 1.0
''')
        space.new_cells('flag','lambda flag: float(lookup(boolean, ("M", flag), "rate"))')
        space.new_cells('defaulted','lambda sex, age: float(lookup(table, (sex, age), "rate", "7.5"))')
        assert space.amount('M',42)==1.5 and space.amount('F',43)==2.5
        assert space.flag(True)==3.5 and space.flag(False)==4.5
        assert space.defaulted('X',9)==7.5
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model as m
m.Rates.amount('M',42);m.Rates.amount('F',43)
m.Rates.flag(True);m.Rates.flag(False)
m.Rates.defaulted('M',42);m.Rates.defaulted('X',9)
''')
    # The replacement scenarios use ages 44 and 45 after the sample stops at 43.
    (tmp_path/'spec.py').write_text(repr({'spaces': {'Rates': {
        'cells_param_size': {('sex', 'age'): (1, 46)},
    }}}))
    build=subprocess.run([sys.executable,'-m','modelx_cython','exported','--sample','sample.py','--spec','spec.py'],
                         cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ))
    (tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stdout+build.stderr
    source=(tmp_path/'exported_cy/_mx_classes.py').read_text()
    probe=textwrap.dedent('''
        import importlib,pickle,sys
        from modelx_helpers import typed_inputs as h
        from modelx_helpers.immutable_inputs import Table
        m=importlib.import_module(sys.argv[1]).mx_model
        s=m.Rates
        calls=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is h.lookup.__code__:calls.append('lookup')
        if sys.argv[1].endswith('_cy'):sys.setprofile(profile)
        try:
            assert s.amount('M',42)==1.5
            assert s.amount('F',43)==2.5
            assert s.flag(True)==3.5 and s.flag(False)==4.5
            assert s.defaulted('X',9)==7.5
            try:s.amount('X',42)
            except KeyError as error:assert error.args==(('X',42),)
            else:raise AssertionError('missing key did not raise')
        finally:sys.setprofile(None)
        restored=pickle.loads(pickle.dumps(m))
        assert restored.Rates.amount('F',43)==2.5
        assert restored.Rates.flag(False)==4.5
        s.table=Table((('M',42),('M',44)),((9.5,),(10.5,)),('rate',))
        assert s.amount('M',42)==1.5
        assert s.amount('M',44)==10.5
        fresh=type(m)()
        fresh.Rates.table=s.table
        assert fresh.Rates.amount('M',42)==9.5
        effects=[]
        def replacement(table,key,column,*default):
            effects.append(key);return '8.5'
        s.lookup=replacement
        assert s.amount('M',45)==8.5
        assert effects[-1]==('M',45)
        fresh.Rates.lookup=replacement
        assert fresh.Rates.flag(True)==8.5
        assert type(effects[-1][1]) is bool
        print('financial, native-path, replacement, empty/default/error and pickle checks PASS')
    ''')
    for package in ('exported','exported_cy'):
        run=subprocess.run([sys.executable,'-c',probe,package],cwd=tmp_path,capture_output=True,text=True,env=dict(os.environ))
        (tmp_path/(package+'-validation.log')).write_text(run.stdout+run.stderr)
        assert run.returncode==0,run.stdout+run.stderr
