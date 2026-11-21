"""Automatic storage preserves valid cold indices and published cache state."""
import json
import subprocess
import sys
import textwrap
import modelx as mx


def test_default_policy_needs_no_cell_tuning(tmp_path):
    model=mx.new_model('AutomaticPolicy')
    try:
        s=model.new_space('Policy',formula='lambda key: None')
        s.seen=[0,0]
        s.new_cells('narrow','''def narrow(t):
    seen[0] += 1
    return t * 1.0
''')
        s.new_cells('wide','lambda t: t * 1.0')
        s.new_cells('factor','lambda t: 1.05 ** (1 / 12) + t * 0.0')
        s.new_cells('keyed','''def keyed(rate):
    seen[1] += 1
    return rate - 1.0
''')
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model\ns=mx_model.Policy[1]\ns.narrow(5);s.wide(20);s.keyed(0.0);s.keyed(0.5);s.factor(5)\n')
    subprocess.run([sys.executable,'-B','-m','modelx_cython','exported','--no-spec'],cwd=tmp_path,check=True)
    admission=json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    narrow=next(v for k,v in admission.items() if k.endswith('._f_narrow'))
    keyed=next(v for k,v in admission.items() if k.endswith('._f_keyed'))
    assert narrow['dense_offsets']==[5] and narrow['exact_spill']
    assert keyed['cache']=='native-numeric-keys'
    (tmp_path/'probe.py').write_text(textwrap.dedent('''\
        import pickle
        from exported_cy import mx_model
        s=mx_model.Policy[2]
        assert s.narrow(5)==5.0 and s.narrow(5)==5.0
        # Keys not observed for this Cell remain in its original logical domain.
        assert s.narrow(0)==0.0 and s.narrow(7)==7.0
        assert s.narrow(7)==7.0 and s.seen[0]==3
        try:
            s.narrow(21)
        except IndexError as error:
            assert 'Policy.narrow' in str(error) and 'cells_param_size' in str(error)
        else:
            raise AssertionError('automatic compact storage admitted an out-of-range index')
        assert s.seen[0]==3
        assert s.keyed(0.0)==-1.0 and s.keyed(-0.0)==-1.0
        assert s.keyed(0.5)==-0.5 and s.keyed(0.75)==-0.25
        assert s.seen[1]==3
        expected=s.factor(5)
        restored=pickle.loads(pickle.dumps(mx_model)).Policy[2]
        assert restored.keyed(0.0)==-1.0 and restored.keyed(0.5)==-0.5
        assert restored.keyed(0.75)==-0.25 and restored.seen[1]==3
        assert restored.keyed(1.25)==0.25 and restored.seen[1]==4
        assert restored.narrow(7)==7.0 and restored.seen[0]==3
        assert restored.factor(5)==expected and restored.factor(6)==expected
        '''))
    subprocess.run([sys.executable,'-B','probe.py'],cwd=tmp_path,check=True)
