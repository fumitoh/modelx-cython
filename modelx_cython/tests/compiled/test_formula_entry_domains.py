"""Direct-call admission covers long/dense/scalar/hash cases with one contract."""
import ast
import json
import subprocess
import sys
import textwrap
import modelx as mx


def test_long_and_hash_formula_contracts(tmp_path):
    long='def long_total(t, offset=1):\n    result = '+' + '.join(['value(t)']*24)+'''\n    if t == 2:
        raise failure
    return result - offset
'''
    assert sum(1 for s in ast.parse(long).body[0].body for _ in ast.walk(s))>150
    model=mx.new_model('FormulaContracts')
    try:
        s=model.new_space('Policy',formula='lambda key: None')
        s.failure=RuntimeError('sentinel')
        s.counter=[0]
        s.new_cells('value','''def value(t):
    counter[0] += 1
    return float(t)
''')
        s.new_cells('long_total',long)
        s.new_cells('hash_total','''def hash_total(t, x):
    if t == 2:
        raise failure
    return value(t) + x
''')
        s.new_cells('untyped','lambda t: {"value": t}')
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model\ns=mx_model.Policy[1]\n'
        'for t in (0,1,3):s.long_total(t);s.long_total(t,2);s.hash_total(t,0.25);s.untyped(t)\n')
    (tmp_path/'spec.py').write_text(repr(dict(dense_miss_binding=True,propagate_exact_receiver=True,
        spaces={'Policy':{'cells':{'hash_total':{'numeric_cache_capacity':8}}}})))
    subprocess.run([sys.executable,'-B','-m','modelx_cython','exported'],cwd=tmp_path,check=True)
    admission = json.loads((tmp_path/'exported_cy/_mx_admission.json').read_bytes())
    cells = {name.rsplit('._f_', 1)[-1]: entry for name, entry in admission.items()}
    assert cells['hash_total']['associative_logical_domain'] == {'t': [0, 4]}
    assert cells['untyped']['associative_logical_domain'] == {'t': [0, 4]}
    assert cells['value']['associative_logical_domain'] == {}
    (tmp_path/'override.pyx').write_text(textwrap.dedent('''\
        # cython: freethreading_compatible=True
        from exported_cy._mx_classes cimport _c_Policy
        cdef class Alternate(_c_Policy):
            cdef double _f_value(self, long long t) except? -1:
                return 100.0 + t
        cdef class PublicAlternate(_c_Policy):
            cpdef double value(self, long long t) except? -1:
                return 200.0 + t
        '''))
    (tmp_path/'setup_override.py').write_text('from setuptools import setup\n'
        'from Cython.Build import cythonize\nfrom modelx_cython.build import package_include_path\nsetup(ext_modules=cythonize("override.pyx", include_path=[package_include_path()]))\n')
    subprocess.run([sys.executable,'-B','setup_override.py','build_ext','--inplace'],cwd=tmp_path,check=True)
    subprocess.run([sys.executable,'-B','-c',textwrap.dedent('''\
        from exported_cy import mx_model
        from override import Alternate, PublicAlternate
        s=mx_model.Policy[1]
        assert s.long_total(0)==-1.0 and s.long_total(0)==-1.0
        assert s.long_total(1)==23.0 and s.long_total(1,2)==22.0
        assert s.hash_total(0,-1.0)==-1.0 and s.hash_total(0,-1.0)==-1.0
        assert s.hash_total(1,0.25)==1.25
        for call in (lambda:s.long_total(2),lambda:s.hash_total(2,0.25)):
            for _ in range(2):
                try:call()
                except RuntimeError as error:assert error is s.failure
                else:raise AssertionError('exception cached or lost')
        alternate=Alternate(mx_model.Policy)
        assert alternate.long_total(1)==2423.0
        assert alternate.hash_total(1,0.25)==101.25
        public_alternate=PublicAlternate(mx_model.Policy)
        assert public_alternate.long_total(1)==4823.0
        assert public_alternate.hash_total(1,0.25)==201.25
        fresh=mx_model.Policy[2]
        before=fresh.counter[0]
        assert fresh.value(0)==0.0
        assert fresh.hash_total(0,0.25)==0.25
        assert fresh.counter[0]==before+1
        assert fresh.hash_total(1,0.25)==1.25
        assert fresh.value(1)==1.0
        assert fresh.counter[0]==before+2
        for call, cell in ((lambda:fresh.value(7), 'Policy.value'),
                           (lambda:fresh.hash_total(7,0.25), 'Policy.hash_total')):
            for _ in range(2):
                try:
                    call()
                except IndexError as error:
                    assert cell in str(error)
                else:
                    raise AssertionError('out-of-range dependency was evaluated')
        assert fresh.counter[0]==before+2
        assert s.untyped(1)=={'value':1}
        ''')],cwd=tmp_path,check=True)
