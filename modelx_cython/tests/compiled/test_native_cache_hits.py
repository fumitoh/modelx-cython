"""Native hits preserve shared caches, errors, ranges, and virtual overrides."""
import subprocess
import sys
import textwrap
import modelx as mx


def test_native_cache_hit_contract(tmp_path):
    model=mx.new_model('CacheHitContract')
    try:
        s=model.new_space('Policy',formula='lambda key: None')
        s.counter=[0,0,0];s.state=[False];s.failure=RuntimeError('cache-hit sentinel')
        s.new_cells('scalar','''def scalar():
    counter[0] += 1
    return -1.0
''')
        s.new_cells('value','''def value(u):
    counter[1] += 1
    return -1.0 + u * 0.5
''')
        s.new_cells('total','''def total(t=0):
    return scalar() + scalar() + value(t) + value(t)
''')
        s.new_cells('retry','''def retry(t):
    counter[2] += 1
    if state[0]:
        raise failure
    return -1.0 + t * 0.5
''')
        s.new_cells('retry_total','lambda t: retry(t) + retry(t)')
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model\ns=mx_model.Policy[1]\nfor t in range(3):s.total(t);s.retry_total(t)\ns.value(3)\n')
    (tmp_path/'spec.py').write_text(repr({'dense_miss_binding':True,'propagate_exact_receiver':True,
        'dense_one_axis':True,'dense_bounds_proofs':True,'native_cache_hits':'both',
        'spaces':{'Policy':{'cells_param_size':{'t':5}}}}))
    subprocess.run([sys.executable,'-B','-m','modelx_cython','exported'],cwd=tmp_path,check=True)
    (tmp_path/'override.pyx').write_text(textwrap.dedent('''\
        # cython: freethreading_compatible=True
        from exported_cy._mx_classes cimport _c_Policy
        cdef class FormulaOverride(_c_Policy):
            cdef double _f_value(self, long long u) except? -1:
                return 100.0 + u
        cdef class PublicOverride(_c_Policy):
            cpdef double value(self, long long u) except? -1:
                return 200.0 + u
        '''))
    (tmp_path/'setup_override.py').write_text('from setuptools import setup\nfrom Cython.Build import cythonize\nfrom modelx_cython.build import package_include_path\nsetup(ext_modules=cythonize("override.pyx", include_path=[package_include_path()]))\n')
    subprocess.run([sys.executable,'-B','setup_override.py','build_ext','--inplace'],cwd=tmp_path,check=True)
    (tmp_path/'probe.py').write_text(textwrap.dedent('''\
        from exported_cy import mx_model
        from override import FormulaOverride,PublicOverride
        s=mx_model.Policy[2]
        assert s.total()==-4.0 and s.total()==-4.0
        assert s.total(t=1)==-3.0
        assert s.counter[:2]==[1,2],s.counter
        assert s.scalar()==-1.0 and s.value(0)==-1.0
        assert s.counter[:2]==[1,2]
        # The t specification declares [0,5); u only has a sampled window.
        assert s.total(3)==-1.0 and s.total(3)==-1.0
        assert s.value(3)==0.5 and s.counter[:2]==[1,3]
        for t in (5,-1,2**63-1,-2**63):
            for repeat in range(2):
                try:s.total(t)
                except IndexError:pass
                else:raise AssertionError(('lost domain check',t))
        s.state[0]=True
        for repeat in range(2):
            try:s.retry_total(0)
            except RuntimeError as error:assert error is s.failure
            else:raise AssertionError('error swallowed or cached')
        assert s.counter[2]==2
        s.state[0]=False
        assert s.retry_total(0)==-2.0 and s.retry_total(0)==-2.0
        assert s.counter[2]==3,s.counter
        formula_override=FormulaOverride(mx_model.Policy)
        public_override=PublicOverride(mx_model.Policy)
        # Synthetic extension subclasses initialize their object References.
        formula_override.counter=[0,0,0]
        public_override.counter=[0,0,0]
        assert formula_override.total(1)==200.0
        assert public_override.total(1)==400.0
        '''))
    subprocess.run([sys.executable,'-B','probe.py'],cwd=tmp_path,check=True)
