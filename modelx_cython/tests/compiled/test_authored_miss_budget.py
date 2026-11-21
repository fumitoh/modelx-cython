"""Private formula calls require a proven contract, independent of length."""
import ast
import subprocess
import sys
import textwrap

import libcst as cst
import modelx as mx
import pytest
from modelx_cython.entry_pipeline import _eligible


def eligible(body, guard_safe=True):
    source='@_mx_cy.cfunc\ndef formula(self, t: _mx_cy.longlong) -> _mx_cy.double:\n'+textwrap.indent(body,'    ')
    return _eligible(cst.parse_module(source).body[0],('t',),guard_safe)


@pytest.mark.parametrize('body',[
    'return locals()\n', 'return (lambda: t)()\n',
    'return self.__class__\n', 'return sum(x for x in range(t))\n',
    'self = None\nreturn t * 1.0\n',
    'try:\n    return t * 1.0\nexcept Exception as self:\n    return 0.0\n',
    'import math as self\nreturn t * 1.0\n',
    pytest.param('match t:\n    case self:\n        return 0.0\n',
        marks=pytest.mark.skipif(sys.version_info < (3,10),reason='pattern matching requires Python 3.10')),
])
def test_authored_budget_keeps_full_emitted_safety_checks(body):
    assert not eligible(body)


def test_formula_length_is_not_call_eligibility():
    body='return '+' + '.join(['t']*60)+'\n'
    assert eligible(body)
    assert not eligible('return t * 1.0\n',False)


def test_unknown_signature_is_not_admitted():
    source='@_mx_cy.cfunc\ndef formula(self, t) -> object:\n    return t\n'
    assert not _eligible(cst.parse_module(source).body[0],('t',),True)


def test_guard_shadowing_stays_on_method_path(tmp_path):
    model=mx.new_model('GuardShadowing')
    try:
        s=model.new_space('Projection')
        s.new_cells('value','lambda t, type: float(t + type)')
        s.new_cells('other','lambda t, _c_Projection: float(t + _c_Projection)')
        s.failure=ValueError('sentinel')
        s.new_cells('risky','''def risky(t):
    if t == 2:
        raise failure
    return float(t)
''')
        s.new_cells('catches','''def catches(t):
    try:
        return risky(t)
    except ValueError as _mx_exact_receiver:
        return -1.0
''')
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model\ns=mx_model.Projection\n'
        'for t in range(4):s.value(t,2);s.other(t,2);s.catches(t)\n')
    (tmp_path/'spec.py').write_text(repr(dict(dense_miss_binding=True,propagate_exact_receiver=True)))
    subprocess.run([sys.executable,'-B','-m','modelx_cython','exported'],cwd=tmp_path,check=True)
    subprocess.run([sys.executable,'-B','-c','from exported_cy import mx_model\n'
        's=mx_model.Projection\nassert s.value(1,2)==3.0\nassert s.other(1,2)==3.0\n'
        'assert s.catches(1)==1.0\nassert s.catches(2)==-1.0\n'],cwd=tmp_path,check=True)


def test_small_authored_formula_with_expanded_calls(tmp_path):
    formula='def total(t, offset=1):\n    result = '+' + '.join(['value(t)']*12)+'''\n    if t == 2:
        raise failure
    return result - offset
'''
    assert sum(1 for s in ast.parse(formula).body[0].body for _ in ast.walk(s))<=150
    model=mx.new_model('AuthoredBudget')
    try:
        s=model.new_space('Policy',formula='lambda key: None')
        s.failure=RuntimeError('sentinel')
        s.new_cells('value','lambda t: float(t)')
        s.new_cells('total',formula)
        model.export(tmp_path/'exported')
    finally:model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model\ns=mx_model.Policy[1]\n'
        'for t in (0,1,3):\n    s.total(t);s.total(t,2)\n')
    (tmp_path/'spec.py').write_text(repr(dict(dense_miss_binding=True,propagate_exact_receiver=True)))
    subprocess.run([sys.executable,'-B','-m','modelx_cython','exported'],cwd=tmp_path,check=True)
    (tmp_path/'override.pyx').write_text(textwrap.dedent('''\
        # cython: freethreading_compatible=True
        from exported_cy._mx_classes cimport _c_Policy
        cdef class Alternate(_c_Policy):
            cdef double _f_value(self, long long t) except? -1:
                return 100.0 + t
        '''))
    (tmp_path/'setup_override.py').write_text('from setuptools import setup\n'
        'from Cython.Build import cythonize\nfrom modelx_cython.build import package_include_path\nsetup(ext_modules=cythonize("override.pyx", include_path=[package_include_path()]))\n')
    subprocess.run([sys.executable,'-B','setup_override.py','build_ext','--inplace'],cwd=tmp_path,check=True)
    subprocess.run([sys.executable,'-B','-c',textwrap.dedent('''\
        from exported_cy import mx_model
        from override import Alternate
        s=mx_model.Policy[1]
        assert s.total(0)==-1.0 and s.total(0)==-1.0
        assert s.total(1)==11.0 and s.total(1,2)==10.0
        for _ in range(2):
            try:s.total(2)
            except RuntimeError as error:assert error is s.failure
            else:raise AssertionError('exception cached or lost')
        assert Alternate(mx_model.Policy).total(1)==1211.0
        ''')],cwd=tmp_path,check=True)
