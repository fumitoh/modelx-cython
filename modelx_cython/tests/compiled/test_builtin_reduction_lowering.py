"""Compile the owned lowering/provider without editing shared pipeline wiring."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import libcst as cst
from modelx_helpers import typed_inputs as operations

from modelx_cython.helpers import module_name
from modelx_cython.helper_members import module_helpers
from modelx_cython.reductions import NativeReductions


def test_compiled_builtin_reduction_lowering(tmp_path):
    def info(result, params):
        return SimpleNamespace(params=params, has_formula_def=True,
            has_typeinfo=lambda: True, called_with_kwargs=False, body_has_closure=False,
            formula_allows_exact_receiver=True,
            get_rettype_expr=lambda **kw: result,
            get_argtype_expr=lambda *args, **kw: 'long long' if kw.get('c_style') else '_mx_cy.longlong')
    cls = SimpleNamespace(name='_c_Test',
        refs={'operations': SimpleNamespace(helper_members=module_helpers(operations))},
        cells={'predicate': info('bint', ('i',)), 'amount': info('double', ('i',))})
    alias = module_name('modelx_cython._native_iterables')
    source = Path(__file__).parents[3] / 'modelx_cython'
    for suffix in ('.pyx', '.pxd'):
        (tmp_path / (alias + suffix)).write_bytes((source / ('_native_iterables' + suffix)).read_bytes())
    from modelx_cython import runtime as native_runtime
    helper_alias = module_name('modelx_cython.runtime._native_inputs')
    for suffix in ('.pyx', '.pxd'):
        (tmp_path / (helper_alias + suffix)).write_bytes(
            (Path(native_runtime.__file__).parent / ('_native_inputs' + suffix)).read_bytes())
    methods, cold = [], []
    for operation in ('all', 'any', 'min', 'max'):
        boolean = operation in ('all', 'any')
        result = 'bint' if boolean else 'long long'
        annotation = '_mx_cy.bint' if boolean else '_mx_cy.longlong'
        payload = 'self.predicate(i)' if boolean else 'i'
        text = (f'def {operation}_check(self, n: _mx_cy.longlong) -> {annotation}:\n'
                f'    return {operation}({payload} for i in range(n) if i != 1)\n')
        formula = cst.parse_module(text).body[0]
        pass_ = NativeReductions(cls, info(result, ('n',)), set(), formula,
            lambda op=operation: '_cold_' + op, True)
        rewritten = cst.Module([formula.visit(pass_)]).code
        assert pass_.count == 0
        assert rewritten == text
        methods.append('\n'.join('    ' + line if line else '' for line in rewritten.splitlines()))
        cold.extend(cst.Module([function]).code for function in pass_.fallbacks)
    prefix = f'''import cython as _mx_cy
from cython.cimports import {alias}
from cython.cimports import {helper_alias}
from modelx_helpers import typed_inputs as operations
@_mx_cy.cclass
class _c_Test:
    hook: object
    operations: object
    def __init__(self):
        self.hook = lambda i: i != 2
        self.operations = operations
    @_mx_cy.ccall
    def set_hook(self, function: object):
        self.hook = function
    @_mx_cy.ccall
    def predicate(self, i: _mx_cy.longlong) -> _mx_cy.bint:
        return self.hook(i)
    @_mx_cy.ccall
    def amount(self, i: _mx_cy.longlong) -> _mx_cy.double:
        return 1.0 + i * 0.5
    @_mx_cy.ccall
    def set_operations(self, value: object):
        self.operations = value
'''
    (tmp_path / 'probe.py').write_text(prefix + '\n'.join(methods) + '\n' + '\n'.join(cold))
    (tmp_path / 'setup.py').write_text('from setuptools import setup\nfrom Cython.Build import cythonize\nfrom modelx_cython.build import package_include_path\n'
        f'setup(ext_modules=cythonize(["probe.py", "{alias}.pyx", "{helper_alias}.pyx"], include_path=[package_include_path()], compiler_directives={{"freethreading_compatible":True}}))\n')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=tmp_path, check=True)
    subprocess.run([sys.executable, '-c', '''from probe import _c_Test
p = _c_Test()
assert p.all_check(0) is True and p.all_check(4) is False
assert p.any_check(0) is False and p.any_check(4) is True
assert p.min_check(4) == 0 and p.max_check(4) == 3
for name in ('min', 'max'):
    try:
        getattr(__import__('builtins'), name)(())
    except ValueError as error:
        expected = error.args
    try:
        getattr(p, name + '_check')(0)
    except ValueError as error:
        assert error.args == expected
    else:
        raise AssertionError('empty extrema did not raise')
seen = []
error = StopIteration('sentinel')
def broken(i):
    seen.append(i)
    raise error
p.set_hook(broken)
for name in ('all', 'any'):
    try:
        getattr(p, name + '_check')(4)
    except RuntimeError as actual:
        assert actual.__cause__ is error and actual.__context__ is error
    else:
        raise AssertionError('body StopIteration not converted')
assert seen == [0, 0]
'''], cwd=tmp_path, check=True)
