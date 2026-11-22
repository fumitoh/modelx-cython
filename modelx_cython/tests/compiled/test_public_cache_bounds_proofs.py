"""Range proofs must cover both cache entries and every extra formula use."""
import ast
import pytest
import libcst as cst
from modelx_cython.cell_entries import DenseAxis,DenseEntry,EntryModule
from modelx_cython.config import CompilerPolicy
from modelx_cython.entry_pipeline import lower_entries


@pytest.mark.parametrize('public_guard,extra_use,expected',[
    ('0 <= t < 4',False,True),
    ('-1 <= t < 5',False,False),
    ('0 <= t < 4',True,False),
])
@pytest.mark.parametrize('nested', [False,True])
def test_public_cache_range_must_be_checked(public_guard,extra_use,expected,nested):
    call='_mx_cache_entry_0(self, _mx_cache_entry_0(self, t))' if nested else '_mx_cache_entry_0(self, t)'
    source=f'''# Unicode before a private call must not affect proof attachment: 年
class _c_Policy:
    @_mx_cy.cfunc
    def _f_value(self, t):
        return float(t)
    @_mx_cy.cfunc
    def _f_derived(self, t):
        if type(self) is _c_Policy:
            return {call} + 1.0
        return self.value(t) + 1.0
    def value(self, t):
        if 0 <= t < 4:
            return self._f_value(t)
        else:
            raise IndexError()
    def derived(self, t):
        if {public_guard}:
            return self._f_derived(t)
        else:
            raise IndexError()
def _mx_cache_entry_0(self, t):
    if 0 <= t < 4:
        return self._f_value(t)
    else:
        raise IndexError()
def _mx_cache_entry_1(self, t):
    if 0 <= t < 4:
        return self._f_derived(t)
    else:
        raise IndexError()
'''
    if extra_use:source+='def unchecked(s, t):\n    return s._f_derived(t)\n'
    layouts={f'_mx_cache_entry_{i}':DenseEntry('_c_Policy',formula,
        (DenseAxis('t',0,4,'long long'),),False, guard=cst.parse_expression('0 <= t < 4'))
        for i,formula in enumerate(('_f_value','_f_derived'))}
    tree=cst.parse_module(source)
    policy = CompilerPolicy.resolve({'automatic_optimizations': False, 'dense_bounds_proofs': True})
    result = lower_entries({'m': EntryModule(tree)}, {'m': layouts}, {}, policy)['m'].syntax.bytes
    assert (b'_mx_proven_axes' in result) is expected
    if expected and nested:
        result_tree=ast.parse(result)
        calls=[node for node in ast.walk(result_tree) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Name) and node.func.id=='_mx_cache_entry_0']
        outer=next(node for node in calls if isinstance(node.args[1],ast.Call))
        assert outer.args[-1].value==0
        assert outer.args[1].args[-1].value==1
