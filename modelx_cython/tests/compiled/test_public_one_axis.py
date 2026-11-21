"""Public dense checks preserve range and virtual-dispatch contracts."""
import pytest
from modelx_cython.cell_entries import DenseAxis, DenseEntry
import libcst as cst
from modelx_cython.cell_entries import EntryModule
from modelx_cython.config import CompilerPolicy
from modelx_cython.entry_pipeline import lower_entries


@pytest.mark.parametrize('owner,annotation,guard,spill,changed', [
    ('Policy', '_mx_cy.longlong', '-2 <= t < 3', False, True),
    ('Other', '_mx_cy.longlong', '-2 <= t < 3', False, False),
    ('Policy', 'object', '-2 <= t < 3', False, False),
    ('Policy', '_mx_cy.longlong', '-3 <= t < 3', False, False),
    ('Policy', '_mx_cy.longlong', '-2 <= t < 3', True, False),
])
def test_only_proven_public_range_is_normalized(owner, annotation, guard, spill, changed):
    source = f'''class {owner}:
    def value(self, t: {annotation}) -> _mx_cy.double:
        if {guard}:
            return self._f_value(t)
        else:
            raise IndexError(t)
'''
    layout = DenseEntry('Policy', '_f_value', (DenseAxis('t', -2, 3, 'long long'),), spill,
                        guard=cst.parse_expression('-2 <= t < 3'))
    policy = CompilerPolicy.resolve({'automatic_optimizations': False, 'dense_one_axis': True})
    result = lower_entries({'m': EntryModule(cst.parse_module(source))},
                           {'m': {'entry': layout}}, {}, policy)['m'].syntax.code
    assert ('_mx_cy.cast' in result) == changed
    assert 'return self._f_value(t)' in result
    assert 'raise IndexError(t)' in result


@pytest.mark.parametrize('lower,stop', [
    (0, 4), (-2, 3), (-(1 << 63), -(1 << 63) + 4),
    ((1 << 63) - 4, 1 << 63), (-(1 << 63), (1 << 63) - 1),
])
def test_unsigned_range_preserves_signed_extremes(lower, stop):
    mask = (1 << 64) - 1
    keys = {-(1 << 63), (1 << 63) - 1, -1, 0, 1,
            lower - 1, lower, lower + 1, stop - 1, stop, stop + 1}
    for key in keys:
        if -(1 << 63) <= key < 1 << 63:
            assert (lower <= key < stop) == (((key & mask) - (lower & mask)) & mask < stop - lower)
