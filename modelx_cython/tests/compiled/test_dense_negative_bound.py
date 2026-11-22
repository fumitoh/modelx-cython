"""D15: the public one-axis range check stays overflow-free under ``overflowcheck``.

The round-2 text ``cast(ulonglong, key) - cast(ulonglong, lower) < span`` raises
OverflowError under Cython's overflow check whenever key < lower, i.e. on every call
when an axis starts below zero (BU_DE_S: an axis from -9). The emitted check is now
``cast(ulonglong, key - (lower)) < span`` and ``cast(ulonglong, key) < span`` for lower 0.
"""
import pytest
import libcst as cst

from modelx_cython.cell_entries import DenseAxis, DenseEntry, EntryModule
from modelx_cython.config import CompilerPolicy
from modelx_cython.entry_pipeline import lower_entries


def _check_text(lower, stop):
    source = f'''class Policy:
    def value(self, t: _mx_cy.longlong) -> _mx_cy.double:
        if {lower} <= t < {stop}:
            return self._f_value(t)
        else:
            raise IndexError(t)
'''
    layout = DenseEntry('Policy', '_f_value', (DenseAxis('t', lower, stop, 'long long'),), False,
                        guard=cst.parse_expression(f'{lower} <= t < {stop}'))
    policy = CompilerPolicy.resolve({'automatic_optimizations': False, 'dense_one_axis': True})
    code = lower_entries({'m': EntryModule(cst.parse_module(source))},
                         {'m': {'entry': layout}}, {}, policy)['m'].syntax.code
    line, = (line.strip() for line in code.splitlines() if '_mx_cy.cast' in line)
    return line.removeprefix('if ').removesuffix(':')


@pytest.mark.parametrize('lower,stop,expected', [
    (0, 4, '_mx_cy.cast(_mx_cy.ulonglong, t) < 4'),
    (-9, 6, '_mx_cy.cast(_mx_cy.ulonglong, t - (-9)) < 15'),
    (3, 8, '_mx_cy.cast(_mx_cy.ulonglong, t - (3)) < 5'),
])
def test_range_check_subtracts_signed_before_the_cast(lower, stop, expected):
    assert _check_text(lower, stop) == expected


@pytest.mark.parametrize('lower,stop', [(-9, 6), (0, 4), (3, 8)])
def test_range_check_compiled_with_overflowcheck(lower, stop):
    cython = pytest.importorskip('cython')
    check = _check_text(lower, stop).replace('_mx_cy.', 'cython.')
    body = ('cdef long long t\nresult = []\nfor t in range(%d, %d):\n'
            '    result.append(bool(%s))\nreturn result\n') % (lower - 4, stop + 4, check)
    got = cython.inline(body, cython_compiler_directives={'overflowcheck': True}, quiet=True)
    assert got == [lower <= t < stop for t in range(lower - 4, stop + 4)]
