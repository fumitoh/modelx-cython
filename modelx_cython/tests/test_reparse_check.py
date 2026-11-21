"""The emitted module text must parse back to the grouping of the compiled tree."""
import libcst as cst
import pytest

from modelx_cython.reparse_check import PrecedenceError, check

VALID = '''\
import cython as _mx_cy

@_mx_cy.cfunc
def f(self, a, x, *args, k=(1 or 2), **kw) -> object:
    y = (z := a + 1) * 2 if x else -(a ** -x)
    w = [i for i in args if not i in (1, 2) and i is not None]
    v = {k: (lambda q=(a or x): q) for k in kw}
    s = args[(a + 1):a or 2]
    t = f"{a and x!r:>{(x or 3)}}"
    del w[a - 1]
    return (yield a) if a < x <= y else not (a or x) and y


async def g(a):
    return -(await a) + 1
'''


def test_valid_code_passes_unchanged():
    module = cst.parse_module(VALID)
    assert check(module) == VALID


class _DropParens(cst.CSTTransformer):
    """A faulty lowering: ``x in (p, q)`` -> ``p == x or q == x`` spliced without parentheses."""

    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def leave_Comparison(self, original, updated):
        if not isinstance(updated.comparisons[0].operator, cst.In):
            return updated
        chain = cst.parse_expression('(p == x or q == x)')
        return chain if self.keep else chain.with_changes(lpar=(), rpar=())


@pytest.mark.parametrize('expression', [
    'a and x in (p, q)',          # and operand
    'not x in (p, q) == b',       # comparison operand ... inside not
    '(x in (p, q)) + 1',          # arithmetic operand, outer parentheses dropped too
    '-(x in (p, q))',
    '(x in (p, q)).real',
])
def test_dropped_parentheses_stop_the_build(expression):
    module = cst.parse_module(f'def cell(a, b, x, p, q):\n    return {expression}\n')
    faulty = module.visit(_DropParens(keep=False))
    with pytest.raises(PrecedenceError, match=r'_mx_classes.py: function cell: .*p == x'):
        check(faulty, name='_mx_classes.py')
    # the same lowering that keeps its parentheses passes
    check(module.visit(_DropParens(keep=True)))


def test_an_if_test_needs_no_parentheses():
    module = cst.parse_module('def cell(x, p, q):\n    if x in (p, q):\n        return 1\n')
    check(module.visit(_DropParens(keep=False)))


def test_explicit_text_is_compared_with_the_tree():
    module = cst.parse_module('x = a + b * c\n')
    with pytest.raises(PrecedenceError, match='module level'):
        check(module, text='x = (a + b) * c\n')


def test_nested_boolean_operations_of_one_operator_are_one_chain():
    # libcst nests `a or b or c`; with every operator node in parentheses ast would nest it too
    module = cst.parse_module('def cell(a, b, c, d):\n    return d > 2 or a or b or c and (a or b) or c\n')
    check(module)
