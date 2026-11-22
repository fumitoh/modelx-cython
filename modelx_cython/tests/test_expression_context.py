import ast
import libcst as cst
import pytest

from modelx_cython.scopes import CstFacts
from modelx_cython.syntax import code


@pytest.mark.parametrize('source', [
    'call(\n    left  # continued expression\n    <= right\n)',
    'call(\n    left % divisor\n    == 0  # comment before the call closes\n)',
    'call(\n    left\n    * (right - 1)\n)',
])
def test_detached_call_arguments_keep_their_implicit_context(source):
    node = cst.parse_expression(source).args[0].value
    expected = ast.parse(source, mode='eval').body.args[0]
    assert ast.dump(ast.parse('(\n'+code(node)+'\n)', mode='eval').body) == ast.dump(expected)
    assert CstFacts(node).reads
