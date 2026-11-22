"""The specification reader keeps its literal-only contract."""
import ast

import libcst as cst
import pytest

from modelx_cython.syntax import read_literal


@pytest.mark.parametrize('source', [
    "{'spaces': {'Values': {'cells_param_size': {('t','k'): (25,7)}}}}",
    'None', 'True', 'False', '...', '0xff', '-9223372036854775809', '+2', '1e309',
    '-0.0', '3.5', '2j', '-2j', '1+2j', '-1-2j', '(1,)', '()', '[1,2]', '{1,2}',
    'set()', "{b'bytes': r'raw\\value'}", "'adjacent' 'strings'", '# header\n {1:2}\n',
    "{'same': 1, 'same': 2}", "{'escaped': '\\u00e9\\n'}",
])
def test_specification_literals_match_the_original_reader(source):
    try:
        expected = ast.literal_eval(source)
    except (ValueError, SyntaxError, IndentationError):
        with pytest.raises((ValueError, TypeError, cst.ParserSyntaxError)):
            read_literal(source)
    else:
        actual = read_literal(source)
        assert type(actual) is type(expected)
        assert actual == expected


@pytest.mark.parametrize('source', [
    'dict(a=1)', "{'a': len([])}", "{**{}}", '[x for x in ()]', 'f"{1}"', '1+2',
    '1+-2j', '--1', 'True+2j', 'x', '1;', '1; 2', 'lambda: 1', "f'{1}' 'text'",
])
def test_specification_reader_does_not_execute_expressions(source):
    with pytest.raises((ValueError, TypeError, cst.ParserSyntaxError)):
        read_literal(source)
