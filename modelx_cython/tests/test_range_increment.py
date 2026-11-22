import libcst as cst
import pytest

from modelx_cython.reductions import safe_range_increment


@pytest.mark.parametrize('arguments,safe', [
    ('self.low, self.high', True), ('self.low, self.high, 1', True),
    ('self.low, self.high, -1', True), ('self.low, self.high, 2', False),
    ('0, 10, self.step', False), ('0, 10, 0', False), ('10, 0, 2', True),
    (f'{(1<<63)-2}, {(1<<63)-1}, 2', False),
    (f'{-(1<<63)+1}, {-(1<<63)}, -2', False),
    (f'{-(1<<63)}, {(1<<63)-1}, 2', False),
    (f'{(1<<63)-3}, {(1<<63)-1}, 2', True),
])
def test_native_range_final_increment_is_representable(arguments, safe):
    iterator = cst.parse_expression('range('+arguments+')')
    assert safe_range_increment(iterator) is safe
