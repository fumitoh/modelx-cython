"""W1.5 (D25 Q7): typed locals with overflow checks are the default."""
import pytest

from modelx_cython.config import CompilerPolicy


def test_infer_types_is_on_by_default():
    assert CompilerPolicy.resolve({}).infer_types is True
    assert CompilerPolicy.resolve({'automatic_optimizations': False}).infer_types is True


def test_infer_types_can_be_turned_off():
    assert CompilerPolicy.resolve({'infer_types': False}).infer_types is False


@pytest.mark.parametrize('value', ['no', 0])
def test_infer_types_must_be_boolean(value):
    with pytest.raises(ValueError, match='infer_types'):
        CompilerPolicy.resolve({'infer_types': value})
