"""Numeric widening preserves values; declared object results retain classes."""
import numbers

import numpy as np
import pytest

from modelx_cython.monkeytype_tracing import CallTrace
from modelx_cython.tracer import ReturnTypeInfo, RuntimeCellsInfo


def formula(self, n):
    pass


def merged(values, function=formula):
    return RuntimeCellsInfo([CallTrace(function, {'self': None, 'n': n}, ret_val=value)
                             for n, value in enumerate(values)]).ret_type


@pytest.mark.parametrize('values', [[0, 1.5], [1.5, 0], [0, np.float64(1.5)],
                                  [np.float64(1.5), 0], [0, 1.5, 0, 2.5]])
def test_scalar_integral_nonintegral_union_uses_upstream_numeric_abi(values):
    assert merged(values) == ReturnTypeInfo(numbers.Real)


@pytest.mark.parametrize('integer', [2**53+1, -(2**53+1), np.int64(2**53+1)])
def test_numeric_widening_declines_inexact_integer_values_in_either_order(integer):
    for values in ([integer, 1.5], [1.5, integer], [0, 1.5, integer]):
        assert merged(values) == ReturnTypeInfo(object)


def test_exact_large_integer_can_widen_without_an_arbitrary_size_cutoff():
    assert merged([2**60, 1.5]) == ReturnTypeInfo(numbers.Real)


def test_concrete_object_declaration_keeps_empty_sum_scalar_class():
    def raw(self, n) -> object:
        pass
    assert merged([0, 1.5], raw) == ReturnTypeInfo(object)


def test_homogeneous_float_and_integral_scalar_contracts_are_unchanged():
    assert merged([0.0, 1.5]) == ReturnTypeInfo(float)
    assert merged([0, 1]) == ReturnTypeInfo(int)
    assert merged([0, np.int64(1)]) == ReturnTypeInfo(numbers.Integral)


def test_array_numeric_promotion_is_unchanged():
    result = merged([np.array([0], dtype=np.int64), np.array([1.5], dtype=float)])
    assert result == ReturnTypeInfo(numbers.Real, is_array=True, ndim=1)
