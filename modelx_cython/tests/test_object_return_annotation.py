"""Explicit object results preserve the source-declared Cell boundary."""
import numpy as np
import pytest
from modelx_cython.monkeytype_tracing import CallTrace
from modelx_cython.tracer import ReturnTypeInfo, RuntimeCellsInfo


def result(function, value):
    return RuntimeCellsInfo([CallTrace(function, {'self': None, 'key': 0}, ret_val=value)]).ret_type


def annotated(self, key) -> object:
    pass


def ordinary(self, key):
    pass


def deferred_string(self, key) -> 'object':
    pass


@pytest.mark.parametrize('value', [0, 1., np.float64(.1), np.array([1., 2.]), object()])
def test_concrete_object_signature_controls_sampled_result(value):
    assert result(annotated, value) == ReturnTypeInfo(object)


def test_unannotated_and_string_annotations_keep_existing_inference():
    assert result(ordinary, 1.) == ReturnTypeInfo(float)
    assert result(deferred_string, 1.) == ReturnTypeInfo(float)


def test_arbitrary_string_annotation_is_never_evaluated():
    def formula(self, key):
        pass
    formula.__annotations__ = {'return': 'raise_if_evaluated()'}
    assert result(formula, 1.) == ReturnTypeInfo(float)
