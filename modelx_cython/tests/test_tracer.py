import logging

import numpy as np
import pytest

from modelx_cython.tracer import ReturnTypeInfo, RuntimeCellsInfo
from modelx_cython.monkeytype_tracing import CallTrace


def foo(self, i):
    pass


def make_trace(ret_val, i):
    return CallTrace(foo, {"self": None, "i": i}, ret_val=ret_val)


@pytest.mark.parametrize("ret_vals", [
    [np.array([1.0, 2.0]), 3],
    [3, np.array([1.0, 2.0])],
    [np.array([1.0, 2.0]), 3, np.array([1.0, 2.0])]
])
def test_ret_type_array_and_non_array(ret_vals, caplog):
    """Array and non-array values returned from the same cells collapse to object"""
    traces = [make_trace(val, i) for i, val in enumerate(ret_vals)]
    with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
        info = RuntimeCellsInfo(traces)

    assert info.ret_type == ReturnTypeInfo(object)
    msgs = [r.message for r in caplog.records
            if "varying array and non-array types returned from" in r.message]
    assert len(msgs) == 1
    assert "array of float64" in msgs[0]
    assert "int" in msgs[0]


def test_ret_type_varying_non_array_types(caplog):
    """Both conflicting types appear in the log for non-array return values"""
    traces = [make_trace(1, 0), make_trace("abc", 1)]
    with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
        info = RuntimeCellsInfo(traces)

    assert info.ret_type == ReturnTypeInfo(object)
    msgs = [r.message for r in caplog.records
            if "varying types returned from" in r.message]
    assert len(msgs) == 1
    assert "int for i=0" in msgs[0]
    assert "str for i=1" in msgs[0]
