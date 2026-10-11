"""Regression tests added by review of the bondportfolio/mx2cy-improvements
changes.  Each test demonstrates a defect and fails on the reviewed code."""
from fractions import Fraction

import numpy as np

from modelx_cython.tracer import RuntimeCellsInfo
from modelx_cython.monkeytype_tracing import CallTrace


def foo(self, i):
    pass


def make_trace(ret_val, i):
    return CallTrace(foo, {"self": None, "i": i}, ret_val=ret_val)


def test_float32_and_float_args_are_not_typed_double():
    """{float, numpy.float32} is widened to double, but float32 arithmetic
    differs from double arithmetic: the compiled cells returns
    0.30000000447034836 for x * 3 where the exported model returns
    float32(0.3).  The same mixture used to be typed object, which kept
    the Python result."""
    info = RuntimeCellsInfo([make_trace(1.0, 1.5),
                             make_trace(1.0, np.float32(0.1))])
    assert info.arg_types["i"] is object


def test_fraction_and_float_args_are_not_typed_double():
    """{float, Fraction} is widened to double: Fraction is numbers.Real
    but not exactly representable, so the compiled cells loses the exact
    rational arithmetic the exported model performs."""
    info = RuntimeCellsInfo([make_trace(1.0, 1.5),
                             make_trace(1.0, Fraction(1, 3))])
    assert info.arg_types["i"] is object
