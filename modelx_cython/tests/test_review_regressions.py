"""Regression tests added by review of the bondportfolio/mx2cy-improvements
changes.  Each test demonstrates a defect and fails on the reviewed code."""
import logging

import numpy as np

from modelx_cython.tracer import MxCallTraceLogger, RuntimeCellsInfo
from modelx_cython.monkeytype_tracing import CallTrace


def foo(self, i):
    pass


FOO = foo.__module__ + ".foo"


def make_trace(ret_val, i):
    return CallTrace(foo, {"self": None, "i": i}, ret_val=ret_val)


def test_dedup_keeps_varying_return_messages(caplog):
    """MxCallTraceLogger.log claims what RuntimeCellsInfo derives from the
    kept traces, including its log messages, equals what it derives from
    all traces.  A scalar return seen again after an array return, with a
    smaller integer argument, is dropped, and with it the "varying types
    returned" message that the full trace list emits."""
    traces = [make_trace(1.0, 2), make_trace(np.array([1.0]), 1),
              make_trace(1.0, 0)]
    logger = MxCallTraceLogger(module="M")
    for tr in traces:
        logger.log(tr)
    kept = logger._traces[FOO]

    def messages(trs):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
            info = RuntimeCellsInfo(trs)
        return info.ret_type, [r.message for r in caplog.records]

    full_ret, full_msgs = messages(traces)
    kept_ret, kept_msgs = messages(kept)
    assert full_ret == kept_ret
    assert full_msgs == kept_msgs, (full_msgs, kept_msgs)


UNCACHED_SAMPLE = """
from RevDu_nomx import mx_model as m
m.S.g(1.0, 2.0)
m.S.h(2, 3)
m.S.r(5)
"""


def test_uncached_cells_typing_keeps_compilable_signatures(tmp_path):
    """Typing an uncached cells must not make a model that compiled before
    fail to compile.  On main these three compiled as object; now:

    * ``g(x, y=None)`` traced with floats: ``y: double = None`` ->
      "Signature not compatible with previous declaration";
    * ``h(x, k=0.5)`` traced with ints: ``k: long long = 0.5`` ->
      "Cannot assign type 'double' to 'long long'";
    * ``r(rate)`` with ``rate = rate / 100`` in the body, traced with an
      int: ``rate: long long`` -> "Cannot assign type 'double' to
      'long long'".

    (Cached cells have the same problem in their ``_f_`` formula; the
    change extends it to every uncached cells the sample reaches.)
    """
    import os
    import subprocess
    import sys
    import modelx as mx

    m = mx.new_model("RevDu")
    s = m.new_space("S")

    def g(x, y=None):
        return x if y is None else x * y

    def h(x, k=0.5):
        return x * k

    def r(rate):
        rate = rate / 100
        return rate

    for f in (g, h, r):
        s.new_cells(f.__name__, f, is_cached=False)
    m.export(tmp_path / "RevDu_nomx")
    m.close()
    (tmp_path / "sample.py").write_text(UNCACHED_SAMPLE)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "modelx_cython", str(tmp_path / "RevDu_nomx"),
         "--sample", str(tmp_path / "sample.py"), "--no-spec",
         "--translate-only"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    src = (tmp_path / "RevDu_nomx_cy" / "_mx_classes.py").read_text(
        encoding="utf-8")
    assert "y: _mx_cy.double=None" not in src
    assert "k: _mx_cy.longlong=0.5" not in src
    assert "def r(self, rate: _mx_cy.longlong)" not in src
