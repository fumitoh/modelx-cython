"""Regression tests added by review of the bondportfolio/mx2cy-improvements
changes.  Each test demonstrates a defect and fails on the reviewed code."""
import logging
from fractions import Fraction

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


def test_float32_and_float_args_are_not_typed_double():
    """{float, numpy.float32} is widened to double, but float32 arithmetic
    differs from double arithmetic: the compiled cells returns
    0.30000000447034836 for x * 3 where the exported model returns
    float32(0.3).  The same mixture used to be typed object, which kept
    the Python result."""
    info = RuntimeCellsInfo([make_trace(1.0, 1.5),
                             make_trace(1.0, np.float32(0.1))])
    assert info.arg_types["i"] is object


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


def _rewrite_powers_then_math(body):
    """Both passes, in the order ModuleTransformer._rewrite_powers runs
    them, with use_libm on: ``x`` is a C double parameter."""
    import libcst as cst
    from modelx_cython.powers import rewrite_math_calls, rewrite_powers
    from modelx_cython.usage import CellsResolver

    cls = "Model_nomx._mx_classes._c_Space1"

    class Cells:
        fqname = formula_fqname = cls + "._f_target"
        params = ("x",)

        def has_args(self):
            return True

        def has_typeinfo(self):
            return True

        def get_argtype_expr(self, arg, c_style=False, formula=False):
            return "double"

    resolver = CellsResolver(cells_by_class={cls: {}},
                             refs_by_class={cls: {"math": ""}},
                             spaces_by_class={cls: {}})
    node = cst.parse_statement(f"def _f_target(self, x):\n    {body}\n")
    # as ModuleTransformer._rewrite_powers calls it under use_libm
    node = rewrite_powers(node, resolver, {}, {cls: {}}, cls, Cells(),
                          math_refs={cls: {"math"}})
    node = rewrite_math_calls(node, resolver, {}, {cls: {}},
                              {cls: {"math"}}, cls, Cells())
    return cst.Module([]).code_for_node(node)


def test_use_libm_power_of_math_call_is_real():
    """With use_libm, ``self.math.exp(x)`` becomes a C double, so a ``**``
    on it that is left alone goes on Cython's complex path: the model
    ``max(math.exp(x) ** 0.5, 1.0)`` compiles without use_libm and fails
    with "complex types are unordered" with it.  The power has to be
    rewritten as well (or the math call left alone)."""
    out = _rewrite_powers_then_math("return max(self.math.exp(x) ** 0.5, 1.0)")
    assert ("_mx_sys._mx_pow(_mx_sys._mx_exp(x), 0.5)" in out
            or "self.math.exp(x) ** 0.5" in out), out


def test_use_libm_math_call_on_rewritten_power():
    """``math.exp(x ** 2.0)``: the power is rewritten first, and the math
    pass then sees ``_mx_sys._mx_pow(...)`` as an unknown operand, leaves
    the exp a Python call and logs that its argument is not provably a C
    number."""
    out = _rewrite_powers_then_math("return self.math.exp(x ** 2.0)")
    assert "_mx_sys._mx_exp(_mx_sys._mx_pow(x, 2.0))" in out, out


def test_fraction_and_float_args_are_not_typed_double():
    """{float, Fraction} is widened to double: Fraction is numbers.Real
    but not exactly representable, so the compiled cells loses the exact
    rational arithmetic the exported model performs."""
    info = RuntimeCellsInfo([make_trace(1.0, 1.5),
                             make_trace(1.0, Fraction(1, 3))])
    assert info.arg_types["i"] is object
