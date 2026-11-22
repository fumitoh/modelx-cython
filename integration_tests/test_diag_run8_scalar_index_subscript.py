"""diag-run8: a subscript of an index read that is a scalar in pandas (``T.index.max()[0]`` on one integer
level: numpy int, so ``[0]`` raises IndexError) keeps the read as written, so the compiled model gives live's
error type (both raise; the type differs, see below); the plain ``T.index.max()`` beside it is still a C integer read. Compiled = exported = pandas.
The public row of an all-numeric row cell equals live's pandas row (ints upcast to float)."""

import subprocess
import sys
import textwrap

import pytest

MODEL = """\
import modelx as mx
import pandas as pd

SINGLE = pd.DataFrame({"q": [0.1, 0.2]}, index=pd.Index([3, 7], name="a"))
POINTS = pd.DataFrame({"sa": [100.0, 200.0], "age": [30, 40]}, index=pd.Index([1, 2], name="policy_id"))
IDS = (1, 2)


def build():
    model = mx.new_model("ScalarIndex")
    data = model.new_space("Data", refs={"SINGLE": SINGLE, "POINTS": POINTS})
    data.new_cells("single", "def single():\\n    return SINGLE.copy()\\n")
    data.new_cells("points", "def points():\\n    return POINTS.copy()\\n")
    proj = model.new_space("Projection", formula="lambda policy_id: None", refs={"data": data, "ids": IDS},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"v": v()}}')
    proj.new_cells("top", "lambda: data.single().index.max()")
    proj.new_cells("top0", "lambda: data.single().index.max()[0]")
    proj.new_cells("model_point", "lambda: data.points().loc[policy_id]")
    proj.new_cells("v", 'lambda: float(top() + policy_id) + float(model_point()["sa"])')
    return model
"""

CHECKS = """\
import sys
from importlib import import_module
from scalar_model import SINGLE
p = import_module(sys.argv[1]).mx_model.Projection[1]


def outcome(f):
    try:
        return ("value", int(f()))
    except Exception as e:
        return ("raise", type(e).__name__, str(e))


got = [outcome(p.top), outcome(p.top0)]
want = [outcome(SINGLE.index.max), outcome(lambda: SINGLE.index.max()[0])]
assert got[0] == want[0] and got[1][0] == want[1][0] == "raise", (got, want)
print(repr(got[0]), "|", repr(got[1][1]), "|", repr(sorted((str(k), repr(v)) for k, v in p.model_point().items())))
"""


def _build(tmp_path):
    (tmp_path / "scalar_model.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / "build_model.py").write_text(textwrap.dedent("""\
        from scalar_model import build
        m = build()
        m.export("scalar_nomx", locked_spaces=["Data"])
        m.close()
        """), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from scalar_nomx import mx_model\np = mx_model.Projection\nfor k in (1, 2):\n"
        "    p[k].collect_outputs()\n    p[k].model_point()\n    try:\n        p[k].top0()\n"
        "    except (IndexError, TypeError):\n        pass\nmx_model.Projection.collect()\n", encoding="utf-8")
    (tmp_path / "spec.py").write_text(
        repr({"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}), encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600,
                                    capture_output=True, text=True)
    run("build_model.py")
    run("-m", "modelx_cython", "scalar_nomx", "--sample", "sample.py", "--spec", "spec.py")
    generated = (tmp_path / "scalar_nomx_cy/_mx_classes.py").read_text(encoding="utf-8")
    return {pkg: run("checks.py", pkg).stdout.strip().split(" | ") for pkg in ("scalar_nomx_cy", "scalar_nomx")}, generated


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("scalar"))


def test_subscript_of_a_scalar_index_read_keeps_the_python_path(built):
    outs, generated = built
    assert outs["scalar_nomx_cy"][0] == outs["scalar_nomx"][0], outs     # the plain extreme, a C integer read
    # both raise at run time (asserted in checks.py); T-5c: the subscript of the C read takes pandas' value, so
    # the error is live's (IndexError); it was TypeError on the helper layer (an open difference)
    assert outs["scalar_nomx_cy"][1] == outs["scalar_nomx"][1] == "'IndexError'", outs
    assert "max_ll(" in generated and ".index.max())[0]" in generated, generated[-3000:]


# T-5c: was an open difference on the helper layer (TypeError compiled, IndexError live); now equal
def test_scalar_index_subscript_error_type_equals_live(built):
    outs, _ = built
    assert outs["scalar_nomx_cy"][1] == outs["scalar_nomx"][1], outs


def test_all_numeric_row_from_python_equals_live(built):
    outs, _ = built         # a plain Space: the public row is the pandas row, ints upcast as live
    assert outs["scalar_nomx_cy"][2] == outs["scalar_nomx"][2], outs
