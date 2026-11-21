"""diag-run8 F1, more forms, compiled against the exported model and pandas: ``index.max()[k]`` /
``min()[k]`` on a complete product of two integer levels (k = 1), on an incomplete one (the other storage
form), and on a table emptied in place (pandas: NaN, so ``[k]`` raises TypeError). Values and errors equal live."""

import subprocess
import sys
import textwrap

MODEL = """\
import modelx as mx
import pandas as pd

PRODUCT = pd.DataFrame({"q": [float(i) for i in range(6)]},
                       index=pd.MultiIndex.from_product([[1, 2], [10, 20, 30]], names=["a", "b"]))
SPARSE = pd.DataFrame({"q": [0.1, 0.2, 0.3, 0.4]},
                      index=pd.MultiIndex.from_tuples([(60, 5), (95, 1), (95, 3), (80, 10)], names=["a", "b"]))
SINGLE = pd.DataFrame({"q": [0.1, 0.2]}, index=pd.Index([3, 7], name="a"))
IDS = (1, 2)
SINGLE_CELL = False


def build():
    model = mx.new_model("ExtremeForms")
    data = model.new_space("Data", refs={"PRODUCT": PRODUCT, "SPARSE": SPARSE, "SINGLE": SINGLE})
    data.new_cells("product", "def product():\\n    return PRODUCT.copy()\\n")
    data.new_cells("sparse", "def sparse():\\n    return SPARSE.copy()\\n")
    data.new_cells("single", "def single():\\n    return SINGLE.copy()\\n")
    proj = model.new_space("Projection", formula="lambda policy_id: None", refs={"data": data, "ids": IDS},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"v": v()}}')
    proj.new_cells("p_top1", "lambda: data.product().index.max()[1]")
    proj.new_cells("p_low1", "lambda: data.product().index.min()[1]")
    proj.new_cells("s_top1", "lambda: data.sparse().index.max()[1]")
    proj.new_cells("s_low0", "lambda: data.sparse().index.min()[0]")
    if SINGLE_CELL:
        proj.new_cells("one_top0", "lambda: data.single().index.max()[0]")
    proj.new_cells("v", "lambda: float(p_top1() + p_low1() + s_top1() + s_low0() + policy_id)")
    return model
"""

CHECKS = """\
import sys
from importlib import import_module
from extremes_forms import PRODUCT, SPARSE, SINGLE

package = import_module(sys.argv[1])
Model = type(package.mx_model)
NAMES = ("p_top1", "p_low1", "s_top1", "s_low0")


def outcome(f):
    try:
        return ("value", int(f()))
    except Exception as e:
        return ("raise", type(e).__name__, str(e))


def pandas(frame, part, last):
    ix = frame.index
    return outcome(lambda: (ix.max() if last else ix.min())[part])


m = package.mx_model
p = m.Projection[1]
got = [outcome(getattr(p, n)) for n in NAMES]
want = [pandas(PRODUCT, 1, True), pandas(PRODUCT, 1, False), pandas(SPARSE, 1, True), pandas(SPARSE, 0, False)]
assert got == want, (got, want)
assert got[2] == ("value", 3), got
m = Model()                                              # a new model whose inputs are empty
m.Data.PRODUCT = PRODUCT.iloc[:0]
m.Data.SPARSE = SPARSE.iloc[:0]
p = m.Projection[1]
empty = [outcome(getattr(p, n)) for n in NAMES[:4]]
want = [pandas(PRODUCT.iloc[:0], 1, True), pandas(PRODUCT.iloc[:0], 1, False),
        pandas(SPARSE.iloc[:0], 1, True), pandas(SPARSE.iloc[:0], 0, False)]
assert empty == want, (empty, want)
print(repr(got + empty))
"""


def test_extreme_part_forms_equal_live(tmp_path):
    (tmp_path / "extremes_forms.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / "build_model.py").write_text(textwrap.dedent("""\
        from extremes_forms import build
        m = build()
        m.export("forms_nomx", locked_spaces=["Data"])
        m.close()
        """), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from forms_nomx import mx_model\np = mx_model.Projection\nfor k in (1, 2):\n"
        "    p[k].collect_outputs()\n"
        "mx_model.Projection.collect()\n", encoding="utf-8")
    (tmp_path / "spec.py").write_text(
        repr({"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}), encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600,
                                    capture_output=True, text=True)
    run("build_model.py")
    run("-m", "modelx_cython", "forms_nomx", "--sample", "sample.py", "--spec", "spec.py")
    generated = (tmp_path / "forms_nomx_cy/_mx_classes.py").read_text(encoding="utf-8")
    assert generated.count("_part_ll(") >= 4, generated.count("_part_ll(")      # T-5c: prepared-table parts
    outs = {pkg: run("checks.py", pkg).stdout for pkg in ("forms_nomx_cy", "forms_nomx")}
    assert outs["forms_nomx_cy"] == outs["forms_nomx"], outs
