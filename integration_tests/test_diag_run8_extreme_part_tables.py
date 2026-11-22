"""diag-run8 F1 on a compiled model: ``T.index.max()[k]`` / ``min()[k]`` read as a C integer keep pandas'
meaning, the k-th part of the lexicographically largest (smallest) key, not the extreme of level k
(entries (60, 5), (95, 1), (95, 3), (80, 10): ``max()[1]`` is 3, ``min()[1]`` is 5). A new model whose input holds
other labels reads them. The exported pure-Python model behaves the same (asserted on both)."""

import subprocess
import sys
import textwrap

MODEL = """\
import modelx as mx
import pandas as pd

KEYS = [(60, 5), (95, 1), (95, 3), (80, 10)]
RAW = pd.DataFrame({"q": [0.1, 0.2, 0.3, 0.4]}, index=pd.MultiIndex.from_tuples(KEYS, names=["age", "dur"]))
IDS = (1, 2)


def build():
    model = mx.new_model("Extremes")
    data = model.new_space("Data", refs={"pd": pd, "RAW": RAW})
    data.new_cells("tbl", "def tbl():\\n    return RAW.copy()\\n")
    proj = model.new_space("Projection", formula="lambda policy_id: None",
                           refs={"data": data, "ids": IDS},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"v": v()}}')
    proj.new_cells("top1", "lambda: data.tbl().index.max()[1]")
    proj.new_cells("low1", "lambda: data.tbl().index.min()[1]")
    proj.new_cells("top0", "lambda: data.tbl().index.max()[0]")
    proj.new_cells("v", "lambda: float(100 * top1() + 10 * low1() + top0() + policy_id)")
    return model
"""

CHECKS = """\
import sys
from importlib import import_module
import pandas as pd
from extremes_model import RAW, build

package = import_module(sys.argv[1])
Model = type(package.mx_model)


def values(m):
    p = m.Projection[1]
    return p.top1(), p.low1(), p.top0(), p.v()


def pandas_values(frame):
    ix = frame.index
    return ix.max()[1], ix.min()[1], ix.max()[0]


m = Model()
first = values(m)
assert first[:3] == pandas_values(RAW) == (3, 5, 95), first          # lexicographic, not per level
assert first[:3] != (RAW.index.get_level_values(1).max(), RAW.index.get_level_values(1).min(), 95)
edited = pd.MultiIndex.from_tuples([(60, 5), (95, 1), (99, 7), (80, 10), ], names=["age", "dur"])
m = Model()                                                        # a new model whose input holds other labels
m.Data.RAW = RAW.set_axis(edited)
after = values(m)
assert after[:3] == pandas_values(m.Data.tbl()) == (7, 5, 99), after
print("ok", sys.argv[1])
"""


def test_extreme_part_is_lexicographic_and_cleared_with_the_space(tmp_path):
    (tmp_path / "extremes_model.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / "build_model.py").write_text(textwrap.dedent("""\
        from extremes_model import build
        m = build()
        m.export("extremes_nomx", locked_spaces=["Data"])
        m.close()
        """), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from extremes_nomx import mx_model\nfor k in (1, 2):\n    mx_model.Projection[k].collect_outputs()\n"
        "mx_model.Projection.collect()\n", encoding="utf-8")
    (tmp_path / "spec.py").write_text(
        repr({"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}), encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600)
    run("build_model.py")
    run("-m", "modelx_cython", "extremes_nomx", "--sample", "sample.py", "--spec", "spec.py")
    generated = (tmp_path / "extremes_nomx_cy/_mx_classes.py").read_text(encoding="utf-8")
    # T-5c (prepared tables): the int parts of the extremes, kept as C integers when the table is prepared
    assert "max_part_ll(" in generated and "min_part_ll(" in generated, generated[-4000:]
    assert "self.data._mx_prepared_tbl()" in generated
    # the lowered read takes no reference on a shared object (the regression diag-run8 first found by timing)
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from _diag_run8_c import refcount_operations
    c_text = (tmp_path / "extremes_nomx_cy" / "_mx_classes.c").read_text(encoding="utf-8")
    for function in ("max_part_ll", "min_part_ll"):
        assert refcount_operations(c_text, function) == [], function
    run("checks.py", "extremes_nomx_cy")
    run("checks.py", "extremes_nomx")
