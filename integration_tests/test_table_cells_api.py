"""Python access to compiled table cells and References (D101a), as upstream modelx-cython.

A table cell returns the plain DataFrame its formula returned, the same object on every call. References are
plain attributes, assignable as upstream allows (values cached before an assignment stay; the drivers bind
inputs on a new model before running it). A compiled model has no clear and no table-cell assignment. The
exported pure-Python model behaves the same for what both offer (asserted on both)."""

import subprocess
import sys
import textwrap


MODEL = """\
import modelx as mx
import pandas as pd
from modelx_helpers import pdn

FRAME = pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0], "rate": [0.01, 0.02, 0.03, 0.04]},
                     index=pd.Index([1, 2, 3, 4], name="policy_id"))
IDS = (1, 2, 3, 4)


def build():
    model = mx.new_model("Tables")
    data = model.new_space("Data", refs={"pd": pd, "pdn": pdn})
    data.raw = FRAME
    data.new_cells("points", "def points():\\n    return raw.copy()\\n")       # plain pandas (D121)
    data.new_cells("table", "def table():\\n    return FRAME2.copy()\\n")
    data.FRAME2 = FRAME
    proj = model.new_space("Projection", formula="lambda policy_id: None",
                           refs={"data": data, "ids": IDS, "factor": 1.0},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"pv": pv()}}')
    proj.new_cells("sa", 'lambda: data.points().loc[policy_id, "sa"]')
    proj.new_cells("rate", 'lambda: data.table().loc[policy_id, "rate"]')
    proj.new_cells("pv", "lambda: factor * sum(sa() * (1.0 + rate()) ** t for t in range(12))")
    return model
"""

CHECKS = """\
import sys, threading
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
import pandas as pd
from modelx_helpers import pdn
from tables_model import FRAME, build

name = sys.argv[1]
compiled = name.endswith("_cy")
package = import_module(name)
if compiled:
    assert isinstance(import_module(name + "._mx_classes").__loader__, ExtensionFileLoader)
    runtime = import_module("modelx_cython.runtime.table_cells")
Model = type(package.mx_model)


def total(m, workers):
    m.collect_workers = workers
    return m.Projection.collect().sum.pv


def direct(m):
    return sum(m.Projection[i].pv() for i in (1, 2, 3, 4))


def live(raw=None, factor=None):
    m = build()
    try:
        if raw is not None:
            m.Data.raw = raw
        if factor is not None:
            m.Projection.factor = factor
        return sum(m.Projection[i].pv() for i in (1, 2, 3, 4))
    finally:
        m.close()


def close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


base = live()
doubled = FRAME.assign(sa=FRAME["sa"] * 2)

# (1) a table cell returns a plain DataFrame, the same object every time; both routes equal live
m = Model()
v = m.Data.points()
assert type(v) is pd.DataFrame and v is m.Data.points() and v.equals(FRAME), type(v)
assert close(direct(m), base) and close(total(m, 1), base) and close(total(m, 4), base)

# (2) References assigned on a new model before it runs: a Reference holding a table, and a scalar one
m = Model()
m.Data.raw = doubled
assert m.Data.raw.equals(doubled)
assert close(direct(m), live(raw=doubled)) and close(total(m, 4), live(raw=doubled))
m = Model()
m.Projection.factor = 3.0
assert close(direct(m), live(factor=3.0)) and close(total(m, 4), live(factor=3.0))

# (3) one ItemSpace is deleted and created again on the next access, as before (D114)
m = Model()
item = m.Projection[3]
del m.Projection[3]
assert m.Projection[3] is not item and close(direct(m), base)

if compiled:
    # (4) as upstream: no clear of a Space and no table-cell assignment on a compiled model
    for space in (m.Data, m.Projection):
        for method in ("clear_all", "clear_cells", "clear_items", "clear_at"):
            assert method not in vars(type(space)), (type(space).__name__, method)
    import operator
    for attempt, error in ((lambda: operator.setitem(m.Data.points, (), doubled), TypeError),
                           (lambda: setattr(m.Data, "points", doubled), AttributeError)):
        try:
            attempt()
        except error:
            pass
        else:
            raise AssertionError("a table cell was assigned")
print("ok", name)
"""


def test_table_cells_and_references(tmp_path):
    (tmp_path / "tables_model.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / "build_model.py").write_text(textwrap.dedent("""\
        from tables_model import build
        m = build()
        m.export("tables_nomx", locked_spaces=["Data"])
        m.close()
        """), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from tables_nomx import mx_model\nfor k in (1, 2, 3, 4):\n    mx_model.Projection[k].collect_outputs()\n"
        "mx_model.Projection.collect()\n", encoding="utf-8")
    (tmp_path / "spec.py").write_text(
        repr({"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}), encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600)
    run("build_model.py")
    run("-m", "modelx_cython", "tables_nomx", "--sample", "sample.py", "--spec", "spec.py")
    generated = (tmp_path / "tables_nomx_cy/_mx_classes.py").read_text(encoding="utf-8")
    pxd = (tmp_path / "tables_nomx_cy/_mx_classes.pxd").read_text(encoding="utf-8")
    assert "def _mx_cell_points(self)" in generated and "def _mx_cell_table(self)" in generated
    assert "cdef public object raw" in pxd and "cdef public object ids" in pxd      # References: plain attributes
    assert "_mx_ref_" not in generated and "collect_begin" not in generated
    run("checks.py", "tables_nomx_cy")
    run("checks.py", "tables_nomx")         # the exported pure-Python model behaves the same (D130)
