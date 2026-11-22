"""diag-run8 F3 on a compiled model: fields of a row cell (``model_point = points().loc[policy_id]``) whose
every use is a field read are read from the table row by the readers, without building the row; the row
cell itself, called from Python, still gives its row. A key missing from the table is raised by the
reader with the type and message live gives. The row position is held per item
(an integer or a str key, looked up once per item, reset between policies). A model
where the row escapes (returned by a public Cell) keeps the Record path. Compiled equals the exported model, values and errors."""

import json
import subprocess
import sys
import textwrap

MODEL = """\
import enum
import modelx as mx
import pandas as pd


class Sex(enum.IntEnum):
    M = 0
    F = 1


POINTS = pd.DataFrame({"sa": [100.0, 200.0, 300.0], "age": [30, 40, 50], "sex": ["M", "F", "M"]},
                      index=pd.Index([1, 2, 3], name="policy_id"))
COSTS = pd.DataFrame({"fee": [1.5, 2.5]}, index=pd.Index(["A", "B"], name="cost_id"))
IDS = (1, 2, 3)


def build(escape=False):
    model = mx.new_model("Rows")
    data = model.new_space("Data", refs={"POINTS": POINTS, "COSTS": COSTS})
    data.new_cells("points", "def points():\\n    return POINTS.copy()\\n")
    data.new_cells("costs", "def costs():\\n    return COSTS.copy()\\n")
    proj = model.new_space("Projection", formula="lambda policy_id: None",
                           refs={"data": data, "ids": IDS, "Sex": Sex},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"pv": pv()}}')
    proj.new_cells("model_point", "lambda: data.points().loc[policy_id]")
    proj.new_cells("sa", 'lambda: float(model_point()["sa"])')
    proj.new_cells("age", 'lambda: int(model_point()["age"])')
    proj.new_cells("sex", 'lambda: Sex[model_point()["sex"]]')
    proj.new_cells("both", 'def both():\\n    mp = model_point()\\n    return mp["sa"] * mp["age"]\\n')
    proj.new_cells("cost_id", 'lambda: "A" if policy_id < 3 else "B"')
    proj.new_cells("cost_row", "lambda: data.costs().loc[cost_id()]")      # a str key, held per item
    proj.new_cells("fee", 'lambda: float(cost_row()["fee"])')
    proj.new_cells("pv", "lambda: sa() * (1.0 + 0.01 * age()) + 10.0 * sex() + both() + fee()")
    if escape:
        proj.new_cells("whole", "lambda: model_point()")
    return model
"""

CHECKS = """\
import sys
from importlib import import_module
package = import_module(sys.argv[1])
Model = type(package.mx_model)
p = package.mx_model.Projection
out = []
for i in (1, 2, 3):
    out.append([p[i].sa(), p[i].age(), int(p[i].sex()), float(p[i].both()), float(p[i].pv())])
    row = p[i].model_point()
    out.append(sorted((str(k), str(v)) for k, v in row.items()))
for call in ("sa", "pv"):
    try:
        getattr(p[99], call)()
        out.append(("value", call))
    except Exception as e:
        out.append(("raise", call, type(e).__name__, str(e)))
m = package.mx_model
for workers in (1, 4):                   # the held row is reset between policies, on both routes
    m.collect_workers = workers
    out.append(("collect", workers, round(float(m.Projection.collect().sum.pv), 9)))
# a new model whose input has another row value, and one whose input lacks a key: the direct route and a
# collect at 1 and 4 workers
edited = m.Data.points().copy()
edited.loc[1, "sa"] = 999.0
m = Model()
m.Data.POINTS = edited
out.append(("edited input", m.Projection[1].sa(), float(m.Projection[1].pv())))
m = Model()
m.Data.POINTS = edited.drop(2)
for call in ("sa", "pv"):
    try:
        getattr(m.Projection[2], call)()
        out.append(("value", "direct", call))
    except Exception as e:
        out.append(("raise", "direct", call, type(e).__name__, str(e)))
for workers in (1, 4):
    m.collect_workers = workers
    try:
        m.Projection.collect()
        out.append(("value", "collect", workers))
    except Exception as e:
        out.append(("raise", "collect", workers, type(e).__name__, str(e)))
print(repr(out))
"""


def _build(tmp_path, escape):
    name = "rows_esc" if escape else "rows"
    (tmp_path / "rows_model.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / f"build_{name}.py").write_text(textwrap.dedent(f"""\
        from rows_model import build
        m = build(escape={escape})
        m.export("{name}_nomx", locked_spaces=["Data"])
        m.close()
        """), encoding="utf-8")
    calls = "".join(f"    p[k].{c}()\n" for c in ("sa", "age", "sex", "both", "fee", "pv", "model_point"))
    (tmp_path / f"sample_{name}.py").write_text(
        f"from {name}_nomx import mx_model\np = mx_model.Projection\nfor k in (1, 2, 3):\n{calls}"
        + ("    p[k].whole()\n" if escape else "") + "mx_model.Projection.collect()\n", encoding="utf-8")
    (tmp_path / "spec.py").write_text(
        repr({"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}), encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600,
                                    capture_output=True, text=True)
    run(f"build_{name}.py")
    run("-m", "modelx_cython", f"{name}_nomx", "--sample", f"sample_{name}.py", "--spec", "spec.py")
    outs = {pkg: run("checks.py", pkg).stdout for pkg in (f"{name}_nomx_cy", f"{name}_nomx")}
    report = json.loads((tmp_path / f"{name}_nomx_cy/_mx_admission.json").read_text(encoding="utf-8"))
    cells = {key.rsplit("._f_", 1)[-1]: value for key, value in report.items()
             if isinstance(value, dict) and "._f_" in key}
    generated = (tmp_path / f"{name}_nomx_cy/_mx_classes.py").read_text(encoding="utf-8")
    return outs, cells, generated, name


def test_row_fields_read_without_the_row(tmp_path):
    outs, cells, generated, name = _build(tmp_path, escape=False)
    assert outs[f"{name}_nomx_cy"] == outs[f"{name}_nomx"], outs
    assert "'raise', 'sa', 'KeyError', '99'" in outs[f"{name}_nomx"], outs       # the missing key, as live
    for route in ("'direct', 'sa'", "'collect', 1", "'collect', 4"):          # a removed key, as live
        assert f"('raise', {route}, 'KeyError', '2')" in outs[f"{name}_nomx"], (route, outs)
    assert cells["model_point"]["row_cell_inlined"] is True, cells["model_point"]
    assert sum(cells[c]["row_fields_inlined"] for c in ("sa", "age", "sex", "both")) == 4, cells
    # a str key is held per item too (diag-run8 F3 held row): read once per item, then by position
    assert cells["cost_row"]["row_cell_inlined"] is True, cells["cost_row"]
    assert cells["fee"]["row_fields_inlined"] == 1, cells["fee"]
    assert "hold_row(" in generated         # T-5c: the row held per item on the prepared table
    assert "'edited input', 999.0" in outs[f"{name}_nomx"], outs


def test_an_escaping_row_keeps_the_record_path(tmp_path):
    outs, cells, generated, name = _build(tmp_path, escape=True)
    assert outs[f"{name}_nomx_cy"] == outs[f"{name}_nomx"], outs
    assert cells["model_point"]["row_cell_inlined"] is False and cells["model_point"]["row_cell_kept"], cells
    assert all(cells[c]["row_fields_inlined"] == 0 for c in ("sa", "age", "sex", "both")), cells
    assert "return self.data._mx_cell_points().loc[self.policy_id]" in generated   # T-5c: the escaping row is pandas'
