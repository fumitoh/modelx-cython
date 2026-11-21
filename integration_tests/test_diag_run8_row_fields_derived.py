"""diag-run8 F3 with a derived Space: the exported model writes the base and the derived Space as classes
that each define the row cell, so the rule refuses it (another class defines it) whether the derived
Space overrides the row cell, overrides a reader, or overrides nothing; every reader keeps the Record path.
Compiled equals the exported model."""

import json
import subprocess
import sys
import textwrap

import pytest

MODEL = """\
import modelx as mx
import pandas as pd
POINTS = pd.DataFrame({"sa": [100.0, 200.0, 300.0], "age": [30, 40, 50]}, index=pd.Index([1, 2, 3], name="policy_id"))
OTHER = POINTS.assign(sa=POINTS["sa"] * 10)
IDS = (1, 2, 3)


def build(override):
    model = mx.new_model("Derived")
    data = model.new_space("Data", refs={"POINTS": POINTS, "OTHER": OTHER})
    data.new_cells("points", "def points():\\n    return POINTS.copy()\\n")
    data.new_cells("other", "def other():\\n    return OTHER.copy()\\n")
    base = model.new_space("Base", refs={"data": data})
    base.new_cells("model_point", "lambda: data.points().loc[policy_id]")
    base.new_cells("sa", 'lambda: float(model_point()["sa"])')
    base.new_cells("pv", "lambda: sa() * 2.0")
    proj = model.new_space("Projection", bases=base, formula="lambda policy_id: None", refs={"ids": IDS},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"pv": pv()}}')
    if override == "row":
        proj.model_point.formula = "lambda: data.other().loc[policy_id]"
    elif override == "reader":
        proj.sa.formula = 'lambda: float(model_point()["sa"]) + 1.0'
    return model
"""

CHECKS = """\
import sys
from importlib import import_module
p = import_module(sys.argv[1]).mx_model.Projection
# row values compared as numbers; the type of the public row is compared in the test below
print(repr([(p[i].sa(), p[i].pv(), sorted((str(k), float(v)) for k, v in p[i].model_point().items())) for i in (1, 2, 3)]))
print(type(p[1].model_point()).__name__)
"""


@pytest.mark.parametrize("override", ["row", "reader", "none"])
def test_derived_space_keeps_the_record_path(tmp_path, override):
    (tmp_path / "derived_model.py").write_text(MODEL, encoding="utf-8")
    name = f"d_{override}"
    (tmp_path / "build_model.py").write_text(textwrap.dedent(f"""\
        from derived_model import build
        m = build({override!r})
        m.export({name!r}, locked_spaces=["Data"])
        m.close()
        """), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        f"from {name} import mx_model\nfor k in (1, 2, 3):\n    mx_model.Projection[k].collect_outputs()\n"
        "    mx_model.Projection[k].model_point()\nmx_model.Projection.collect()\n", encoding="utf-8")
    (tmp_path / "spec.py").write_text(
        repr({"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}), encoding="utf-8")
    (tmp_path / "checks.py").write_text(CHECKS, encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600,
                                    capture_output=True, text=True)
    run("build_model.py")
    run("-m", "modelx_cython", name, "--sample", "sample.py", "--spec", "spec.py")
    outs = {pkg: run("checks.py", pkg).stdout.splitlines() for pkg in (name + "_cy", name)}
    assert outs[name + "_cy"][0] == outs[name][0], outs
    report = json.loads((tmp_path / f"{name}_cy/_mx_admission.json").read_text(encoding="utf-8"))
    rows = {k: v for k, v in report.items() if isinstance(v, dict) and k.endswith("._f_model_point")}
    assert rows and all(v["row_cell_inlined"] is False and "another class" in (v["row_cell_kept"] or "")
                        for v in rows.values()), rows
    readers = [v for k, v in report.items() if isinstance(v, dict) and k.endswith("._f_sa")]
    assert readers and all(v["row_fields_inlined"] == 0 for v in readers), readers


# T-5c: was an open difference on the helper layer (the compiled public row cell returned the helper Record); on
# prepared tables the row is live's pandas Series
def test_derived_space_public_row_is_a_pandas_row(tmp_path):
    test_derived_space_keeps_the_record_path(tmp_path, "none")
    out = subprocess.run([sys.executable, "checks.py", "d_none_cy"], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout.splitlines()
    assert out[1] == "Series", out
