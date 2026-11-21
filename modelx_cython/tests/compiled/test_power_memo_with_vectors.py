"""diag-us2 (D91): in a model with vector cells, a scalar power whose operands read no vector (only
parameters, Cell calls and literals) is left to powers.py, so its one-entry memo applies as in a model
without vectors (VA_US_S mort_rate_mth: ``1 - (1 - q) ** (1/12)``). A power over vector elements is still
lowered by the vector pass. Values equal the exported model bit for bit."""
import json
import subprocess
import sys

import modelx as mx

FORMULAS = {
    "rates": "def rates(t):\n    return [0.01 * (k + 1) + 0.001 * t for k in range(3)]\n",
    "q": "def q(t):\n    return 0.02 if t < 6 else 0.03\n",
    "q_mth": "def q_mth(t):\n    return 1.0 - (1.0 - q(t)) ** (1.0 / 12.0)\n",
    "vec_pow": "def vec_pow(t):\n    return sum((1.0 + r) ** (-1.0 / 12.0) for r in rates(t))\n",
}
NAMES = ["q_mth", "vec_pow"]

SAMPLE = """\
from exported import mx_model
p = mx_model.Projection[1]
for t in range(0, 12):
    p.q_mth(t); p.vec_pow(t)
"""

CHECK = """\
from exported import mx_model as original
from exported_cy import mx_model as compiled
o, c = original.Projection[1], compiled.Projection[1]
for t in range(0, 12):
    for name in ("q_mth", "vec_pow"):
        a, b = getattr(o, name)(t), getattr(c, name)(t)
        assert a.hex() == b.hex(), (name, t, a, b)
print('power memo ok')
"""


def _reports(admission):
    reports = {}

    def walk(d):
        for key, value in d.items():
            if isinstance(value, dict) and "vectors" in value:
                reports[key.rsplit("._f_", 1)[-1]] = value
            elif isinstance(value, dict):
                walk(value)
    walk(admission)
    return reports


def test_scalar_power_keeps_its_memo_in_a_model_with_vectors(tmp_path):
    model = mx.new_model("PowerMemoVectors")
    try:
        space = model.new_space("Projection", formula=lambda point_id: None)
        for formula in FORMULAS.values():
            space.new_cells(formula=formula)
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "spec.py").write_text("{}\n", encoding="utf-8")       # defaults: automatic optimizations on
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py", "--spec", "spec.py"],
                   cwd=tmp_path, check=True)
    admission = json.loads((tmp_path / "exported_cy" / "_mx_admission.json").read_text(encoding="utf-8"))
    reports = _reports(admission)
    assert reports["rates"]["vectors"]["shape"] == "V", reports["rates"]
    assert reports["q_mth"]["power_memo"] is True, reports["q_mth"]
    assert reports["q_mth"]["vectors"]["powers"] == 0, reports["q_mth"]["vectors"]
    assert reports["vec_pow"]["vectors"]["powers"] >= 1, reports["vec_pow"]["vectors"]
    (tmp_path / "check.py").write_text(CHECK, encoding="utf-8")
    result = subprocess.run([sys.executable, "check.py"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "power memo ok" in result.stdout
