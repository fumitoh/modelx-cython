"""D81 (diag-vec): a cell returning a list of integers built by comprehension (or an empty list) is an
integer vector (I64Vec) on the dense vector cache; iteration, filtered generator sums, len, index and the
empty vector are lowered and equal the exported model. Integer results keep exact integer values and
type; a float sum over an empty vector is 0.0 compiled where Python gives int 0 (D24(2)): equal as a number.
"""
import json
import subprocess
import sys

import modelx as mx

FORMULAS = {
    "entries": ("def entries(t):\n    if t % 3 == 0:\n        return []\n"
                "    return [s for s in range(t % 5 + 1)]\n"),
    "w": "def w(t, s):\n    return 0.1 * (s + 1) + 0.01 * t\n",
    "total": "def total(t):\n    return sum(w(t, s) for s in entries(t))\n",
    "filt": "def filt(t):\n    return sum(w(t, s) * 2.0 for s in entries(t) if w(t, s) > 0.25)\n",
    "count": "def count(t):\n    return len(entries(t))\n",
    "last": "def last(t):\n    v = entries(t)\n    return v[-1] if len(v) > 0 else -1\n",
    "loop": "def loop(t):\n    acc = 0\n    for s in entries(t):\n        acc += s * s + 3\n    return acc\n",
    "isum": "def isum(t):\n    return sum(entries(t))\n",
}
NAMES = ["total", "filt", "count", "last", "loop", "isum"]

SAMPLE = """\
from exported import mx_model
p = mx_model.Projection
for t in range(0, 12):
    for name in {names!r}:
        getattr(p, name)(t)
"""

CHECK = """\
from exported import mx_model as original
from exported_cy import mx_model as compiled

o, c = original.Projection, compiled.Projection
empty_float = 0
for t in range(0, 12):
    v = c.entries(t)
    assert type(v).__name__ == 'list' and list(v) == o.entries(t) and v == o.entries(t), t     # D154
    assert all(type(x) is int for x in v), t
    for name in {names!r}:
        a, b = getattr(o, name)(t), getattr(c, name)(t)
        assert a == b, (name, t, a, b)
        if type(a) is not type(b):
            # D24(2): a float sum over an empty vector
            assert name in ('total', 'filt') and a == 0 and type(a) is int and b == 0.0, (name, t, a, b)
            empty_float += 1
        if name in ('total', 'filt') and type(a) is float:
            assert a.hex() == b.hex(), (name, t)
assert empty_float > 0
# D154: a new list per Python call; changing it never reaches the cache
assert c.entries(4) == c.entries(4) and c.entries(4) is not c.entries(4)
assert c._mx_cell_entries(4) is c._mx_cell_entries(4)        # cached, not rebuilt
mine = c.entries(4)
mine[0] = 7
mine.append(8)
assert c.entries(4) == o.entries(4)
print('integer vectors ok')
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


def test_integer_vector_cells(tmp_path):
    model = mx.new_model("IntVectors")
    try:
        space = model.new_space("Projection")
        for formula in FORMULAS.values():
            space.new_cells(formula=formula)
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(SAMPLE.format(names=NAMES), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--no-spec"], cwd=tmp_path, check=True)
    admission = json.loads((tmp_path / "exported_cy" / "_mx_admission.json").read_text(encoding="utf-8"))
    reports = _reports(admission)
    assert reports["entries"]["vectors"]["shape"] == "I", reports["entries"]
    assert reports["entries"]["cache"] == "vector-dense", reports["entries"]
    for name in NAMES:
        assert reports[name]["vectors"]["lowered"] >= 1, (name, reports[name]["vectors"])
        assert not reports[name]["vectors"]["fallbacks"], (name, reports[name]["vectors"])
    (tmp_path / "check.py").write_text(CHECK.format(names=NAMES), encoding="utf-8")
    result = subprocess.run([sys.executable, "check.py"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "integer vectors ok" in result.stdout
