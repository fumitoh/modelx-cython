"""D94 (diag-vec2): a module with no vector cell whose formulas allocate local vectors is lowered too.

Only the formulas with a typed local vector change (the module's other formulas are generated as before);
a local list that is not a typed vector (written with an int, a mixed min/max or an untyped object, or
appended with one) stays a Python list with exact list semantics. Compiled = exported, bit by bit and by type.
"""
import json
import subprocess
import sys

import modelx as mx

FORMULAS = {
    "n": "def n():\n    return 40\n",
    "rate": "def rate(t):\n    return 0.002 * (t + 1) + 1e-17 * t\n",
    "price": "def price():\n    return 100.0\n",
    # the Sofort_DE_S form: preallocated float vectors, an appended int vector, a bisection
    "solve": '''def solve():
    v = 1.0 / 1.03
    disc = [0.0] * n()
    dens = [0.0] * n()
    paid = []
    m = 0
    for t in range(n()):
        d = rate(t) - rate(t) * 0.5
        if d <= 0.0:
            continue
        disc[m] = v ** (t / 12.0)
        dens[m] = d
        paid.append(0 if t < 5 else (t - 5) // 3 + 1)
        m += 1
    lo = 0.0
    hi = 50.0
    for it in range(60):
        mid = 0.5 * (lo + hi)
        g = mid * 3.0
        for j in range(m):
            g += disc[j] * dens[j] * max(price() - paid[j] * mid, 0.0)
        if g > 40.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
''',
    # not typed: an int written into a float list, an object written, an int appended to an empty list
    "int_write": "def int_write(t):\n    xs = [0.0] * 3\n    xs[1] = t\n    return [type(x).__name__ for x in xs]\n",
    "obj_write": "def obj_write(t):\n    xs = [0.0] * 3\n    xs[2] = None if t % 2 else 1.5\n    return sum(1 for x in xs if x is None)\n",
    "mixed_append": ("def mixed_append(t):\n    out = []\n    for k in range(4):\n"
                     "        out.append(max(0.5, k))\n    return [type(x).__name__ for x in out]\n"),
    # untouched: a formula without local vectors keeps its generated text
    "plain": "def plain(t):\n    total = 0.0\n    for k in range(t):\n        total += rate(k) ** 2.0\n    return total\n",
}
NAMES = ["int_write", "obj_write", "mixed_append", "plain"]

SAMPLE = """\
from exported import mx_model
p = mx_model.Projection
p.solve()
for t in range(0, 6):
    for name in {names!r}:
        getattr(p, name)(t)
"""

CHECK = """\
from exported import mx_model as original
from exported_cy import mx_model as compiled

o, c = original.Projection, compiled.Projection
assert o.solve().hex() == c.solve().hex(), (o.solve(), c.solve())
for t in range(0, 6):
    for name in {names!r}:
        a, b = getattr(o, name)(t), getattr(c, name)(t)
        assert a == b and type(a) is type(b), (name, t, a, b)
        if isinstance(a, float):
            assert a.hex() == b.hex(), (name, t)
assert o.int_write(3) == ['float', 'int', 'float'] and o.mixed_append(0) == ['float', 'int', 'int', 'int']
print('local vectors ok')
"""


def _reports(admission):
    reports = {}

    def walk(d):
        for key, value in d.items():
            if isinstance(value, dict) and ("vectors" in value or "cache" in value):
                reports[key.rsplit("._f_", 1)[-1]] = value
            elif isinstance(value, dict):
                walk(value)
    walk(admission)
    return reports


def test_local_vectors_in_a_module_without_vector_cells(tmp_path):
    model = mx.new_model("LocalVectors")
    try:
        space = model.new_space("Projection")
        for formula in FORMULAS.values():
            space.new_cells(formula=formula)
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(SAMPLE.format(names=NAMES), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--no-spec"], cwd=tmp_path, check=True)
    reports = _reports(json.loads((tmp_path / "exported_cy" / "_mx_admission.json").read_text(encoding="utf-8")))
    assert all(not r.get("vectors", {}).get("shape") for r in reports.values())      # no vector cell
    solve = reports["solve"]["vectors"]
    assert solve["vector_locals"] == {"dens": "V", "disc": "V", "paid": "I"}, solve
    assert solve["lowered"] >= 5 and not solve["fallbacks"], solve
    for name in ("int_write", "obj_write", "mixed_append"):
        vectors = reports[name].get("vectors")
        assert vectors is None or not vectors.get("vector_locals"), (name, vectors)
    assert "vectors" not in reports["plain"], reports["plain"]           # not lowered: generated as before
    (tmp_path / "check.py").write_text(CHECK.format(names=NAMES), encoding="utf-8")
    result = subprocess.run([sys.executable, "check.py"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "local vectors ok" in result.stdout
