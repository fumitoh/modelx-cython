"""D87 (diag-vec): a list display of typed floats (or typed integers) returned by a vector cell, or bound to
a local used as a vector, is written straight into the typed vector (no Python list). Elements are evaluated
left to right; an exception in an element leaves no cached value; mixed or untyped elements and the empty
display keep the list path. Compiled = exported, value by value, bit by bit and by type.
"""
import json
import subprocess
import sys

import modelx as mx

FORMULAS = {
    "rate": "def rate(t):\n    return 0.01 * (t + 1) + 1e-17 * t\n",
    "count": "def count(t):\n    return t % 4\n",
    "trans": ("def trans(t, g):\n    q = rate(t) * g\n    stay = 1.0 - q\n"
              "    return [stay, q * 0.25, q * 0.75, -0.0 * q]\n"),
    "ints": "def ints(t):\n    return [count(t), count(t) + 1, 7]\n",
    "local": ("def local(t):\n    v = [rate(t), rate(t) * 2.0]\n    acc = 0.0\n"
              "    for x in v:\n        acc += x\n    return acc + v[1]\n"),
    "risky": ("def risky(t):\n    q = rate(t)\n"
              "    return [q, 1.0 / (q - rate(4)), q * 2.0]\n"),       # ZeroDivisionError at t == 4
    "mixed": "def mixed(t):\n    return [rate(t), float(count(t)), 1.5] if t > 100 else [rate(t), 2.0]\n",
    "untyped": "def untyped(t):\n    return [rate(t), float(str(t))]\n",
    "empty": "def empty(t):\n    return [] if t % 2 else [rate(t)]\n",
    "reader": ("def reader(t):\n    p = trans(t, 2)\n"
               "    return p[0] + p[1] * 2.0 + sum(trans(t, 3)) + sum(ints(t)) + len(empty(t))\n"),
}
NAMES = ["ints", "local", "mixed", "untyped", "empty", "reader"]

SAMPLE = """\
from exported import mx_model
p = mx_model.Projection
for t in range(0, 10):
    for name in {names!r}:
        getattr(p, name)(t)
    for g in range(1, 4):
        p.trans(t, g)
    if t != 4:
        p.risky(t)
"""

CHECK = """\
from exported import mx_model as original
from exported_cy import mx_model as compiled


def bits(value):
    if isinstance(value, float):
        return ('f', value.hex())
    if isinstance(value, int):
        return ('i', value)
    if type(value).__name__ in ('list', 'F64Vec', 'I64Vec'):
        return ('v',) + tuple(bits(v) for v in value)
    return ('r', repr(value))


o, c = original.Projection, compiled.Projection
for repeat in range(2):
    for t in range(0, 10):
        for name in {names!r}:
            assert bits(getattr(o, name)(t)) == bits(getattr(c, name)(t)), (name, t)
        for g in range(1, 4):
            assert bits(o.trans(t, g)) == bits(c.trans(t, g)), (t, g)
        if t != 4:
            assert bits(o.risky(t)) == bits(c.risky(t)), t
assert type(c.trans(3, 2)).__name__ == 'list' and type(c.ints(3)).__name__ == 'list'     # D154
for attempt in range(2):           # raises every time: no partial value was cached
    for model in (o, c):
        try:
            model.risky(4)
        except ZeroDivisionError:
            pass
        else:
            raise AssertionError('risky(4) must raise')
print('vector displays ok')
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


def test_list_displays_are_written_into_typed_vectors(tmp_path):
    model = mx.new_model("Displays")
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

    lowered = [name for name in ("trans", "ints", "local", "risky") if reports[name]["vectors"]["lowered"] >= 1]
    assert lowered == ["trans", "ints", "local", "risky"], {n: reports[n]["vectors"] for n in reports}
    (tmp_path / "check.py").write_text(CHECK.format(names=NAMES), encoding="utf-8")
    result = subprocess.run([sys.executable, "check.py"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "vector displays ok" in result.stdout
