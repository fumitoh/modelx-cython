"""Vector cells (diag-vec): min/max inside a reduction over vectors, and the dense vector cache
of a vector cell with two or three integer parameters.

The exported (Python) model and the compiled one must agree exactly, value by value, bit by bit
and type by type; the forms named below must be lowered (admission report), and a mixed
float/integer min/max whose result is not made a float by arithmetic must not be.
"""
import json
import subprocess
import sys

import modelx as mx

FORMULAS = {
    "limit": "def limit():\n    return 60\n",
    "floor": "def floor():\n    return 5.0\n",
    "days": "def days(band):\n    return [2, 8, 20, 45 + band, 160 - band]\n",
    "probs": "def probs(band):\n    return [0.3 - 0.01 * band, 0.25, 0.2, 0.15, 0.1 + 0.01 * band]\n",
    # lowered: a float times a mixed min/max is a float
    "capped": "def capped(t):\n    b = t % 3\n    return sum(p * min(g, limit()) for g, p in zip(days(b), probs(b)))\n",
    "floored": ("def floored(t):\n    b = t % 3\n    f = floor()\n"
                "    return sum(p * max(f, min(g, limit())) for g, p in zip(days(b), probs(b)))\n"),
    "raw": "def raw(t):\n    b = t % 3\n    f = floor()\n    return sum(p * max(f, g) for g, p in zip(days(b), probs(b)))\n",
    "fmax": "def fmax(t):\n    return sum(max(p, 0.2) * 2.0 for p in probs(t % 3))\n",
    "signed": ("def signed(t):\n    b = t % 3\n"
               "    return sum(p * max(-0.0, min(0.0, g - 2)) for g, p in zip(days(b), probs(b)))\n"),
    # not lowered: the element is a mixed min/max or an integer; Python's value and type are kept
    "mixed": "def mixed(t):\n    b = t % 3\n    return sum(max(floor(), g) for g, p in zip(days(b), probs(b)))\n",
    "ints": "def ints(t):\n    return sum(min(g, limit()) for g in days(t % 3))\n",
    # ties and signed zeros: Python returns the first of equal arguments; elementwise, bit by bit
    "zeros": "def zeros(band):\n    return [0.0, -0.0, 0.0, -0.0, 1.5]\n",
    "nzeros": "def nzeros(band):\n    return [-0.0, 0.0, 0.0, -0.0, 1.5]\n",
    "fives": "def fives(band):\n    return [5, 4, 6, 5, 2 + band]\n",
    "days2": "def days2(band):\n    return [2, 9, 20, 44, 160 - band]\n",
    "tie_ff": "def tie_ff(t):\n    return [max(a, b) * 1.0 for a, b in zip(zeros(t % 3), nzeros(t % 3))]\n",
    "tie_ff_min": "def tie_ff_min(t):\n    return [min(a, b) * 1.0 for a, b in zip(zeros(t % 3), nzeros(t % 3))]\n",
    "tie_ii": "def tie_ii(t):\n    return [min(g, h) * 1.0 for g, h in zip(days(t % 3), days2(t % 3))]\n",
    "tie_if": "def tie_if(t):\n    return [max(5.0, g) * 1.0 + max(g, 5.0) * 0.5 for g in fives(t % 3)]\n",
    "tie_zero": ("def tie_zero(t):\n"
                 "    return [max(0, z) * -1.0 + max(z, 0) * -1.0 for z in zeros(t % 3)]\n"),
    "sum_ff": "def sum_ff(t):\n    return sum(max(a, b) * 1.0 for a, b in zip(zeros(t % 3), nzeros(t % 3)))\n",
    # vector cells with two and three integer parameters
    "pairs": "def pairs(t, g):\n    q = 0.001 * (t + 1) * g\n    return [q, 1.0 - q, float(g)]\n",
    "triple": "def triple(t, g, z):\n    return [0.5 * t + g, float(z), 1e-17 * (t * g * z)]\n",
    "reader": ("def reader(t, g):\n    if t <= 0:\n        return pairs(t, g)[1]\n"
               "    p = pairs(t - 1, g)\n    return reader(t - 1, g) * p[1] + p[0] * pairs(t, g)[2]\n"),
    "reader3": "def reader3(t):\n    return sum(triple(t, g, z)[0] + triple(t, g, z)[2] for g in range(1, 4) for z in range(2))\n",
}
SCALARS = ["capped", "floored", "raw", "fmax", "signed", "mixed", "ints", "reader3", "tie_ff", "tie_ff_min",
           "tie_ii", "tie_if", "tie_zero", "sum_ff"]

SAMPLE = """\
from exported import mx_model
p = mx_model.Projection
for t in range(0, 8):
    for name in {scalars!r}:
        getattr(p, name)(t)
    for g in range(1, 4):
        p.reader(t, g)
"""

CHECK = """\
from importlib.machinery import ExtensionFileLoader

from exported import mx_model as original
from exported_cy import _mx_classes, mx_model as compiled

assert isinstance(_mx_classes.__loader__, ExtensionFileLoader)


def bits(value):
    if isinstance(value, float):
        return ('f', value.hex())
    if isinstance(value, int):
        return ('i', value)
    if type(value).__name__ in ('list', 'F64Vec', 'I64Vec'):
        return ('v',) + tuple(bits(v) for v in value)
    return ('r', repr(value))


o, c = original.Projection, compiled.Projection
for repeat in range(2):          # the second pass reads the caches
    for t in range(0, 8):
        for name in {scalars!r}:
            a, b = getattr(o, name)(t), getattr(c, name)(t)
            assert bits(a) == bits(b), (name, t, a, b)
        for g in range(1, 4):
            assert bits(o.reader(t, g)) == bits(c.reader(t, g)), (t, g)
            assert bits(o.pairs(t, g)) == bits(c.pairs(t, g)), (t, g)
            for z in range(2):
                assert bits(o.triple(t, g, z)) == bits(c.triple(t, g, z)), (t, g, z)
assert type(c.pairs(3, 2)).__name__ == 'list' and type(c.triple(3, 2, 1)).__name__ == 'list'
# D154: a new list per Python call, equal values; the cache itself is the compiled entry's value
assert c.pairs(3, 2) == c.pairs(3, 2) and c.pairs(3, 2) is not c.pairs(3, 2)
assert c._mx_cell_pairs(3, 2) is c._mx_cell_pairs(3, 2)      # cached, not rebuilt
assert type(o.mixed(1)) is type(c.mixed(1)) is float and type(c.ints(1)) is int
# arguments outside the sampled axes: the compiled domain contract (as for every cached cell):
# the same value, or IndexError naming the sampled range
for t, g in ((100, 2), (3, 50), (-1, 1)):
    try:
        got = c.pairs(t, g)
    except IndexError:
        continue
    assert bits(o.pairs(t, g)) == bits(got), (t, g)
print('vector extrema and cache ok')
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


def test_vector_extrema_and_multi_parameter_cache(tmp_path):
    model = mx.new_model("VectorExtrema")
    try:
        space = model.new_space("Projection")
        for formula in FORMULAS.values():
            space.new_cells(formula=formula)
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(SAMPLE.format(scalars=SCALARS), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--no-spec"],
                   cwd=tmp_path, check=True)
    admission = json.loads((tmp_path / "exported_cy" / "_mx_admission.json").read_text(encoding="utf-8"))
    reports = _reports(admission)
    for name in ("capped", "floored", "raw", "fmax", "signed", "tie_ff", "tie_ff_min", "tie_ii", "tie_if",
                 "tie_zero", "sum_ff"):
        assert reports[name]["vectors"]["lowered"] >= 1, (name, reports[name]["vectors"])
        assert not reports[name]["vectors"]["fallbacks"], (name, reports[name]["vectors"])
    for name in ("mixed", "ints"):
        assert reports[name]["vectors"]["fallbacks"], (name, reports[name]["vectors"])
    for name in ("pairs", "triple"):   # the multi-parameter cells keep a vector cache, not a dict
        assert admission[f"exported_cy._mx_classes._c_Projection._f_{name}"]["cache"] == "vector-dense", name
    (tmp_path / "check.py").write_text(CHECK.format(scalars=SCALARS), encoding="utf-8")
    result = subprocess.run([sys.executable, "check.py"], cwd=tmp_path, check=True,
                            capture_output=True, text=True)
    assert "vector extrema and cache ok" in result.stdout
