"""Vector cells (W1.6): lists of floats and integers compiled to typed buffers.

One model covers every lowered form; the exported (Python) model and the compiled one must
agree exactly, value by value and bit by bit, and the compiled values must be typed vectors.
"""
import subprocess
import sys

import modelx as mx

FORMULAS = {
    "size": "def size(t):\n    return 5 + t % 4\n",
    "rate": "def rate(t):\n    return 0.01 * (t + 1) + 1e-17 * t\n",
    "factors": "def factors():\n    return [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]\n",
    "ids": "def ids(t):\n    out = []\n    for k in range(size(t)):\n        if k % 2 == 0:\n            out.append(k * 3 + t)\n    return out\n",
    "cohorts": '''def cohorts(t):
    n = size(t)
    pols = [0.0] * n
    val = [0.0] * n
    if t <= 0:
        pols[0] = 1.0
        val[0] = 2.0
        return (pols, val)
    ppols, pval = cohorts(t - 1)
    q = factors()
    for i in range(min(len(ppols), n - 1)):
        cy = i // 2
        if cy > 5:
            cy = 5
        keep = 1.0 - rate(t) * q[cy]
        step = 1.1 if (i + 1) % 3 == 0 else 1.0
        pols[i + 1] = ppols[i] * keep
        val[i + 1] = pval[i] * keep * step
    pols[0] = rate(t - 1)
    val[0] = pols[0] * 3.0
    return (pols, val)
''',
    "powered": "def powered(t):\n    q = rate(t)\n    return [1.0 - (1.0 - q * f) ** (1.0 / 12.0) for f in factors()]\n",
    "shifted": '''def shifted(t):
    v = cohorts(t)[0]
    n = len(v)
    out = [0.0] * n
    for z in range(1, n):
        out[z] = v[z - 1] * 0.5
    out[0] += 1.0
    return out
''',
    "zipped": "def zipped(t):\n    a, b = cohorts(t)\n    return [x * y for x, y in zip(a, b)]\n",
    "ranged": "def ranged(t):\n    zs = range(1, size(t) + 1)\n    return [z * 0.25 for z in zs]\n",
    "sliced": "def sliced(t):\n    v = cohorts(t)[1]\n    head = v[1:4]\n    return head\n",
    "total": "def total(t):\n    return sum(cohorts(t)[0])\n",
    "part": "def part(t):\n    v = cohorts(t)[1]\n    return sum(v[2:]) + sum(v[:-1]) + sum(v[1:3])\n",
    "filtered": "def filtered(t):\n    a, b = cohorts(t)\n    return sum(x * y for x, y in zip(a, b) if x > 0.05)\n",
    "looped": '''def looped(t):
    acc = 0.0
    for x in shifted(t):
        acc += x * x
    for x, y in zip(zipped(t), powered(t)):
        acc += x - y
    return acc
''',
    "length": "def length(t):\n    return len(cohorts(t)[0]) + len(ids(t))\n",
    "id_sum": "def id_sum(t):\n    v = ids(t)\n    return sum(v) + v[-1] + sum(v[1:])\n",
    "element": "def element(t, z):\n    v = cohorts(t)[0]\n    return v[z - 1] if 1 <= z <= len(v) else 0.0\n",
    "checked": """def checked(t):
    v = [-1.0, 0.0, 2.0]
    for i in range(t, t + 1):
        cy = i
        return v[cy]
    return 0.0
""",
    "fallback": "def fallback(t):\n    v = shifted(t)\n    return max(v) + len([x for x in v if x > 0.1]) + sum(v[::2])\n",
    "values": "def values():\n    return [float(k) * 0.1 for k in range(1, 9)]\n",
    "noise": '''def noise(k):
    out = [0.0] * 200
    x = 0.123456789 + k
    for i in range(200):
        x = (x * 3.9) % 1.0
        out[i] = (x - 0.5) * 10.0 ** ((i % 31) - 15)
    out[7] = -0.0
    return out
''',
    "noise_sum": "def noise_sum(k):\n    return sum(noise(k)) + sum(noise(k)[k:k + 50])\n",
    "listed": "def listed(t):\n    return [rate(t), float(t)]\n",
}

SAMPLE = """\
from exported import mx_model
p = mx_model.Projection
for t in range(-1, 9):
    for name in {names!r}:
        getattr(p, name)(t)
    for z in range(-2, 12):
        p.element(t, z)
p.factors()
p.values()
for k in range(-3, 3):
    p.checked(k)
"""

CHECK = """\
import pickle
import threading
from importlib.machinery import ExtensionFileLoader

from exported import mx_model as original
from exported_cy import _mx_classes, mx_model as compiled

assert isinstance(_mx_classes.__loader__, ExtensionFileLoader)
NAMES = {names!r}
VECTORS = {vectors!r}


def bits(value):
    if isinstance(value, float):
        return ('f', value.hex())
    if isinstance(value, int):
        return ('i', value)
    if isinstance(value, tuple) or type(value).__name__ in ('list', 'F64Vec', 'I64Vec'):
        return (type(value).__name__ if isinstance(value, tuple) else 'v',) + tuple(bits(v) for v in value)
    return ('r', repr(value))


o, c = original.Projection, compiled.Projection
for t in range(-1, 9):
    for name in NAMES:
        a = getattr(o, name)(t)
        b = getattr(c, name)(t)
        assert bits(a) == bits(b), (name, t, a, b)
        assert b == a and a == b, (name, t)
    for z in range(-2, 12):
        assert bits(o.element(t, z)) == bits(c.element(t, z)), (t, z)

for name, kind in VECTORS.items():
    value = getattr(c, name)(3)
    parts = value if isinstance(value, tuple) else (value,)
    assert all(type(p).__name__ == kind for p in parts), (name, type(value))
    for p in parts:
        assert pickle.loads(pickle.dumps(p)) == p
        assert list(p) == p and len(p) == len(list(p))      # D154: a list (never equal to a tuple)
assert type(c.listed(2)).__name__ == 'list' and c.listed(2) == [o.rate(2), 2.0]

for k in range(-3, 3):
    assert bits(o.checked(k)) == bits(c.checked(k)), k
for k in (-4, 3):
    try:
        c.checked(k)
    except IndexError:
        pass
    else:
        raise AssertionError(k)

for k in range(40):
    assert bits(o.noise_sum(k)) == bits(c.noise_sum(k)), k

# a cached vector is read by four threads with no lock
shared = c.cohorts(6)[1]
expected = list(shared)
errors = []

def read():
    for _ in range(2000):
        if list(shared) != expected or sum(shared) != sum(expected):
            errors.append('changed')

threads = [threading.Thread(target=read) for _ in range(4)]
for th in threads:
    th.start()
for th in threads:
    th.join()
assert not errors

# D154: a Python caller gets new lists; changing them never reaches the cache
mine = c.cohorts(6)
assert type(mine) is tuple and all(type(part) is list for part in mine)
again = c.cohorts(6)
assert all(a == b and a is not b for a, b in zip(mine, again))     # each inner list is fresh
before = [list(part) for part in mine]
mine[1][0] = 1.0
mine[0].append(2.0)
assert [list(part) for part in c.cohorts(6)] == before
single = c.listed(2)
single[0] = -1.0
assert c.listed(2) == [o.rate(2), 2.0]
print('vector cells ok')
"""


def _build(tmp_path, formulas):
    model = mx.new_model("VectorCells")
    try:
        space = model.new_space("Projection")
        for formula in formulas.values():
            space.new_cells(formula=formula)
        model.export(tmp_path / "exported")
    finally:
        model.close()


def test_vector_cells_compile_to_typed_buffers(tmp_path):
    _build(tmp_path, FORMULAS)
    names = sorted(name for name in FORMULAS if name not in ("factors", "values", "element", "checked", "noise",
                                                              "noise_sum"))
    (tmp_path / "sample.py").write_text(SAMPLE.format(names=names) + (
        "for k in range(40):\n    p.noise_sum(k)\n"), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--no-spec"],
                   cwd=tmp_path, check=True)
    import json
    admission = json.loads((tmp_path / "exported_cy" / "_mx_admission.json").read_text(encoding="utf-8"))
    reports = {key.rsplit("._f_", 1)[-1]: value["vectors"] for module in admission.values()
               for key, value in (module.items() if isinstance(module, dict) else ())
               if isinstance(value, dict) and "vectors" in value}
    if not reports:   # admission is keyed per class
        def walk(d):
            for key, value in d.items():
                if isinstance(value, dict) and "vectors" in value:
                    reports[key.rsplit("._f_", 1)[-1]] = value["vectors"]
                elif isinstance(value, dict):
                    walk(value)
        walk(admission)
    assert reports["cohorts"]["shape"] == ["T", ["V", "V"]]
    assert reports["ids"]["shape"] == "I"
    assert reports["cohorts"]["bounds_hoisted"] == 4
    assert reports["cohorts"]["derived_index_locals"] == ["cy"]
    assert reports["fallback"]["fallbacks"], reports["fallback"]
    # D154: Python callers get lists (a tuple of lists for a tuple of vectors)
    vectors = {"cohorts": "list", "powered": "list", "shifted": "list", "zipped": "list",
               "ranged": "list", "sliced": "list", "ids": "list", "listed": "list"}
    (tmp_path / "check.py").write_text(CHECK.format(names=names, vectors=vectors), encoding="utf-8")
    result = subprocess.run([sys.executable, "check.py"], cwd=tmp_path, check=True,
                            capture_output=True, text=True)
    assert "vector cells ok" in result.stdout


def test_write_to_a_cached_vector_is_a_build_error(tmp_path):
    _build(tmp_path, {
        "vec": "def vec(t):\n    return [1.0, 2.0, float(t)]\n",
        "reader": "def reader(t):\n    v = vec(t)\n    v[0] = 5.0\n    return v[0]\n",
    })
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Projection.vec(1)\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--no-spec"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "VectorWriteError" in result.stderr and "did not allocate" in result.stderr
