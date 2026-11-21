"""diag-us2: a vector sum that is the leftmost operand of a ``+``/``-``/``*``/``/`` chain is lowered.

``return (sum(f(t, i) for i in ids()) + g(t))`` (VUL_US_S unloaned_av_pp_at, av_pp): Python evaluates the
leftmost operand first, so the sum computed just before the statement keeps the order of evaluation. It is
lowered to a native loop over the vector (no object iteration, no fallback) and equals the exported model
bit for bit, including an empty vector (D24(2): 0.0 + g(t) is a float either way), a subtraction chain,
an assignment, and a chain where a later operand raises (the exception is the same). The leading operand of
a ``*`` or ``/`` chain is evaluated first as well (``sum(...) / av_pp(t)``, VA_KR_S bond_weight), so it is
lowered the same way; a sum that is not the leftmost operand of any of the four stays on today's path.
"""
import json
import subprocess
import sys

import modelx as mx

FORMULAS = {
    "ids": ("def ids(t):\n    if t % 4 == 0:\n        return []\n"
            "    return [s + 1 for s in range(t % 3 + 1)]\n"),
    "w": "def w(t, s):\n    return 0.1 * s + 0.013 * t\n",
    "g": "def g(t):\n    return 1.0 / (t - 7) if t != 7 else float(t)\n",
    "h": "def h(t):\n    if t == 9:\n        raise ValueError('h')\n    return 0.5 * t\n",
    "lead_add": "def lead_add(t):\n    return (sum(w(t, s) for s in ids(t)) + g(t))\n",
    "lead_sub": "def lead_sub(t):\n    return sum(w(t, s) * 2.0 for s in ids(t)) - g(t) + 1.0\n",
    "lead_assign": "def lead_assign(t):\n    x = sum(w(t, s) for s in ids(t) if s > 1) + g(t)\n    return x * 2.0\n",
    "lead_raise": "def lead_raise(t):\n    return sum(w(t, s) for s in ids(t)) + h(t)\n",
    "lead_mul": "def lead_mul(t):\n    return sum(w(t, s) for s in ids(t)) * g(t) + 1.0\n",
    "lead_div": "def lead_div(t):\n    return sum(w(t, s) for s in ids(t)) / (g(t) + 3.0)\n",
    "lead_div_raise": "def lead_div_raise(t):\n    return sum(w(t, s) for s in ids(t)) / h(t)\n",
    # D93: not the leftmost operand (g(t) is evaluated first): stays on today's path
    "trail": "def trail(t):\n    return g(t) + sum(w(t, s) for s in ids(t))\n",
    "trail_sub": "def trail_sub(t):\n    return g(t) - sum(w(t, s) for s in ids(t))\n",
    "trail_mul": "def trail_mul(t):\n    return g(t) * sum(w(t, s) for s in ids(t))\n",
    "trail_div": "def trail_div(t):\n    return g(t) / (sum(w(t, s) for s in ids(t)) + 1.0)\n",
}
LOWERED = ["lead_add", "lead_sub", "lead_assign", "lead_raise", "lead_mul", "lead_div", "lead_div_raise"]
TRAILING = ["trail", "trail_sub", "trail_mul", "trail_div"]
NAMES = LOWERED + TRAILING

SAMPLE = """\
from exported import mx_model
p = mx_model.Projection
for t in range(0, 12):
    for name in {names!r}:
        try:
            getattr(p, name)(t)
        except (ValueError, ZeroDivisionError):
            pass
"""

CHECK = """\
from exported import mx_model as original
from exported_cy import mx_model as compiled

o, c = original.Projection, compiled.Projection
empty = 0
for t in range(0, 12):
    for name in {names!r}:
        try:
            a = getattr(o, name)(t)
        except (ValueError, ZeroDivisionError) as error:
            try:
                getattr(c, name)(t)
            except (ValueError, ZeroDivisionError) as other:
                # the type is the same; a compiled double division's ZeroDivisionError has Cython's text
                assert type(other) is type(error), (name, t)
                assert other.args == error.args or type(error) is ZeroDivisionError, (name, t)
                continue
            raise AssertionError((name, t, 'no ' + type(error).__name__))
        b = getattr(c, name)(t)
        assert a == b and float(a).hex() == float(b).hex(), (name, t, a, b)
        if len(o.ids(t)) == 0:
            empty += 1
assert empty > 0                       # the empty vector is exercised (t % 4 == 0)
print('leading sums ok')
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


def test_leading_vector_sum(tmp_path):
    model = mx.new_model("LeadingSums")
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
    assert reports["ids"]["vectors"]["shape"] == "I", reports["ids"]
    for name in LOWERED:
        assert reports[name]["vectors"]["lowered"] >= 1, (name, reports[name]["vectors"])
        assert not reports[name]["vectors"]["fallbacks"], (name, reports[name]["vectors"])
    for name in TRAILING:
        assert reports[name]["vectors"]["lowered"] == 0, (name, reports[name]["vectors"])
        assert any("generator over vectors not lowered" in f for f in reports[name]["vectors"]["fallbacks"]), name
    (tmp_path / "check.py").write_text(CHECK.format(names=NAMES), encoding="utf-8")
    result = subprocess.run([sys.executable, "check.py"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "leading sums ok" in result.stdout
