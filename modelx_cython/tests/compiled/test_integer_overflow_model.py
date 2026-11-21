"""The integer overflow model in both integer-check modes (owner's rule).

Each Cell computes a C integer value whose exact (Python) result leaves 64 bits, or depends on
Python's integer semantics. The sample keeps every value small, as a real book does, so the Cells
are typed as C integers; the run then calls them with the values below.

Checked mode: every case is exact or raises an error naming the Cell; none is silently wrong.
Unchecked mode: C integer arithmetic is plain C, as on upstream. The cases marked "plain" are
outside its guarantee (their value is whatever C gives); the others are exact or raise as in the
checked mode: the conversions at the Python boundary stay checked and named, and a power stored as
a C integer takes the checked power in both modes.
"""
import json
import re
import subprocess
import sys
import textwrap

import modelx as mx
import pytest

CELLS = {
    "h": "def h(t):\n    return t\n",
    "c": "def c(t):\n    return h(t) * h(t) * h(t)\n",
    "neg": "def neg(t):\n    x = t * BIG - BIG - BIG\n    return -x\n",
    "pw": "def pw(t):\n    return h(t) ** 20\n",
    "sh": "def sh(t):\n    return h(t) << 60\n",
    "rs": "def rs(t):\n    return (h(t) - 10) >> 1\n",
    "fd": "def fd(t):\n    return (h(t) - 10) // 4\n",
    "md": "def md(t):\n    return (h(t) - 10) % 4\n",
    "fm": "def fm(t):\n    x = t * BIG - BIG - BIG\n    return x // (h(t) - 1)\n",
    "il": "def il(t):\n    x = h(t)\n    return x * x * x * x // (x * x)\n",
    "ie": "def ie(t):\n    return h(t) * h(t) * h(t) * h(t) // (h(t) * h(t))\n",
    "ls": "def ls(t):\n    s = 0\n    for i in range(4):\n        s += c(t)\n    return s\n",
    "sm": "def sm(t):\n    return sum(c(k) for k in range(t, t + 4))\n",
    "di": "def di(t):\n    return int(h(t) * 1e18)\n",
    "mx": "def mx(t):\n    return max(c(t), c(t) + c(t))\n",
    "pl": "def pl(t):\n    x = h(t)\n    return x ** 20\n",
    "pf": "def pf(t):\n    return float(h(t) ** 20)\n",
    "pv": "def pv(t):\n    return h(t) ** h(t)\n",
    "pc": "def pc(t):\n    return h(t) ** 20 > 10\n",
    "dl": "def dl(t):\n    x = int(h(t) * 1e18)\n    return x // 1000\n",
    "lf": "def lf(t):\n    return h(t) + 4611686018427387904 * 4\n",
    "lg": "def lg(t):\n    return h(t) + (4611686018427387904 * 4) // 8\n",
    # LLONG_MIN // -1 and LLONG_MIN % -1 trap in C on every platform; Cython checks the division
    # only where long long is as wide as long (not on Windows) and never the remainder
    "ff": "def ff(t):\n    x = t * BIG - BIG - BIG\n    return float(x // (h(t) - 1))\n",
    "mf": "def mf(t):\n    x = t * BIG - BIG - BIG\n    return float(x % (h(t) - 1))\n",
    "fn": "def fn(t):\n    x = t * BIG - BIG - BIG\n    return x // -1\n",
    "fa": "def fa(t):\n    x = t * BIG - BIG - BIG\n    x //= h(t) - 1\n    return x\n",
    "ma": "def ma(t):\n    x = t * BIG - BIG - BIG\n    x %= h(t) - 1\n    return x\n",
}
RUNS = {"neg": 0, "pw": 9, "sh": 9, "rs": 3, "fd": 3, "md": 3, "fm": 0, "il": 100000, "ie": 100000,
        "ls": 2097151, "sm": 2097148, "di": 100, "mx": 2097151, "pl": 9, "pf": 9, "pv": 16, "pc": 9,
        "dl": 100, "lf": 1, "lg": 1, "ff": 0, "mf": 0, "fn": 0, "fa": 0, "ma": 0}
EXACT = {"rs", "fd", "md", "dl", "lf", "lg", "mf", "ma"}
# checked mode: everything else raises naming the Cell
UNCHECKED_PLAIN = {"neg", "il", "ie", "ls", "mx", "pf", "pc"}
# division check, named boundary conversions, the checked power of a power stored as a C integer,
# the left-shift helper (Python's shift rules, checked in both modes)
UNCHECKED_RAISES = {"fm", "sm", "di", "pw", "pl", "pv", "sh", "ff", "fn", "fa"}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    work = tmp_path_factory.mktemp("overflow-model")
    model = mx.new_model("OverflowModel")
    try:
        space = model.new_space("Sp", refs={"BIG": 2 ** 62})
        for name, formula in CELLS.items():
            space.new_cells(name, formula)
        live = {name: repr(getattr(space, name)(t)) for name, t in RUNS.items()}
        model.export(work / "exported")
    finally:
        model.close()
    return work, live


def _build(exported, tmp_path, mode):
    work, live = exported
    build = tmp_path / mode
    build.mkdir()
    subprocess.run(["cp", "-r", str(work / "exported"), str(build / "exported")], check=True)
    (build / "sample.py").write_text("from exported import mx_model\nSp = mx_model.Sp\n"
                                     + "".join(f"Sp.{n}(2); Sp.{n}(3)\n" for n in CELLS) + "Sp.h(0)\n")
    spec = {"spaces": {"Sp": {"cells_param_size": {"t": 2097156}}}}
    if mode != "auto":
        spec["integer_checks"] = mode
    (build / "spec.py").write_text(repr(spec))
    run = subprocess.run([sys.executable, "-B", "-m", "modelx_cython", "exported", "--sample", "sample.py",
                          "--spec", "spec.py"], cwd=build, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout + run.stderr
    probe = subprocess.run([sys.executable, "-B", "-c", textwrap.dedent(f'''
        import json
        from exported_cy import mx_model
        out = {{}}
        for name, t in {RUNS!r}.items():
            try:
                out[name] = ["value", repr(getattr(mx_model.Sp, name)(t))]
            except (OverflowError, TypeError) as error:
                out[name] = [type(error).__name__, str(error)]
        print(json.dumps(out))
    ''')], cwd=build, capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    report = json.loads((build / "exported_cy" / "_mx_integer_checks.json").read_text())
    assert _unchecked_divisions(build / "exported_cy" / "_mx_classes.c") == []
    return json.loads(probe.stdout.strip().splitlines()[-1]), live, report


_DIVISION = re.compile(r'__Pyx_(div|mod)_\w+\((.*), 0\);')
_SOURCE = re.compile(r'^\s*/\* "([^"]+)":(\d+)')


def _unchecked_divisions(c_file):
    """Division and remainder sites of C integers with a variable divisor outside _mx_sys.pxd's
    helpers: each can trap at LLONG_MIN / -1, whatever the platform checks."""
    found, where = [], None
    for line in c_file.read_text(encoding="utf-8").splitlines():
        if (source := _SOURCE.match(line)) is not None:
            where = source.groups()
        elif _DIVISION.search(line) and not line.lstrip().startswith("static") and not (
                where and where[0].endswith("_mx_sys.pxd")):
            found.append((where, line.strip()))
    return found


def test_checked_mode_is_exact_or_names_the_cell(exported, tmp_path):
    outcome, live, report = _build(exported, tmp_path, "checked")
    assert report["mode"] == "checked" and report["requested"] == "checked"
    for name in RUNS:
        kind, text = outcome[name]
        if name in EXACT:
            assert outcome[name] == ["value", live[name]], name
        else:
            assert kind == "OverflowError" and text.startswith(f"Sp.{name}"), (name, outcome[name])


def test_unchecked_mode_is_plain_c_outside_the_boundary(exported, tmp_path):
    outcome, live, report = _build(exported, tmp_path, "unchecked")
    assert report["mode"] == "unchecked" and "intermediate values are not traced" in report["limit"]
    for name in EXACT:
        assert outcome[name] == ["value", live[name]], name
    for name in UNCHECKED_RAISES:
        assert outcome[name][0] == "OverflowError", (name, outcome[name])
    for name in ("sm", "di", "pw", "pl", "pv", "sh"):
        assert outcome[name][1].startswith(f"Sp.{name}"), (name, outcome[name])
    for name in UNCHECKED_PLAIN:         # plain C arithmetic: no check, no exception
        assert outcome[name][0] == "value", (name, outcome[name])


def test_auto_mode_follows_the_largest_traced_integer(exported, tmp_path):
    _, _, report = _build(exported, tmp_path, "auto")
    # the Reference BIG = 2**62 is traced: beyond 2**31, so the checked mode
    assert report["mode"] == "checked" and report["requested"] == "auto"
    assert report["largest_traced_integer"] >= 2 ** 62
