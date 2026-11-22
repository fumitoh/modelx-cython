"""An integer power of C integers is exact or raises naming the Cell; it never wraps.

Cython computes a C integer raised to a non-negative integer literal (and any C integer power
stored in a C integer) with its own unchecked integer power, which wraps where Python gives a
big integer. The sample here keeps every value within 64 bits, as a real book does, so the
Cells are typed as C integers; the run then calls them past that range.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx
import pytest

CELLS = {
    "h": "def h(t):\n    return t\n",
    "const_exponent": "def const_exponent(t):\n    return h(t) ** 20\n",
    "local_base": "def local_base(t):\n    x = h(t)\n    return x ** 20\n",
    "float_context": "def float_context(t):\n    return float(h(t) ** 20)\n",
    "variable_exponent": "def variable_exponent(t):\n    return h(t) ** h(t)\n",
    "comparison": "def comparison(t):\n    return h(t) ** 20 > 10\n",
    "negative_exponent": "def negative_exponent(t):\n    return 2 ** -h(t)\n",
    "zero_exponent": "def zero_exponent(t):\n    return h(t) ** 0\n",
}
INSIDE = {"const_exponent": [0, 1, 2, 3, 8], "local_base": [0, 3, 8], "float_context": [3, 8],
          "variable_exponent": [0, 1, 2, 3, 15], "comparison": [0, 1, 3], "negative_exponent": [0, 1, 5],
          "zero_exponent": [0, 7]}
# exact value leaves 64 bits: 9**20, 16**16
OUTSIDE = {"const_exponent": 9, "local_base": 9, "float_context": 9, "variable_exponent": 16, "comparison": 9}


@pytest.fixture(scope="module")
def power_build(tmp_path_factory):
    # the checked mode (owner's rule): the small sample would resolve the unchecked mode
    return _power_build(tmp_path_factory, "checked")


@pytest.fixture(scope="module")
def power_build_unchecked(tmp_path_factory):
    return _power_build(tmp_path_factory, "unchecked")


def _power_build(tmp_path_factory, mode):
    work = tmp_path_factory.mktemp(f"integer-power-{mode}")
    model = mx.new_model("IntegerPower")
    try:
        space = model.new_space("Sp")
        for name, formula in CELLS.items():
            space.new_cells(name, formula)
        live = {name: {str(t): repr(getattr(space, name)(t)) for t in ts} for name, ts in INSIDE.items()}
        model.export(work / "exported")
    finally:
        model.close()
    (work / "live.json").write_text(json.dumps(live))
    (work / "sample.py").write_text(
        "from exported import mx_model\nSp = mx_model.Sp\n"
        + "".join(f"Sp.{name}(2); Sp.{name}(3)\n" for name in INSIDE) + "Sp.h(0)\n")
    (work / "spec.py").write_text(repr({"integer_checks": mode, "spaces": {"Sp": {"cells_param_size": {"t": 20}}}}))
    run = subprocess.run([sys.executable, "-B", "-m", "modelx_cython", "exported", "--sample", "sample.py",
                          "--spec", "spec.py"], cwd=work, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    return work


def _probe(work, code):
    run = subprocess.run([sys.executable, "-B", "-c", textwrap.dedent(code)], cwd=work,
                         capture_output=True, text=True, timeout=240)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode", ["checked", "unchecked"])
def test_power_inside_the_range_equals_live(power_build, power_build_unchecked, mode):
    work = power_build if mode == "checked" else power_build_unchecked
    live = json.loads((work / "live.json").read_text())
    compiled = _probe(work, f'''
        import json
        from exported_cy import mx_model
        print(json.dumps({{name: {{str(t): repr(getattr(mx_model.Sp, name)(t)) for t in ts}}
                          for name, ts in {INSIDE!r}.items()}}))
    ''')
    for name in INSIDE:
        if name == "negative_exponent":
            # Cython types ``2 ** -t`` as a double (unchanged here): equal values, 1.0 for live 1 at t = 0
            assert {t: float(v) for t, v in compiled[name].items()} == {t: float(v) for t, v in live[name].items()}
        else:
            assert compiled[name] == live[name], name


def test_power_beyond_64_bits_raises_naming_the_cell(power_build):
    outcome = _probe(power_build, f'''
        import json
        from exported_cy import mx_model
        out = {{}}
        for name, t in {OUTSIDE!r}.items():
            try:
                out[name] = ["value", repr(getattr(mx_model.Sp, name)(t))]
            except (OverflowError, TypeError) as error:
                out[name] = [type(error).__name__, str(error)]
        print(json.dumps(out))
    ''')
    for name in OUTSIDE:
        kind, text = outcome[name]
        assert kind == "OverflowError", (name, outcome[name])
        assert text.startswith(f"Sp.{name}:"), (name, text)




def test_unchecked_mode_power(power_build_unchecked):
    """Unchecked mode: a power stored as a C integer still takes the checked power (Cython's own
    wraps, and gives a double for a variable exponent, which a C integer cannot take); in a float
    or comparison context it is Cython's C power and wraps beyond 64 bits, as on upstream."""
    outcome = _probe(power_build_unchecked, f'''
        import json
        from exported_cy import mx_model
        out = {{}}
        for name, t in {OUTSIDE!r}.items():
            try:
                out[name] = ["value", repr(getattr(mx_model.Sp, name)(t))]
            except (OverflowError, TypeError) as error:
                out[name] = [type(error).__name__, str(error)]
        print(json.dumps(out))
    ''')
    for name in ("const_exponent", "local_base", "variable_exponent"):
        assert outcome[name][0] == "OverflowError" and outcome[name][1].startswith(f"Sp.{name}:"), outcome[name]
    for name in ("float_context", "comparison"):
        assert outcome[name][0] == "value", outcome[name]          # plain C power
