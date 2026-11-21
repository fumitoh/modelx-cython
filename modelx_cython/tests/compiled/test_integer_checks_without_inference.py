"""The checked integer mode does not depend on type inference.

With ``"infer_types": false`` a formula's C integer arithmetic is still C arithmetic wherever its operands are
typed (Cell results, typed parameters); the checked mode must check it as with inference on: an exact value or an
OverflowError naming the Cell, never a wrapped value.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx
import pytest

CELLS = {
    "h": "def h(t):\n    return t\n",
    "c": "def c(t):\n    return h(t) * h(t) * h(t)\n",
}


@pytest.mark.parametrize("infer_types", [True, False])
def test_checked_mode_checks_c_arithmetic_with_or_without_inference(tmp_path, infer_types):
    model = mx.new_model("ChecksWithoutInference")
    try:
        space = model.new_space("Sp")
        for name, formula in CELLS.items():
            space.new_cells(name, formula)
        live = repr(space.c(3))
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text("from exported import mx_model\nSp = mx_model.Sp\nSp.c(2); Sp.c(3)\n")
    (tmp_path / "spec.py").write_text(repr({"integer_checks": "checked", "infer_types": infer_types,
                                            "spaces": {"Sp": {"cells_param_size": {"t": 2 ** 23}}}}))
    run = subprocess.run([sys.executable, "-B", "-m", "modelx_cython", "exported", "--sample", "sample.py",
                          "--spec", "spec.py"], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout + run.stderr
    probe = subprocess.run([sys.executable, "-B", "-c", textwrap.dedent('''
        import json
        from exported_cy import mx_model
        out = {"small": repr(mx_model.Sp.c(3))}
        try:
            out["large"] = ["value", repr(mx_model.Sp.c(2 ** 22))]      # (2**22)**3 = 2**66
        except OverflowError as error:
            out["large"] = ["OverflowError", str(error)]
        print(json.dumps(out))
    ''')], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    outcome = json.loads(probe.stdout.strip().splitlines()[-1])
    assert outcome["small"] == live
    kind, text = outcome["large"]
    assert kind == "OverflowError" and text.startswith("Sp.c"), outcome["large"]
