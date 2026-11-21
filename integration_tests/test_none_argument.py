"""None passed for a str-typed parameter that the sample only called with strings behaves as live.

Compiled classes are final, so a call between Cells is a direct C call; a parameter typed from the
trace as ``str`` is declared to accept None, as Python does, and the formula decides what None means:
the same value as live, or an exception as live (a str method applied to None raises TypeError
compiled, AttributeError live: Cython calls the str method directly; a known difference).
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx


def compile_model(tmp_path, sample):
    (tmp_path / "sample.py").write_text(sample, encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--no-spec"], cwd=tmp_path, check=True)


def test_none_for_a_str_parameter_behaves_as_live(tmp_path):
    model = mx.new_model("NoneArgument")
    try:
        space = model.new_space("Projection")
        space.new_cells("size", "lambda t, kind: t + (len(kind) if kind is not None else 100)")
        space.new_cells("upper", "lambda t, kind: kind.upper() + str(t)")
        space.new_cells("total", "def total(kind):\n    return sum(size(t, kind) for t in range(3))\n")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    compile_model(tmp_path, "from exported import mx_model\np = mx_model.Projection\n"
                            "p.total('ab'); p.upper(1, 'x')\n")
    probe = textwrap.dedent("""\
        import json
        from {package} import mx_model
        p = mx_model.Projection
        out = {{"size": p.size(1, None), "total": p.total(None)}}
        try:
            p.upper(1, None)
        except Exception as error:
            out["upper"] = type(error).__name__
        print(json.dumps(out))
        """)
    results = {}
    for package in ("exported", "exported_cy"):
        run = subprocess.run([sys.executable, "-c", probe.format(package=package)], cwd=tmp_path,
                             capture_output=True, text=True, check=True)
        results[package] = run.stdout.strip()
    live, compiled = (json.loads(results[p]) for p in ("exported", "exported_cy"))
    assert live == {"size": 101, "total": 303, "upper": "AttributeError"}, live
    assert compiled == dict(live, upper="TypeError"), compiled
    generated = (tmp_path / "exported_cy/_mx_classes.py").read_text(encoding="utf-8")
    assert "kind: str | None" in generated, generated
