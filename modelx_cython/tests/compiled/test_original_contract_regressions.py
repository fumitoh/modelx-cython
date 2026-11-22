import subprocess
import sys

import modelx as mx


def compile_model(tmp_path, sample):
    (tmp_path / "sample.py").write_text(sample, encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--no-spec"], cwd=tmp_path, check=True)


def test_populated_numeric_cache_round_trips(tmp_path):
    model = mx.new_model("NumericPickle")
    try:
        space = model.new_space("Projection")
        space.new_cells("value", "lambda rate: rate + 2.0")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    compile_model(tmp_path, "from exported import mx_model\nmx_model.Projection.value(0.5)\n")
    subprocess.run([sys.executable, "-c", """
import pickle
from exported_cy import mx_model
p = mx_model.Projection
assert p.value(0.5) == 2.5
restored = pickle.loads(pickle.dumps(mx_model)).Projection
assert restored.value(0.5) == 2.5
assert restored.value(0.75) == 2.75
"""], cwd=tmp_path, check=True)


def test_ordinary_generator_captures_typed_parameter(tmp_path):
    model = mx.new_model("GeneratorCapture")
    try:
        space = model.new_space("Projection")
        space.new_cells("claims", "lambda t, kind: t + (1 if kind == 'DEATH' else 10)")
        space.new_cells("total", """
def total(kind=None):
    return sum(claims(t, kind) for t in range(3))
""")
        assert space.total("DEATH") == 6 and space.total("LAPSE") == 33
        model.export(tmp_path / "exported")
    finally:
        model.close()
    compile_model(tmp_path, "from exported import mx_model\nmx_model.Projection.total('DEATH')\n")
    for package in ("exported", "exported_cy"):
        subprocess.run([sys.executable, "-c", f"""
from {package} import mx_model
p = mx_model.Projection
assert p.total("DEATH") == p.total("DEATH") == 6
assert p.total("LAPSE") == 33
assert p.total() == 33
"""], cwd=tmp_path, check=True)
