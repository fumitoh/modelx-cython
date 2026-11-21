import subprocess
import sys

import modelx as mx


def test_fractional_power_in_local_feeding_extrema(tmp_path):
    model = mx.new_model("LocalFractionalPower")
    try:
        space = model.new_space("Projection")
        space.new_cells("native_root", """\
def native_root(value):
    return value ** 0.5
""")
        space.new_cells("clamped_root", """\
def clamped_root(value, ceiling):
    base = value
    root = base ** 0.5
    return min(ceiling, max(0.0, root))
""")
        model.export(tmp_path / "exported")
    finally:
        model.close()

    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\n"
        "mx_model.Projection.native_root(4.0)\n"
        "mx_model.Projection.clamped_root(4.0, 3.0)\n",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-m", "modelx_cython", "exported", "--no-spec"],
        cwd=tmp_path,
        check=True,
    )


    (tmp_path / "assert_compiled.py").write_text(
        """\
from importlib.machinery import ExtensionFileLoader

from exported import mx_model as original
from exported_cy import _mx_classes, mx_model as compiled

assert isinstance(_mx_classes.__loader__, ExtensionFileLoader)

for model in (original, compiled):
    projection = model.Projection
    for value, ceiling in [(0.0, 3.0), (4.0, 3.0), (16.0, 3.0)]:
        assert projection.native_root(value) == value ** 0.5
        expected = min(ceiling, max(0.0, value ** 0.5))
        result = projection.clamped_root(value, ceiling)
        assert result == expected and type(result) is type(expected)
    try:
        projection.clamped_root(-4.0, 3.0)
    except TypeError:
        pass
    else:
        raise AssertionError("a complex root must remain unordered in Python")
""",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "assert_compiled.py"], cwd=tmp_path, check=True)
