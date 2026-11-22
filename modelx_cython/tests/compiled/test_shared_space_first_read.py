"""The first read of a zero-argument cached Cell of a shared, unlocked Space from four threads at once.

A Space that is not an ItemSpace (here ``Data``) is read by every worker of a collect. Exported
without ``locked_spaces`` (as an upstream-style model is), its zero-argument cached Cells publish
their value once: four threads racing on the first read of a fresh model each get the one stored
value, for every result type (typed memoryview, object, C double), and none sees the flag before
the value (``Memoryview is not initialized``) or a value cleared under it.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx


def test_first_read_of_a_shared_space_from_four_threads(tmp_path):
    model = mx.new_model("SharedFirstRead")
    try:
        data = model.new_space("Data")
        data.np = __import__("numpy")
        data.new_cells("rates", "def rates():\n    return np.linspace(0.0, 1.0, 101)\n")
        data.new_cells("table", "def table():\n    return np.arange(12.0).reshape(3, 4)\n")
        data.new_cells("labels", "def labels():\n    return tuple(str(i) for i in range(10))\n")
        data.new_cells("scale", "def scale():\n    return float(rates()[100]) * 2.0\n")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nd = mx_model.Data\n"
        "d.rates()[0], d.table()[0, 0], d.labels()[0], d.scale()\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py", "--no-spec"],
                   cwd=tmp_path, check=True)
    admission = json.loads((tmp_path / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))
    types = {name.rsplit("._f_", 1)[-1]: entry["return_type"] for name, entry in admission.items()}
    assert types["rates"].endswith("[:]") and types["table"].endswith("[:, :]"), types   # typed memoryviews
    probe = textwrap.dedent("""\
        import threading
        from exported_cy import mx_model as m

        errors = []
        for _ in range(1000):
            fresh = type(m)()
            d = fresh.Data
            barrier = threading.Barrier(4, timeout=30)
            def run():
                barrier.wait()
                try:
                    for _ in range(50):
                        assert d.rates()[100] == 1.0
                        assert d.table()[2, 3] == 11.0
                        assert d.labels()[9] == "9"
                        assert d.scale() == 2.0
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
            threads = [threading.Thread(target=run) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert d.rates() is not None and d.labels() is d.labels()   # one object stored
        assert not errors, sorted(set(errors))
        print("ok")
        """)
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stdout + result.stderr
