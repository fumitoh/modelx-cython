"""A typed-memoryview Cell of a shared Space read by subscript in an item's formula is held per item.

``step(t) = data.rates()[t] * i`` would copy the shared ``Data``'s cached slice on every evaluation (an atomic
acquire and release of the one acquisition count every worker shares). The compiled item holds the slice once
(``holds`` kind ``array``) and reads elements of its own field (in formulas with parameters; a zero-argument Cell
runs once per item and keeps its read): the generated formula calls the held read, the
values equal live; four threads racing on the first reads of a fresh model all get the right values (the Cell's
first-fill discipline, mv-race), on a new model per run.
A clear concurrent with evaluation is outside the contract (change an input, clear, run again: D191).
"""
import subprocess
import sys
import textwrap

import modelx as mx


def _build(tmp_path):
    model = mx.new_model("HeldArrays")
    try:
        data = model.new_space("Data")
        data.np = __import__("numpy")
        data.new_cells("rates", "def rates():\n    return np.linspace(0.0, 1.0, 101)\n")
        data.new_cells("grid", "def grid():\n    return np.arange(12.0).reshape(3, 4)\n")
        proj = model.new_space("Projection", formula=lambda i: None)
        proj.data = data
        proj.new_cells("step", "def step(t):\n    return data.rates()[t] * i\n")
        proj.new_cells("corner", "def corner(k):\n    return data.grid()[2, 3] + data.grid()[i % 3, k]\n")
        proj.new_cells("head", "def head():\n    return data.rates()[0] + 1.0\n")      # once per item: not held
        proj.new_cells("total", "def total():\n    s = 0.0\n    for t in range(101):\n"
                                "        s += step(t)\n    return s + corner(1) + head()\n")
        expected = {i: float(model.Projection[i].total()) for i in range(1, 9)}
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nfor i in (1, 2):\n    mx_model.Projection[i].total()\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py", "--no-spec"],
                   cwd=tmp_path, check=True)
    return expected


def test_held_array_reads_values_and_generated_code(tmp_path):
    expected = _build(tmp_path)
    source = (tmp_path / "exported_cy/_mx_classes.py").read_text(encoding="utf-8")
    assert "_mx_held__c_Projection_mx_ta_" in source
    assert "data.rates()[t]" not in source and "data.grid()[2, 3]" not in source, "subscripts not held"
    assert "data.rates()[0]" in source, "a zero-argument Cell runs once per item: its read stays as written"
    probe = textwrap.dedent(f"""\
        from exported_cy import mx_model as m
        expected = {expected!r}
        got = {{i: m.Projection[i].total() for i in expected}}
        assert got == expected, (got, expected)
        print("ok")
        """)
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stdout + result.stderr


def test_held_array_reads_four_threads_on_fresh_models(tmp_path):
    expected = _build(tmp_path)
    probe = textwrap.dedent(f"""\
        import threading
        from exported_cy import mx_model as m

        expected = {expected!r}
        errors = []
        for run in range(400):
            fresh = type(m)()           # a new model per run: the first reads of each fill and hold the array
            barrier = threading.Barrier(4, timeout=30)
            def reader(k):
                barrier.wait()
                try:
                    for _ in range(5):
                        for i in range(1 + k, 9, 4):
                            value = fresh.Projection[i].total()
                            if value != expected[i]:
                                errors.append(f"wrong value {{i}}: {{value}}")
                            del fresh.Projection[i]
                except Exception as error:
                    errors.append(f"{{type(error).__name__}}: {{error}}")
            threads = [threading.Thread(target=reader, args=(k,)) for k in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert not errors, sorted(set(errors))[:5]
        print("ok")
        """)
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, capture_output=True, text=True,
                            timeout=1800)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stdout[-2000:] + result.stderr[-2000:]
