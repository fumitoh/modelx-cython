"""A compiled cell holding an admitted IntEnum gives Python the live member (D103)."""

import subprocess
import sys
import textwrap


def test_enum_cell_results_are_members_where_they_reach_python(tmp_path):
    (tmp_path / "phases.py").write_text(textwrap.dedent("""\
        from enum import IntEnum
        class Phase(IntEnum):
            ACC = 0
            WDR = 1
            DEP = 2
        """), encoding="utf-8")
    (tmp_path / "enum_model.py").write_text(textwrap.dedent("""\
        import modelx as mx, pandas as pd
        from phases import Phase
        IDS = (1, 2, 3, 4)
        def build():
            m = mx.new_model("Enums")
            p = m.new_space("Projection", formula="lambda policy_id: None",
                            refs={"Phase": Phase, "pd": pd, "ids": IDS,
                                  "rates": {(0, 0): 1.0, (1, 0): 2.0, (2, 0): 3.0}}, collect_over={"policy_id": IDS},
                            collect_outputs='lambda: {"sum": {"pv": pv()}}')
            p.new_cells("phase", "def phase(t):\\n    return Phase.ACC if t < 3 * policy_id else "
                                 "(Phase.WDR if t < 9 else Phase.DEP)\\n")
            p.new_cells("cf", "def cf(t):\\n    return 10.0 if phase(t) == Phase.WDR else "
                              "(1.0 if phase(t) == Phase.ACC else 0.0)\\n")
            p.new_cells("pv", "def pv():\\n    return sum(cf(t) for t in range(12))\\n")
            p.new_cells("report", "def report():\\n    return pd.DataFrame({'phase': [phase(t).name for t in range(12)]})\\n")
            p.new_cells("phases", "def phases():\\n    return [phase(t) for t in range(12)]\\n")
            p.new_cells("rate", "def rate(t):\\n    key = (phase(t), 0)\\n    return rates[key]\\n")
            p.new_cells("pair", "def pair(t):\\n    return (phase(t), t)\\n")
            return m
        """), encoding="utf-8")
    (tmp_path / "build_model.py").write_text(
        "from enum_model import build\nm = build()\nm.export('enums_nomx')\nm.close()\n", encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from enums_nomx import mx_model\nfor k in (1, 2, 3, 4):\n"
        "    p = mx_model.Projection[k]; p.collect_outputs(); p.report(); p.phases(); p.phase(0)\n"
        "    [p.rate(t) for t in range(12)]; [p.pair(t) for t in range(12)]\n"
        "mx_model.Projection.collect()\n", encoding="utf-8")
    (tmp_path / "spec.py").write_text(
        repr({"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}), encoding="utf-8")
    (tmp_path / "checks.py").write_text(textwrap.dedent("""\
        from importlib import import_module
        from importlib.machinery import ExtensionFileLoader
        from enum_model import build, IDS
        from phases import Phase
        assert isinstance(import_module("enums_nomx_cy._mx_classes").__loader__, ExtensionFileLoader)
        compiled = type(import_module("enums_nomx_cy").mx_model)()
        live = build()
        for i in IDS:
            for t in range(12):
                c, l = compiled.Projection[i].phase(t), live.Projection[i].phase(t)
                assert c is l and type(c) is Phase, (i, t, c, l)
            assert compiled.Projection[i].report().equals(live.Projection[i].report())
            assert all(a is b for a, b in zip(compiled.Projection[i].phases(), live.Projection[i].phases()))
            for t in range(12):
                c, l = compiled.Projection[i].pair(t), live.Projection[i].pair(t)
                assert c[0] is l[0] and c[1] == l[1], (c, l)      # a returned tuple holds the member
                assert compiled.Projection[i].rate(t) == live.Projection[i].rate(t)
        assert compiled.Projection[1].phase(0).name == "ACC" and compiled.Projection[1].phase(10).value == 2
        total = sum(live.Projection[i].pv() for i in IDS)
        for workers in (1, 4):
            m = type(compiled)()
            m.collect_workers = workers
            assert m.Projection.collect().sum.pv == total
        live.close()
        print("ok")
        """), encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=600)
    run("build_model.py")
    run("-m", "modelx_cython", "enums_nomx", "--sample", "sample.py", "--spec", "spec.py")
    run("checks.py")
