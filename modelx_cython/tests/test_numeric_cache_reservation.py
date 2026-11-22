"""Exercise stable reservations through ordinary compiled ItemSpace Cells."""

import os
import subprocess
import sys
import textwrap

import modelx as mx


def test_numeric_reservation_survives_reentry_and_spill(tmp_path):
    model = mx.new_model("NumericReservation")
    try:
        space = model.new_space("Policies", formula="lambda point: None",
                                refs={"callback": max})
        space.new_cells("amount", "lambda t, base: callback(t, base)")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\n"
        "m.Policies[1].amount(0, 0.5)\n"
        "m.Policies[1].amount(-2, 0.5)\n"
        "m.Policies[1].amount(42, 0.5)\n",
        encoding="utf-8",
    )
    (tmp_path / "spec.py").write_text(
        repr({"spaces": {"Policies": {"cells": {
            "amount": {"numeric_cache_capacity": 4},
        }}}}),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-m", "modelx_cython", "exported",
         "--sample", "sample.py", "--spec", "spec.py"],
        cwd=tmp_path, env=dict(os.environ), check=True,
    )
    probe = textwrap.dedent('''\
        from math import copysign, isnan
        from pathlib import Path
        from exported_cy import mx_model as m

        source = Path('exported_cy/_mx_classes.py').read_text(encoding='utf-8')
        assert '.NumericCache(4)' in source and '.reserve(' in source
        p = m.Policies[1]
        calls = []
        failure = ValueError('retry')
        def callback(t, base):
            calls.append(t)
            if t == -2 and calls.count(t) == 1:
                raise failure
            if t == -1:
                for key in range(40):
                    assert p.amount(key, 0.25) == 0.25
            return base
        p.callback = callback
        try:
            p.amount(-2, 2.0)
        except ValueError as error:
            assert error is failure
        else:
            raise AssertionError('missing exception')
        assert p.amount(-1, 1.0) == p.amount(-1, 1.0) == 1.0
        assert p.amount(-2, 2.0) == p.amount(-2, 2.0) == 2.0
        for key in range(40):
            assert p.amount(key, 0.25) == 0.25
        assert calls == [-2, -1, *range(40), -2]
        assert copysign(1.0, p.amount(41, -0.0)) == -1.0
        assert copysign(1.0, p.amount(41, 0.0)) == -1.0
        nan = float('nan')
        assert isnan(p.amount(42, nan)) and isnan(p.amount(42, nan))
        assert calls[-3:] == [41, 42, 42]
        del m.Policies[1]
        q = m.Policies[1]
        q.callback = lambda t, base: 7.0
        assert q.amount(-1, 1.0) == 7.0
        assert p.amount(-1, 1.0) == 1.0
    ''')
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path,
                   env=dict(os.environ), check=True)
