"""Free-threaded stress of the published prepared tables (T-4.2, T-5c), marked ``stress``: left out of the default run.

What is supported, and what this checks:
- **Four-worker collects on a new model each round.** Items read the published table of a Data frame cell without
  a lock; on a new model its first readers bind and publish it while the other workers read it. Every collect
  returns the result of the one-worker run, MX_CY_STRESS_RUNS rounds (200).
- **One read of the published field** (``_mx_prepared_<name>``): the fast path reads the field once and takes the
  locked path when it is None. Checked on the generated source.

A compiled model has no clear and no table-cell assignment (as upstream modelx-cython), so no input changes while a
collect runs.
"""
import os
import subprocess
import sys
import textwrap

import pytest

RUNS = int(os.environ.get('MX_CY_STRESS_RUNS', '200'))

MODEL = """\
import modelx as mx
import pandas as pd

FRAME = pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0], "rate": [0.01, 0.02, 0.03, 0.04]},
                     index=pd.Index([1, 2, 3, 4], name="policy_id"))
IDS = (1, 2, 3, 4)


def build():
    model = mx.new_model("Stress")
    data = model.new_space("Data", refs={"pd": pd})
    data.FRAME = FRAME
    data.new_cells("points", "def points():\\n    return FRAME.copy()\\n")
    proj = model.new_space("Projection", formula="lambda policy_id: None", refs={"data": data, "ids": IDS},
                           collect_over={"policy_id": IDS}, collect_outputs='lambda: {"sum": {"pv": pv()}}')
    proj.new_cells("sa", 'lambda: data.points().loc[policy_id, "sa"]')
    proj.new_cells("rate", 'lambda t: data.points().loc[policy_id, "rate"] * (1 + t % 2)')
    proj.new_cells("pv", "lambda: sum(sa() * (1.0 + rate(t)) ** t for t in range(24))")
    return model
"""

CHECKS = """\
import math, sys
from importlib import import_module

m = import_module("stress_nomx_cy").mx_model
Model = type(m)
m._mx_collect_set_workers_v2(1)
base = m.Projection.collect().sum.pv
results = []
for run in range(int(sys.argv[1])):
    fresh = Model()                 # a new model: the first readers publish the table while the others read it
    fresh._mx_collect_set_workers_v2(4)
    results.append(fresh.Projection.collect().sum.pv)
bad = [r for r in results if not math.isclose(r, base, rel_tol=1e-12)]
assert results and not bad, (base, bad[:3])
print("ok", len(results), "collects")
"""


@pytest.mark.stress
def test_four_worker_collects_on_new_models(tmp_path):
    (tmp_path / 'stress_model.py').write_text(MODEL, encoding='utf-8')
    (tmp_path / 'build_model.py').write_text(textwrap.dedent("""\
        from stress_model import build
        m = build()
        m.export("stress_nomx", locked_spaces=["Data"])
        m.close()
        """), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        "from stress_nomx import mx_model\nfor k in (1, 2, 3, 4):\n    mx_model.Projection[k].collect_outputs()\n"
        "mx_model.Projection.collect()\n", encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'parallel_workers': 1}), encoding='utf-8')
    (tmp_path / 'checks.py').write_text(CHECKS, encoding='utf-8')
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, check=True, timeout=2400,
                                    capture_output=True, text=True)
    run('build_model.py')
    run('-m', 'modelx_cython', 'stress_nomx', '--sample', 'sample.py', '--spec', 'spec.py')
    generated = (tmp_path / 'stress_nomx_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    # the reads of the items go through the published table of Data.points, read once per call of the accessor
    assert 'self.data._mx_prepared_points()' in generated
    accessor = generated[generated.index('def _mx_prepared_points(self)'):][:700]
    assert 'table = self._mx_ptable_points' in accessor and 'if table is not None:' in accessor, accessor
    out = run('checks.py', str(RUNS))
    assert out.stdout.startswith('ok'), out.stdout + out.stderr
