"""One frame read natively and on pandas at once (T-5c): every read agrees with live, for several input frames.

The frame cell ``Data.points`` is read by formulas whose reads the table layer lowers (native) and by formulas it
leaves to pandas (a boolean mask, ``query``, a column reduction), in three arrangements:
(a) inside one formula, (b) across formulas of one Space, (c) from two Spaces (an item Space, Projection, and a
Space without parameters, Report, that Projection reads). The build report must show both kinds on the same frame.

Every value is compared with live modelx, for each input frame, bound through the Reference ``Data.raw`` on new
models before they run (a compiled model has no clear and no table-cell assignment, as upstream): the traced
frame; another frame of the traced kinds; a float index for the traced integer one (its native reads run on
pandas); a third frame of the traced kinds. Each stage compares the direct values of every cell, and ``collect``
at 1 and 4 workers.

Tiers: the default suite runs the fixed cases below. ``-m fuzz_full`` adds formulas that combine a random native
read with a random pandas read of the frame, on the seed MX_CY_FUZZ_SEED or a random one (printed and named in a
failure).
"""
import json
import os
import random
import subprocess
import sys
import textwrap

import pytest

NATIVE = {      # reads the table layer lowers (checked against the build report)
    'own_sa': 'T.loc[policy_id, "sa"]',
    'own_rate': 'T.at[policy_id, "rate"]',
    'last_id': 'float(T.index.max())',
    'count': 'float(len(T.index))',
    'member': '(1.0 if policy_id + 1 in T.index else 0.0)',
    'walk': 'sum(T.loc[k, "rate"] for k in T.index)',
}
PANDAS = {      # reads left to pandas
    'mask': 'float(T[T["grp"] == "a"]["sa"].sum())',
    'query': 'float(T.query("rate > 0.015")["sa"].sum())',
    'mean': 'float(T["sa"].mean())',
    'product': 'float((T["sa"] * T["rate"]).sum())',
    'loc_mask': 'float(T.loc[T["sa"] > 150, "rate"].sum())',
}

# (a) one formula, (b) one Space, (c) two Spaces: Projection (items) and Report (no parameters)
PROJECTION = {
    'sa': 'T = data.points()\nreturn T.loc[policy_id, "sa"]',
    'grp_total': 'T = data.points()\nreturn float(T[T["grp"] == "a"]["sa"].sum())',
    'mixed_one': ('T = data.points()\nown = T.loc[policy_id, "rate"]\npeers = T[T["grp"] == "a"]\n'
                  'return own * len(peers) + float(peers["sa"].sum()) * 0.001'),
    'loc_mask': 'T = data.points()\nreturn float(T.loc[T["sa"] > 150, "rate"].sum())',
    'share': 'return sa() / report.total_sa()',
    'pv': ('return sum((sa() + mixed_one() + 0.01 * grp_total() + share()) * (1.0 + report.first_rate()) ** -t\n'
           '           for t in range(6))'),
}
REPORT = {
    'total_sa': 'T = data.points()\nreturn float(T["sa"].sum())',
    'first_rate': 'T = data.points()\nreturn T.loc[T.index[0], "rate"]',
    'mixed_r': 'T = data.points()\nreturn T.loc[2, "sa"] + float(T.query("sa > 150")["rate"].sum())',
}
# per fixed formula: the read the build report must list as native, and a read of the same frame it must not lower
# (the report lists the reads the layer recognises; a mask or query it does not match is absent from it)
EXPECTED = {
    'Projection.mixed_one': ('T.loc[self.policy_id, "rate"]', 'T["grp"]'),
    'Projection.sa': ('T.loc[self.policy_id, "sa"]', None),
    'Projection.grp_total': (None, 'T["grp"]'),
    'Projection.loc_mask': (None, 'T["sa"] > 150'),
    'Report.first_rate': ('T.loc[T.index[0], "rate"]', None),
    'Report.mixed_r': ('T.loc[2, "sa"]', 'T.query('),
}

MODEL = '''\
import modelx as mx
import pandas as pd

FRAME = pd.DataFrame({{"sa": [100.0, 200.0, 300.0, 400.0], "rate": [0.01, 0.02, 0.03, 0.04],
                      "grp": ["a", "b", "a", "b"]}}, index=pd.Index([1, 2, 3, 4], name="policy_id"))
IDS = (1, 2, 3, 4)
PROJECTION = {projection!r}
REPORT = {report!r}


def _cell(space, name, body, params=""):
    space.new_cells(name, f"def {{name}}({{params}}):\\n" + "".join("    " + line + "\\n" for line in body.split("\\n")))


def build():
    model = mx.new_model("Mixed")
    data = model.new_space("Data")
    data.raw = FRAME
    data.new_cells("points", "def points():\\n    return raw.copy()\\n")
    report = model.new_space("Report", refs={{"data": data}})
    for name, body in REPORT.items():
        _cell(report, name, body)
    outputs = ", ".join(f'"{{n}}": {{n}}()' for n in PROJECTION)
    proj = model.new_space("Projection", formula="lambda policy_id: None", refs={{"data": data, "report": report, "ids": IDS}},
                           collect_over={{"policy_id": IDS}}, collect_outputs="lambda: {{'sum': {{" + outputs + "}}}}")
    for name, body in PROJECTION.items():
        _cell(proj, name, body)
    return model
'''

CHECKS = '''\
import json, pkgutil, re, sys
from importlib import import_module
import pandas as pd
from mixed_model import FRAME, IDS, PROJECTION, REPORT, build

name = sys.argv[1]
package = import_module(name)
Model = type(package.mx_model)
helper = next(import_module(f"{{name}}.{{info.name}}") for info in pkgutil.iter_modules(package.__path__)
              if info.name.startswith("_mx_helper_") and hasattr(import_module(f"{{name}}.{{info.name}}"), "native_slots"))


def outcome(call):
    try:
        value = call()
        return ("ok", round(float(value), 9))
    except Exception as error:        # live wraps the formula's error: compare the original type
        text = type(error).__name__
        if text == "FormulaError":
            found = re.search(r"\\n(\\w+): ", str(error))
            text = found.group(1) if found else text
        return ("raise", text)


def values(m):
    out = {{}}
    for i in IDS:
        for cell in PROJECTION:
            out[f"Projection[{{i}}].{{cell}}"] = outcome(lambda: getattr(m.Projection[i], cell)())
    for cell in REPORT:
        out[f"Report.{{cell}}"] = outcome(lambda: getattr(m.Report, cell)())
    return out


def sums(direct):
    return {{cell: round(sum(direct[f"Projection[{{i}}].{{cell}}"][1] for i in IDS), 6) for cell in PROJECTION}}


def collected(m, workers):
    m._mx_collect_set_workers_v2(workers)
    result = m.Projection.collect().sum
    return {{cell: round(float(getattr(result, cell)), 6) for cell in PROJECTION}}


fit2 = FRAME.assign(sa=FRAME["sa"] * 2, rate=FRAME["rate"][::-1].values, grp=["b", "a", "a", "a"])
STAGES = [      # (stage, the input frame, whether the compiled reads of its frame stay native)
    ("the traced frame", FRAME, True),
    ("another frame of the traced kinds", fit2, True),
    ("a float index for the traced integer one", fit2.set_axis(pd.Index([1.0, 2.0, 3.0, 4.0], name="policy_id")),
     False),
    ("a third frame of the traced kinds", FRAME.assign(sa=FRAME["sa"] + 50.0), True),
]
fit = helper.native_slots(Model().Data.points())
assert fit > 0, fit
failures = []
for stage, frame, native in STAGES:
    def fresh():
        compiled, live = Model(), build()       # the input is bound on new models before they run
        compiled.Data.raw = frame.copy()
        live.Data.raw = frame.copy()
        return compiled, live
    compiled, live = fresh()
    slots = helper.native_slots(compiled.Data.points())
    if (slots == fit) != native:
        failures.append(f"{{stage}}: native slots {{slots}}, {{fit}} on the traced frame")
    want = values(live)
    got = values(compiled)
    failures += [f"{{stage}}: {{k}}: compiled {{got[k]}}, live {{want[k]}}" for k in want if got[k] != want[k]]
    if all(v[0] == "ok" for v in want.values()):
        expected = sums(want)
        for workers in (1, 4):
            compiled, _ = fresh()
            found = collected(compiled, workers)
            failures += [f"{{stage}}: collect at {{workers}} workers: {{c}}: {{found[c]}}, live {{expected[c]}}"
                         for c in expected if abs(found[c] - expected[c]) > 1e-6 * max(1.0, abs(expected[c]))]
print(json.dumps(failures))
'''


def _build(tmp_path, projection, report):
    (tmp_path / 'mixed_model.py').write_text(MODEL.format(projection=projection, report=report), encoding='utf-8')
    (tmp_path / 'build_model.py').write_text(textwrap.dedent('''\
        from mixed_model import build
        m = build()
        m.export("mixed_nomx", locked_spaces=["Data"])
        m.close()
        '''), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from mixed_nomx import mx_model\nfor k in (1, 2, 3, 4):\n    mx_model.Projection[k].collect_outputs()\n'
        'mx_model.Projection.collect()\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'parallel_workers': 1, 'strict_lowering': True}), encoding='utf-8')
    (tmp_path / 'checks.py').write_text(CHECKS.format(), encoding='utf-8')
    def run(*args):
        done = subprocess.run([sys.executable, *args], cwd=tmp_path, timeout=1200, capture_output=True, text=True)
        assert done.returncode == 0, f'{args}\n' + done.stdout[-4000:] + done.stderr[-8000:]
        return done
    run('build_model.py')
    run('-m', 'modelx_cython', 'mixed_nomx', '--sample', 'sample.py', '--spec', 'spec.py')
    reads = json.loads((tmp_path / 'mixed_nomx_cy' / '_mx_table_reads.json').read_text(encoding='utf-8'))
    failures = json.loads(run('checks.py', 'mixed_nomx_cy').stdout.strip().splitlines()[-1])
    return reads, failures


def _entries(reads, cell):
    return [e for c, es in reads.items() if c.endswith(cell) and isinstance(es, list) for e in es if isinstance(e, dict)]


def _native(reads, cell, text):
    return any(text in e.get('expression', '') and e.get('form') != 'pandas' for e in _entries(reads, cell))


def test_mixed_native_and_pandas_reads_of_one_frame(tmp_path):
    reads, failures = _build(tmp_path, PROJECTION, REPORT)
    missing = []
    for cell, (native, pandas) in EXPECTED.items():
        if native and not _native(reads, cell, native):
            missing.append(f'{cell}: {native} not native: {_entries(reads, cell)}')
        if pandas and _native(reads, cell, pandas):
            missing.append(f'{cell}: {pandas} lowered, expected on pandas: {_entries(reads, cell)}')
    assert not missing, '\n'.join(missing)
    assert not failures, '\n'.join(failures[:40])


def _full_seed():
    seed = int(os.environ.get('MX_CY_FUZZ_SEED') or random.SystemRandom().randrange(1, 10 ** 9))
    print(f'mixed-reads fuzz seed {seed} (set MX_CY_FUZZ_SEED to repeat)')
    return seed


def fuzz_cells(seed, count=12):
    """``count`` formulas, each one combining a native read and a pandas read of the frame, split between the two
    Spaces; Projection also reads a Report cell (two Spaces)."""
    rng = random.Random(seed)
    projection, report = dict(PROJECTION), dict(REPORT)
    for n in range(count):
        native, pandas = rng.choice(sorted(NATIVE)), rng.choice(sorted(PANDAS))
        a, b = rng.choice((0.5, 1.0, -0.25, 2.0)), rng.choice((0.001, 0.01, -0.002))
        body = f'T = data.points()\nreturn {a} * {NATIVE[native]} + {b} * {PANDAS[pandas]}'
        if n % 3 == 2:
            report[f'f{n}'] = body.replace('policy_id', '2')
        else:
            projection[f'f{n}'] = body + (f' + report.f{n - 1}()' if f'f{n - 1}' in report else '')
    return projection, report


@pytest.mark.fuzz_full
def test_fuzz_full_mixed_reads(tmp_path):
    seed = _full_seed()
    reads, failures = _build(tmp_path, *fuzz_cells(seed))
    assert not failures, f'seed {seed}\n' + '\n'.join(failures[:40])
