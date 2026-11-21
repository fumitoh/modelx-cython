"""W1.4: exhaustive small cases for lowered min/max/all/any/sum, live against compiled.

Mixed objects (ints at the C range ends, bools, -0.0, nan, inf, complex) go through the
object path; float-only tuples through the typed path. Every tuple of length 0-3 (mixed)
and 0-4 (floats) is evaluated by both builds; type, repr and exceptions must agree.
"""
import itertools
import math
import pickle
import random
import subprocess
import sys
import textwrap

import modelx as mx

MIXED = (0, 1, -1, 2 ** 63 - 1, 2 ** 64, 0.0, -0.0, 1.5, math.nan, math.inf, -math.inf, True, False, 1j)
FLOATS = (0.0, -0.0, 1.0, -1.0, 1e16, -1e16, 0.1, math.nan, math.inf, -math.inf)


def cases():
    mixed = [c for n in range(4) for c in itertools.product(MIXED, repeat=n)]
    floats = [c for n in range(5) for c in itertools.product(FLOATS, repeat=n)]
    rng = random.Random(7)
    floats += [tuple(rng.uniform(-1, 1) * 10.0 ** rng.randint(-30, 30) for _ in range(rng.randint(1, 9)))
               for _ in range(5000)]
    return mixed, floats


REDUCTIONS = {
    'sum_obj': 'lambda k: sum(item(k, i) for i in range(count(k)))',
    'min_obj': 'lambda k: min(item(k, i) for i in range(count(k)))',
    'max_obj': 'lambda k: max(item(k, i) for i in range(count(k)))',
    'all_obj': 'lambda k: all(item(k, i) for i in range(count(k)))',
    'any_obj': 'lambda k: any(item(k, i) for i in range(count(k)))',
    'sum_start': 'lambda k: sum((item(k, i) for i in range(count(k))), -0.0)',
    'sum_tuple': 'lambda k: sum(case(k))',
    'max_tuple': 'lambda k: max(case(k))',
    'sum_f64': 'lambda k: sum(fitem(k, i) for i in range(fcount(k)))',
    'min_f64': 'lambda k: min(fitem(k, i) for i in range(fcount(k)))',
    'max_f64': 'lambda k: max(fitem(k, i) for i in range(fcount(k)))',
    'all_f64': 'lambda k: all(fitem(k, i) for i in range(fcount(k)))',
    'any_f64': 'lambda k: any(fitem(k, i) - 1.0 for i in range(fcount(k)))',
}


def test_reduction_cases(tmp_path):
    mixed, floats = cases()
    model = mx.new_model('ReductionCases')
    try:
        s = model.new_space('Projection', refs={'cases': tuple(mixed), 'fcases': tuple(floats)})
        s.new_cells('count', 'lambda k: len(cases[k])')
        s.new_cells('case', 'lambda k: cases[k]')
        s.new_cells('item', 'lambda k, i: cases[k][i]')
        s.new_cells('fcount', 'lambda k: len(fcases[k])')
        s.new_cells('fitem', 'lambda k, i: fcases[k][i]')
        for name, formula in REDUCTIONS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(textwrap.dedent(f'''\
        from exported import mx_model as m
        p = m.Projection
        for k in range({len(mixed)}):
            for i in range(p.count(k)): p.item(k, i)
            for name in ('sum_obj', 'min_obj', 'max_obj', 'all_obj', 'any_obj', 'sum_start', 'sum_tuple', 'max_tuple'):
                try: getattr(p, name)(k)
                except Exception: pass
        for k in range({len(floats)}):
            for i in range(p.fcount(k)): p.fitem(k, i)
            for name in ('sum_f64', 'min_f64', 'max_f64', 'all_f64', 'any_f64'):
                try: getattr(p, name)(k)
                except Exception: pass
        '''), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    probe = textwrap.dedent(f'''\
        import pickle, sys
        from importlib import import_module
        p = import_module(sys.argv[1]).mx_model.Projection
        def outcome(name, k):
            try:
                value = getattr(p, name)(k)
            except Exception as error:
                return ('raise', type(error).__name__, str(error))
            return (type(value).__name__, repr(value))
        result = {{}}
        for name in {list(REDUCTIONS)!r}:
            for k in range({len(floats)} if name.endswith(('f64',)) else {len(mixed)}):
                result[name, k] = outcome(name, k)
        sys.stdout.buffer.write(pickle.dumps(result))
        ''')
    outcomes = {}
    for package in ('exported', 'exported_cy'):
        done = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True,
                              capture_output=True)
        outcomes[package] = pickle.loads(done.stdout)
    import json
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    returns = {name: report[f'exported_cy._mx_classes._c_Projection._f_{name}']['return_type']
               for name in REDUCTIONS}
    for name in REDUCTIONS:
        if name == 'sum_start':     # sum(iterable, start) is not lowered (R-6: no model uses it)
            continue
        entry = report[f'exported_cy._mx_classes._c_Projection._f_{name}']
        assert entry['native_expressions'] + entry['native_reductions'] == 1, name
    differences = {}
    for key, live in outcomes['exported'].items():
        compiled = outcomes['exported_cy'][key]
        if live == compiled:
            continue
        # A double-returning cell converts an int result at its boundary (unchanged by W1.4).
        if returns[key[0]] == 'double' and live[0] == 'int' and compiled == ('float', repr(float(int(live[1])))):
            continue
        differences[key] = (live, compiled)
    assert differences == {}, (len(differences), list(differences.items())[:10])
