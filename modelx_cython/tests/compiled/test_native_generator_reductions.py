"""W1.4: built-in sum/min/max/all/any over one iterable, lowered to native loops.

Live (exported) and compiled results are compared value for value, including the
type, the sign of zero, nan, the exception type and message. Sources: range,
literal tuples (enum members unrolled), tuple- and list-valued cells, zip of two,
filters, nested clauses, list comprehensions, direct iterables, starts.
"""
from http import HTTPStatus
import json
import subprocess
import sys
import textwrap

import modelx as mx

FORMULAS = {
    # name: formula text (lambda or def)
    'v': 'lambda i: values[i % len(values)]',
    'n': 'lambda i: ints[i % len(ints)]',
    'pair': 'lambda t: (v(t), v(t + 1), v(t + 2))',
    'weights': 'lambda t: (n(t), n(t + 1), 0.5)',
    'cohort': 'lambda t: [v(t + k) for k in range(4)]',
    'legs': 'lambda t: (Status.OK, Status.CREATED) if t % 2 else (Status.ACCEPTED,)',
    'by_status': 'lambda t, s: v(t) * (s - 199)',
    'range_sum': 'lambda t: sum(v(i) for i in range(t))',
    'range_filtered': 'lambda t: sum(v(i) * 2.0 for i in range(t, t + 6) if v(i) != 0.0)',
    'range_nested': 'lambda t: sum(v(i) * n(j) for i in range(3) for j in range(i, t))',
    'start_float': 'lambda t: sum((v(i) for i in range(t)), -0.0)',
    'start_int': 'lambda t: sum((n(i) for i in range(t)), 10)',
    'int_total': 'lambda t: sum(n(i) for i in range(t))',
    'bool_total': 'lambda t: sum(v(i) > 0.0 for i in range(t))',
    'members': 'lambda t: sum(by_status(t, s) for s in (Status.OK, Status.CREATED, Status.ACCEPTED))',
    'members_if': 'lambda t: sum(by_status(t, s) for s in (Status.OK, Status.CREATED) if s != t + 200)',
    'tuple_cell': 'lambda t: sum(by_status(t, s) for s in legs(t))',
    'zip_cells': 'lambda t: sum(a * b for a, b in zip(pair(t), weights(t)) if a != 0.0)',
    'list_direct': 'lambda t: sum(cohort(t))',
    'tuple_direct': 'lambda t: sum(pair(t), 1)',
    'assigned': 'def assigned(t):\n    total = sum(v(i) for i in range(t))\n    return total * 2.0\n',
    'augmented': 'def augmented(t):\n    total = v(t)\n    total += sum(n(i) for i in range(t))\n    return total\n',
    'listcomp_sum': 'lambda t: sum([v(i) for i in range(t)])',
    'lo': 'lambda t: min(v(i) for i in range(t, t + 5))',
    'hi_int': 'lambda t: max(n(i) for i in range(t, t + 5))',
    'hi_filtered': 'lambda t: max(v(i) for i in range(t, t + 6) if i % 2 == 0)',
    'lo_cell': 'lambda t: min(pair(t))',
    'lo_mixed': 'lambda t: min(x for x in weights(t))',
    'lo_empty': 'lambda t: min(v(i) for i in range(t))',
    'every': 'lambda t: all(v(i) >= -10.0 for i in range(t))',
    'some': 'lambda t: any(v(i) < 0.0 for i in range(t))',
    'every_tuple': 'lambda t: all(by_status(t, s) > 0.0 for s in (Status.OK, Status.CREATED))',
    'some_nested': 'def some_nested(t):\n    found = any(v(i) * n(j) < 0.0 for i in range(t) for j in range(3))\n    return found\n',
    'every_list': 'lambda t: all([v(i) >= 0.0 for i in range(t)])',
    'some_cell': 'lambda t: any(cohort(t))',
    'stopper': 'lambda t: next(iter(())) if t == 5 else 1.5',
    'stop_sum': 'lambda t: sum(stopper(i) for i in range(t))',
    'stop_direct': 'lambda t: max(stopper(i) for i in (t, 1))',
}


def build(tmp_path):
    model = mx.new_model('GeneratorReductions')
    try:
        s = model.new_space('Projection', refs={
            'Status': HTTPStatus,
            'values': (0.1, -0.0, 1e16, 1.0, -1e16, 0.7, 2.5, -3.25, 1e-300, 0.0),
            'ints': (3, -7, 2 ** 62, 2 ** 62, 0, 5)})
        for name, formula in FORMULAS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    names = [name for name in FORMULAS
             if name not in ('v', 'n', 'pair', 'weights', 'cohort', 'legs', 'by_status', 'stopper')]
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'p = m.Projection\n'
        'for t in range(0, 14):\n'
        '    for s in (200, 201, 202): p.by_status(t, s)\n'
        f'    for name in {names!r}:\n'
        '        try: getattr(p, name)(t)\n'
        '        except (ValueError, OverflowError, RuntimeError, StopIteration): pass\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    return names


def test_generator_reductions_match_the_interpreter(tmp_path):
    names = build(tmp_path)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    untyped = {}
    for name in names:
        cell = report[f'exported_cy._mx_classes._c_Projection._f_{name}']
        if (cell['native_expressions'] == 0 and cell['native_reductions'] == 0
                and not (cell.get('vectors') or {}).get('lowered')):
            untyped[name] = cell['expressions_untyped']
    # sums with a start argument are not lowered (R-6); every other form is
    assert set(untyped) == {'start_float', 'start_int', 'tuple_direct'}, untyped
    probe = textwrap.dedent(f'''\
        import math, pickle, sys
        from importlib import import_module
        p = import_module(sys.argv[1]).mx_model.Projection
        def outcome(name, t):
            try:
                value = getattr(p, name)(t)
            except Exception as error:
                return ('raise', type(error).__name__, str(error))
            return (type(value).__name__, repr(value))
        result = {{(name, t): outcome(name, t) for t in range(14) for name in {names!r}}}
        sys.stdout.buffer.write(pickle.dumps(result))
        ''')
    import pickle
    outcomes = {}
    for package in ('exported', 'exported_cy'):
        done = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True,
                              capture_output=True)
        outcomes[package] = pickle.loads(done.stdout)
    differences = {key: (outcomes['exported'][key], outcomes['exported_cy'][key])
                   for key in outcomes['exported'] if outcomes['exported'][key] != outcomes['exported_cy'][key]}
    # A double-returning cell converts an int result at its boundary, as before W1.4:
    # an empty sum is 0 live and 0.0 compiled (D24(2)); min over (int, float) items likewise.
    doubles = {name for name in names
               if report[f'exported_cy._mx_classes._c_Projection._f_{name}']['return_type'] == 'double'}
    accepted = {key for key, (live, compiled) in differences.items()
                if key[0] in doubles and live[0] == 'int' and compiled[0] == 'float'
                and float(live[1]) == float(compiled[1])}
    assert {key: differences[key] for key in differences if key not in accepted} == {}
