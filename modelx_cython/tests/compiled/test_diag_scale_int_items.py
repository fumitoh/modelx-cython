"""diag-scale P3: integer items of a Python tuple without reference traffic.

(a) ``_mx_exact_int64`` reads any int subclass (IntEnum, bool, a user subclass) directly, as
PyNumber_Index returns such an object unchanged: same value, same OverflowError.
(b) A lowered reduction over an object iterable whose loop variable is only passed as a C-integer
argument of compiled Cells reads the items of an exact tuple borrowed, as C integers. An item that
is not an int fitting in 64 bits takes the object path (its error is the one raised today), any
other iterable (a list) the loop as written. A loop variable used as an object keeps the object loop.
Compiled equals the exported model; the compiled errors are the ones of the integer boundary.
"""
import json
import os
import subprocess
import sys
import textwrap

import modelx as mx


CELLS = {
    'legs': '''\
def legs():
    return (Leg.A, Leg.B, Leg.C)
''',
    'as_list': '''\
def as_list():
    return [Leg.C, Leg.A]
''',
    'comp': '''\
def comp(t, leg):
    return 1.5 * leg + t
''',
    'total': '''\
def total(t):
    return sum(comp(t, leg) for leg in legs())
''',
    'any_big': '''\
def any_big(t):
    return any(comp(t, leg) > 7.0 for leg in legs())
''',
    'listed': '''\
def listed(t):
    return sum(comp(t, x) for x in as_list())
''',
    'escapes': '''\
def escapes(t):
    return sum(comp(t, leg) + float(leg) for leg in legs())
''',
    'subclass': '''\
def subclass(t):
    return comp(t, odd) + comp(t, True)
''',
    'mixed_items': '''\
def mixed_items(t):
    return sum(comp(t, v) for v in values)
''',
}


def _model(tmp_path, monkeypatch):
    (tmp_path / 'p3_kinds.py').write_text(
        'import enum\n\n'
        'class Leg(enum.IntEnum):\n    A = 1\n    B = 2\n    C = 3\n\n'
        'class Odd(int):\n    def __index__(self):\n        return 99\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    from p3_kinds import Leg, Odd
    model = mx.new_model('IntItems')
    try:
        s = model.new_space('Projection', formula='lambda policy: None',
                            refs={'Leg': Leg, 'odd': Odd(5), 'values': (1, 2)})
        for name, formula in CELLS.items():
            s.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    calls = ('total', 'any_big', 'listed', 'escapes', 'subclass', 'mixed_items')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'p = m.Projection\n'
        'for t in range(6):\n'
        + ''.join(f'    p.{name}(t)\n' for name in calls), encoding='utf-8')
    return calls


PROBE = textwrap.dedent('''\
    import sys
    from importlib import import_module
    m = import_module(sys.argv[1]).mx_model
    p = m.Projection
    names = sys.argv[2].split(',')
    def row(t):
        out = []
        for name in names:
            try:
                out.append(repr(getattr(p, name)(t)))
            except Exception as error:
                out.append(f'{type(error).__name__}: {error}')
        return out
    if sys.argv[3] != '-':
        # items that do not fit in 64 bits, or are not integers: the integer boundary's errors
        p.values = eval(sys.argv[3])
        rows = [row(1)]
    else:
        rows = [row(t) for t in range(6)]
    print(repr(rows))
    ''')


def _run(tmp_path, calls, package, values):
    env = {**os.environ, 'PYTHONPATH': str(tmp_path) + ':' + os.environ.get('PYTHONPATH', '')}
    run = subprocess.run([sys.executable, '-c', PROBE, package, ','.join(calls), values], cwd=tmp_path,
                         capture_output=True, text=True, env=env)
    assert run.returncode == 0, package + run.stdout + run.stderr
    return run.stdout


def test_integer_items_of_a_tuple(tmp_path, monkeypatch):
    calls = _model(tmp_path, monkeypatch)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    # values: an IntEnum tuple (borrowed loop), a list (object loop), an escaping variable, an int
    # subclass whose __index__ differs (read as its int value, as PyNumber_Index does) and bool
    assert _run(tmp_path, calls, 'exported_cy', '-') == _run(tmp_path, calls, 'exported', '-')

    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    cells = {key.rsplit('._f_', 1)[-1]: value for key, value in report.items()
             if isinstance(value, dict) and '._f_' in key}
    loops = {name: cells[name].get('native_tuple_int_loops', 0) for name in calls}
    assert loops['total'] == 1 and loops['any_big'] == 1 and loops['mixed_items'] == 1, loops
    assert loops['listed'] == 1, loops          # emitted; a list takes the loop as written at run time
    assert loops['escapes'] == 0, loops         # the variable is also used as an object


def test_items_outside_64_bits_or_not_integers_raise_the_boundary_errors(tmp_path, monkeypatch):
    calls = _model(tmp_path, monkeypatch)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    for values, error in (('(1, 2 ** 70)', 'OverflowError'), ("(1, 'x')", 'TypeError'),
                          ('(1, 2.5)', 'TypeError')):
        last = eval(_run(tmp_path, ('mixed_items',), 'exported_cy', values))[-1][0]
        assert last.startswith(error + ': Projection.comp requires an exact signed 64-bit integer'), (values, last)
