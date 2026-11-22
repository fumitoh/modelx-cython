"""D154: every exit by which a vector cell's value reaches Python code gives the list live gives.

Compiled code passes the typed vector between compiled cells without a copy; wherever the value reaches
Python (a loop or builtin left on Python objects, a cell that is not itself a vector cell, a dict or tuple
value, a call through another Space's Reference, a keyword call, a closure, collect's ``individual``),
it is a new list, with the values, element types and Python behaviour of the exported model.
"""
import subprocess
import sys
import textwrap

import modelx as mx

PROJECTION = {
    'keys': 'lambda: [1, 2, 3]',
    'fv': 'lambda: [0.5 * policy_id, 1.5, 2.5]',
    'pairs': 'lambda: ([1.0, 2.0], [3.0, 4.0])',
    'kw': 'lambda n=2: [float(i) for i in range(n)]',
    # not vector cells themselves: the vector reaches Python through their results
    'maybe': 'lambda flag: fv() if flag else None',
    'as_dict': 'lambda: {"values": fv(), "n": len(fv())}',
    'as_tuple': 'lambda: (fv(), 1.0)',
    'first_pair': 'lambda: pairs()[0]',
    'via_local': textwrap.dedent('''\
        def via_local(flag):
            x = fv() if flag else keys()
            return sorted(x, reverse=True)
        '''),
    'closure': textwrap.dedent('''\
        def closure():
            scale = 2.0
            return list(map(lambda v: v * scale, fv()))
        '''),
    'kw_caller': 'lambda: sum(kw(n=3))',
    'native_sum': 'lambda: sum(fv()) + sum(pairs()[1])',
}
PORTFOLIO = {
    'loop': textwrap.dedent('''\
        def loop():
            total = 0.0
            for key in projection.keys():
                total += sum(projection[key].fv())
            return total
        '''),
    'listed': 'lambda: list(projection.keys())',
    'indexed': 'lambda: projection[2].fv()[1] + len(projection.keys())',
    'held': 'lambda: projection.keys()',
}

PROBE = '''\
import sys
from importlib import import_module
m = import_module(sys.argv[1]).mx_model
p, q = m.Projection[2], m.Portfolio
values = {
    'keys': p.keys(), 'fv': p.fv(), 'pairs': p.pairs(), 'kw': p.kw(), 'kw3': p.kw(n=3),
    'maybe': p.maybe(True), 'maybe_none': p.maybe(False), 'as_dict': p.as_dict(), 'as_tuple': p.as_tuple(),
    'first_pair': p.first_pair(), 'via_local': p.via_local(True), 'via_local_int': p.via_local(False),
    'closure': p.closure(), 'kw_caller': p.kw_caller(), 'native_sum': p.native_sum(),
    'loop': q.loop(), 'listed': q.listed(), 'indexed': q.indexed(), 'held': q.held(),
}
def kinds(value):
    if isinstance(value, dict):
        return {k: kinds(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [kinds(v) for v in value])
    return type(value).__name__
print(repr({k: (v, kinds(v)) for k, v in values.items()}))
# a vector cell's list never aliases its typed cache, whatever the exit; a cell that is not a vector cell
# caches its own Python value as live does, but changing it never reaches the vector cell it came from
if sys.argv[1] == 'exported_cy':
    for name, got in (('fv', p.fv), ('first_pair', p.first_pair), ('held', q.held), ('kw', p.kw)):
        first = got()
        first.append(99)
        assert 99 not in got(), name
    for value in (p.maybe(True), p.as_dict()['values'], p.as_tuple()[0]):
        value.append(99)
        assert 99 not in p.fv()
'''


def test_vector_python_exits(tmp_path):
    model = mx.new_model('VectorExits')
    try:
        projection = model.new_space('Projection', formula=lambda policy_id: None)
        for name, formula in PROJECTION.items():
            projection.new_cells(name, formula)
        portfolio = model.new_space('Portfolio', refs={'projection': projection})
        for name, formula in PORTFOLIO.items():
            portfolio.new_cells(name, formula)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(textwrap.dedent('''\
        from exported import mx_model as m
        for k in (1, 2, 3):
            p = m.Projection[k]
            p.keys(); p.fv(); p.pairs(); p.kw(); p.maybe(True); p.maybe(False); p.as_dict(); p.as_tuple()
            p.first_pair(); p.via_local(True); p.via_local(False); p.closure(); p.kw_caller(); p.native_sum()
        q = m.Portfolio
        q.loop(); q.listed(); q.indexed(); q.held()
        '''), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    (tmp_path / 'probe.py').write_text(PROBE, encoding='utf-8')
    out = {}
    for package in ('exported', 'exported_cy'):
        result = subprocess.run([sys.executable, 'probe.py', package], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, package + result.stdout + result.stderr
        out[package] = result.stdout
    assert out['exported_cy'] == out['exported'], (out['exported'], out['exported_cy'])
