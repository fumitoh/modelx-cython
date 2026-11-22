"""W1.9 helper table forms produce the same live, exported and compiled results.

One model holds a cell per form the W1.9 lowering adds; every cell is called for each t,
live, from the exported package and from the compiled one, and the results (values, or the
error type) must agree exactly. The generated module and report record the
Python helper reads.

Forms: a sub-table local also used as a Python table; the index as typed keys (len, min/max,
list/sorted, comprehension and for-statement iteration with the row position, an index
local); float and str labels; tuple unpacking inside max; a key held in a local of unknown
arity; asof of an untyped local; a record field as a key part; both chained selections;
get_level_values(n).min()/max().
"""

import json
import subprocess
import sys

import modelx as mx
import pandas as pd
from modelx_helpers import pdn

T_RANGE = range(-1, 13)

FORMULAS = {
    'escaping_sub': '''\
def escaping_sub(t):
    sub = data.two().loc[1 + t % 2]
    keys = [k for k in sub.index]
    return (len(sub), keys, float(sub.loc[keys[-1], "v"]))
''',
    'index_keys': '''\
def index_keys(t):
    tbl = data.gaps()
    years = tbl.index
    y = min(max(t, int(years.min())), int(years.max()))
    return (len(tbl.index), min(tbl.index), max(tbl.index), list(tbl.index), sorted(tbl.index),
            y in years, float(tbl.loc[tbl.index.asof(y), "v"]))
''',
    'compr_rows': '''\
def compr_rows(t):
    tbl = data.gaps()
    return float(sum(tbl.loc[k, "v"] for k in tbl.index if k <= t))
''',
    'for_rows': '''\
def for_rows(t):
    tbl = data.gaps()
    total = 0.0
    for k in tbl.index:
        if k > t:
            break
        total += float(tbl.loc[k, "v"]) * k
    return total
''',
    'float_labels': '''\
def float_labels(t):
    tab = data.curve()
    pv = [(float(k), float(tab.loc[k, "value"])) for k in tab.index]
    xs = sorted(float(a) for a in tab.index)
    return (pv[t % 4], xs[-1], float(tab.loc[xs[t % 4], "value"]))
''',
    'str_labels': '''\
def str_labels(t):
    tab = data.items()
    total = 0.0
    for item in tab.index:
        row = tab.loc[item]
        if row["n"] != 2:
            continue
        total += float(row["amount"]) * t
    return (total, list(tab.index))
''',
    'tuple_max': '''\
def tuple_max(t):
    return (max(a for a, s in data.pairs().index), min(a for a, s in data.pairs().index if s == t % 2))
''',
    'tuple_whole': '''\
def tuple_whole(t):
    sub = data.pairs()
    return min(k for k in sub.index if sub.loc[k, "v"] >= 2.0 + t % 2)
''',
    'tuple_key': '''\
def tuple_key(t):
    a = anchor(t)
    try:
        return float(data.pairs().loc[a, "v"])
    except KeyError:
        return -1.0
''',
    'untyped_asof': '''\
def untyped_asof(t):
    a = loose(t)
    try:
        return data.gaps().index.asof(a)
    except KeyError:
        return -1
''',
    'record_key': '''\
def record_key():
    return float(data.by_id().loc[point()["fund"], "v"])
''',
    'levels': '''\
def levels(t):
    tab = data.pairs()
    ages = tab.index.get_level_values("a")
    xx = min(max(t, int(ages.min())), int(ages.max()))
    return (xx, int(data.pairs().index.get_level_values("s").max()))
''',
    'chains': '''\
def chains(t):
    try:
        return (float(data.two().loc[1 + t % 3].loc[t % 3, "v"]),
                float(data.two().loc[1 + t % 3]["v"][t % 3]))
    except KeyError:
        return -1.0
''',
}
CALLS = [name for name in FORMULAS if name != 'record_key']
TEXT = ('self.data._mx_cell_two().loc[1 + t % 2]', '.index.asof(a)',
        'self.data._mx_cell_pairs().loc[a, "v"]', '.index.get_level_values("s").max()')
RUN = ('import json\nfrom {package} import mx_model as m\n'
       'P = m.Projection\n'
       'out = [[repr(getattr(P, name)(t)) for name in {calls!r}] for t in {trange!r}]\n'
       'out.append(repr(P.record_key()))\n'
       'print(json.dumps(out))\n')


def _model(root):
    model = mx.new_model('TableFormsW19')
    try:
        data = model.new_space('Data', refs={'pdn': pdn})
        data.two_frame = pd.DataFrame({'a': [1, 1, 2, 2], 'b': [0, 1, 0, 2], 'v': [1.0, 2.0, 3.0, 4.0]})\
            .set_index(['a', 'b'])
        data.gaps_frame = pd.DataFrame({'v': [0.5, 1.5, 2.5, 3.5]}, index=pd.Index([1, 2, 5, 9], name='k'))
        data.curve_frame = pd.DataFrame({'value': [0.0, 0.5, 0.9, 1.0]},
                                        index=pd.Index([0.0, 1.0, 2.5, 4.0], name='key'))
        data.items_frame = pd.DataFrame({'amount': [1.0, 2.0, 3.0], 'n': [2, 1, 2]},
                                        index=pd.Index(['b', 'a', 'c'], name='item'))
        data.pairs_frame = pd.DataFrame({'a': [3, 1, 2, 1], 's': [0, 1, 0, 0], 'v': [1.0, 2.0, 3.0, 4.0]})\
            .set_index(['a', 's'])
        data.by_id_frame = pd.DataFrame({'v': [7.0, 8.0]}, index=pd.Index(['F1', 'F2'], name='fund'))
        data.points_frame = pd.DataFrame({'fund': ['F2'], 'age': [40]}, index=pd.Index([1], name='point'))
        for name in ('two', 'gaps', 'curve', 'items', 'pairs', 'by_id', 'points'):
            data.new_cells(name, f'lambda: pdn.table({name}_frame)')
        projection = model.new_space('Projection', refs={'data': data})
        projection.new_cells('anchor', 'lambda t: ((1, 1), (2, 0), (3, 0), 1, (9, 9), (1,))[t % 6]')
        projection.new_cells('loose', 'lambda t: (1, 2, 5, 9, 3, 1.5, 0)[t % 7]')
        projection.new_cells('point', 'lambda: data.points().loc[1]')
        for formula in FORMULAS.values():
            projection.new_cells(formula=formula)
        model.export(root / 'exported')
        live = [[repr(projection.cells[name](t)) for name in CALLS] for t in T_RANGE]
        live.append(repr(projection.record_key()))
    finally:
        model.close()
    return live


def test_w1_9_forms_live_exported_compiled(tmp_path):
    live = _model(tmp_path)
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nP = m.Projection\n'
        f'for t in {T_RANGE!r}:\n' + ''.join(f'    P.{name}(t)\n' for name in CALLS)
        + 'P.record_key()\n', encoding='utf-8')
    for package in ('exported', 'exported_cy'):
        if package == 'exported_cy':
            done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                                   '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
            assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        out = subprocess.run([sys.executable, '-c', RUN.format(package=package, calls=CALLS,
                                                                  trange=list(T_RANGE))],
                             cwd=tmp_path, capture_output=True, text=True)
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
        got = json.loads(out.stdout.strip().splitlines()[-1])
        assert got == live, (package, [(i, a, b) for i, (a, b) in enumerate(zip(got, live)) if a != b][:3])
    text = '\n'.join(p.read_text(encoding='utf-8') for p in (tmp_path / 'exported_cy').rglob('_mx_classes.py'))
    for fragment in TEXT:
        assert fragment in text, fragment
    report = '\n'.join(p.read_text(encoding='utf-8') for p in (tmp_path / 'exported_cy').rglob('_mx_admission.json'))
    cells = json.loads(report)
    inputs = [item for name, cell in cells.items() if '._c_Data.' in name
              for item in cell.get('table_reads_untyped') or ()]
    readers = {name.rsplit('._f_', 1)[-1]: cell['table_reads_untyped']
               for name, cell in cells.items() if '._c_Projection.' in name and cell.get('table_reads_untyped')}
    assert len(inputs) == 7
    assert all(item['reason'].startswith('receiver is a pandas') and not item['per_step']
               for item in inputs), inputs
    for name in ('escaping_sub', 'tuple_key', 'untyped_asof', 'levels', 'chains'):
        assert all(item['reason'] == 'the receiver is a helper Table and the formula uses its Python methods'
                   for item in readers[name]), readers[name]
