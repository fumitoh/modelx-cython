"""Round-2 key vocabulary on a table bound by observed admission stays native (D69).

With observed-table admission (W1.5 0004) a Data cell built by `pdn.table` is bound to a
`NativeTable` (Python sees its read-only DataFrame, D101). The round-2 provider calls `keys(pdn.keys, (), table)`, `key_count` and
`key_at` took their native path only for a helper `Table` and otherwise called the Python
helper on every evaluation (W1.5 hand-back: up to +30% on 18 models). They now unwrap the
bound table to its helper owner. Checked: the Data cell is bound, values equal the
interpreter, and the helper's Python `keys` never runs while the compiled cells evaluate.
"""

import json
import subprocess
import sys

import modelx as mx
import pandas as pd
from modelx_helpers import pdn

RUN = '''
import json, sys
from exported_cy import mx_model as m
from modelx_helpers import table_keys, pdn
codes = {table_keys.keys.__code__}
for name in ('key_count', 'key_at'):
    function = getattr(pdn, name, None)
    if getattr(function, '__code__', None) is not None:
        codes.add(function.__code__)
called = []
def watch(frame, event, arg):
    if event == 'call' and frame.f_code in codes:
        called.append(frame.f_code.co_name)
m.Data.table()
sys.setprofile(watch)
try:
    out = [m.Projection.last_key(t) for t in range(3)] + [m.Projection.count(t) for t in range(3)]
finally:
    sys.setprofile(None)
import pandas as pd
from modelx_helpers.immutable_inputs import Table
value = m.Data.table()
kind = ('the helper table, as live' if isinstance(value, Table) and value is m.Data.table()
        else repr(type(value)))
print(json.dumps([out, called, kind, pdn.to_pandas(value).equals(m.Data.source)]))
'''


def test_round2_keys_on_a_bound_table_take_the_native_path(tmp_path):
    model = mx.new_model('BoundKeys')
    try:
        data = model.new_space('Data', refs={'pdn': pdn})
        data.source = pd.DataFrame({'v': [1.0, 2.0, 3.0]}, index=pd.Index([10, 20, 30], name='k'))
        data.new_cells('table', 'lambda: pdn.table(source)')
        projection = model.new_space('Projection', refs={'data': data, 'pdn': pdn})
        projection.new_cells('last_key', 'lambda t: max(pdn.keys(data.table())) + t')
        projection.new_cells('count', 'lambda t: len(pdn.keys(data.table())) + t')
        live = [projection.last_key(t) for t in range(3)] + [projection.count(t) for t in range(3)]
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\nfor t in range(3):\n    m.Projection.last_key(t); m.Projection.count(t)\n',
        encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    text = '\n'.join(p.read_text(encoding='utf-8') for p in (tmp_path / 'exported_cy').rglob('_mx_classes.py'))
    out = subprocess.run([sys.executable, '-c', RUN], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    values, called, kind, same = json.loads(out.stdout.strip().splitlines()[-1])
    assert values == live
    # D130: Python gets what the formula returned (the helper Table, as live)
    assert kind == 'the helper table, as live' and same
