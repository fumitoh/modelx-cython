"""Ordered ZIP walks keep the prepared-table binding and the ordinary cold path."""

import json
import subprocess
import sys

import modelx as mx
import pandas as pd


def test_ordered_prefix_uses_only_the_eligible_published_snapshot(tmp_path):
    model = mx.new_model('Prefix')
    try:
        data = model.new_space('Data')
        data.raw = pd.DataFrame({'amount': [1.0, 2.0, 4.0, 8.0]}, index=pd.MultiIndex.from_tuples(
            [('B', 0), ('B', 10), ('A', 0), ('A', 10)]))
        data.points = pd.DataFrame({'prefix': ['A', 'B'], 'prime': [1, 2]}, index=[1, 2])
        data.action = (None,)
        data.new_cells('table', 'lambda: raw')
        data.new_cells('point_table', 'lambda: points')
        calc = model.new_space('Calc', formula='lambda policy_id: None', refs={'data': data})
        calc.new_cells('model_point', 'lambda: data.point_table().loc[policy_id]')
        calc.new_cells('prime', 'lambda: model_point()["prime"]')
        calc.new_cells('selector', 'lambda: model_point()["prefix"]')
        calc.new_cells('effectful', '''def effectful():
    action = data.action[0]
    return "A" if action is None else action(data.table())
''')
        for name, selector, column in (
                ('anchors', 'selector', '"amount"'),
                ('second', 'selector', '"amount"'),
                ('effect', 'effectful', '"amount"'),
                ('tuple_column', 'selector', '"amount",')):
            calc.new_cells(name, f'''def {name}():
    tbl = data.table().loc[{selector}()]
    return [(int(k), float(v)) for k, v in zip(tbl.index, tbl[{column}])]
''')
        model.export(tmp_path / 'exported', locked_spaces=['Data'])
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text('''from exported import mx_model as model
item = model.Calc[1]
item.prime()
item.anchors()
item.second()
item.effect()
try:
    item.tuple_column()
except KeyError:
    pass
''', encoding='utf-8')
    (tmp_path / 'spec.py').write_text("{'strict_lowering': True}", encoding='utf-8')

    def run(*args):
        done = subprocess.run([sys.executable, *args], cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr

    run('-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py')
    compiled = tmp_path / 'exported_cy'
    source = (compiled / '_mx_classes.py').read_text(encoding='utf-8')
    assert source.count('.zip_prepared_prefix(') == 3
    assert 'zip(tbl.index, tbl["amount",])' in source
    manifest = json.loads((compiled / '_mx_table_reads.json').read_text(encoding='utf-8'))
    assert any(entry['form'] == 'L2' for entry in manifest['Calc.selector'])
    assert manifest['Calc.anchors'][0]['expression'] == 'self.data.table().loc[self.selector()]'

    run('-c', '''
from contextlib import contextmanager
import sys

import pandas as pd
from pandas.core.indexing import _LocIndexer
from exported import mx_model as example
from exported_cy import mx_model as compiled


base = example.Data.raw
loc_code = _LocIndexer.__getitem__.__code__


@contextmanager
def count(frame):
    hits = []

    def profile(event_frame, event, argument):
        if event == 'call' and event_frame.f_code is loc_code:
            accessor = event_frame.f_locals['self']
            if type(accessor) is _LocIndexer and accessor.obj is frame:
                hits.append(event_frame.f_locals['key'])

    sys.setprofile(profile)
    try:
        yield hits
    finally:
        sys.setprofile(None)


def fresh(frame, action=None):
    model = type(compiled)()
    model.Data.raw = frame
    model.Data.action = (action,)
    return model


first = base.copy()
first.iat[2, 0] = 9.0  # edit before normal publication
for frame, expected in [(first, [(0, 9.0), (10, 8.0)]), (base, [(0, 4.0), (10, 8.0)])]:
    model = fresh(frame)
    item = model.Calc[1]
    item.prime()  # held positive row; selector itself remains cold
    with count(frame) as hits:
        got = item.anchors()
        other = item.second()  # already cached selector
    assert not hits and got == expected and other == expected and got is not other
    assert all(type(k) is int and type(v) is float for k, v in got)
    assert type(item.model_point()) is pd.Series

cold = fresh(base).Calc[1]
with count(base) as hits:
    assert cold.anchors() == [(0, 4.0), (10, 8.0)]
    assert cold.second() == cold.anchors()
assert hits == ['A']  # the cold selector fills the cache but cannot promote that invocation

selected = []


def edit(frame):
    selected.append(frame)
    frame.iat[2, 0] = 42.0
    return 'A'


frame = base.copy()
model = fresh(frame, edit)
model.Data.table()  # snapshot deliberately predates the edit made by the selector
with count(frame) as hits:
    assert model.Calc[1].effect() == [(0, 42.0), (10, 8.0)]
assert hits == ['A'] and len(selected) == 1 and selected[0] is frame

metadata = base.copy()
model = fresh(metadata)
model.Data.table()
item = model.Calc[1]
item.prime()
metadata.attrs['tag'] = 'current metadata'
with count(metadata) as hits:
    assert item.anchors() == [(0, 4.0), (10, 8.0)] and hits == ['A']

changed_at_bind = base.copy()


class ReplaceAxesDuringCopy:
    def __deepcopy__(self, memo):
        changed_at_bind.index = pd.MultiIndex.from_tuples([('B', 20), ('B', 30), ('A', 40), ('A', 50)])
        changed_at_bind.attrs = {}
        return self


changed_at_bind.attrs['callback'] = ReplaceAxesDuringCopy()
model = fresh(changed_at_bind)
model.Data.table()
assert not changed_at_bind.attrs
item = model.Calc[1]
item.prime()
with count(changed_at_bind) as hits:
    assert item.anchors() == [(40, 4.0), (50, 8.0)] and hits == ['A']

shadowed = base.copy()
model = fresh(shadowed)
model.Data.table()
item = model.Calc[1]
item.prime()
xs = shadowed.xs
shadowed.xs = lambda key, axis=0: xs(key, axis=axis).iloc[::-1]
with count(shadowed) as hits:
    assert item.anchors() == [(10, 8.0), (0, 4.0)] and hits == ['A']

redirected = base.copy()
redirected.index = base.index.copy(deep=True)
model = fresh(redirected)
model.Data.table()
item = model.Calc[1]
item.prime()
# A standard bound method on a different Index still changes this lookup.
redirected.index.levels[0].get_loc = pd.Index(['B', 'A']).get_loc
with count(redirected) as hits:
    assert item.anchors() == [(0, 1.0), (10, 2.0)] and hits == ['A']


class ReversedFrame(pd.DataFrame):
    @property
    def iloc(self):
        original = super().iloc

        class Reverse:
            def __getitem__(self, key):
                return original[key].iloc[::-1]

        return Reverse()


reclassified = base.copy()
model = fresh(reclassified)
model.Data.table()
item = model.Calc[1]
item.prime()
reclassified.__class__ = ReversedFrame
selected = reclassified.loc['A']
expected = [(int(k), float(v)) for k, v in zip(selected.index, selected['amount'])]
assert expected != [(0, 4.0), (10, 8.0)]
with count(reclassified) as hits:
    assert item.anchors() == expected and hits == ['A']

replaced = base.copy()
model = fresh(replaced)
model.Data.table()
item = model.Calc[1]
item.prime()
replaced.index = pd.MultiIndex.from_tuples([('A', 40), ('A', 50), ('B', 60), ('B', 70)])
with count(replaced) as hits:
    assert item.anchors() == [(40, 1.0), (50, 2.0)] and hits == ['A']

retry = base.rename(columns={'amount': 'previous'})
model = fresh(retry)
model.Data.table()
item = model.Calc[1]
item.prime()
try:
    item.anchors()
except KeyError:
    pass
else:
    raise AssertionError('missing column returned anchors')
retry.columns = pd.Index(['amount'])
with count(retry) as hits:
    assert item.anchors() == [(0, 4.0), (10, 8.0)] and hits == ['A']
''')
    (tmp_path / 'spec.py').write_text("{'strict_lowering': True, 'verify_inputs': True}", encoding='utf-8')
    run('-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py', '--translate-only')
    assert '.zip_prepared_prefix(' not in (compiled / '_mx_classes.py').read_text(encoding='utf-8')
