"""One Index vocabulary across live, exported and compiled execution."""
import os
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd

from modelx_helpers import pandas_inputs
from modelx_helpers.table_keys import keys


def test_owned_index_order_duplicates_empty_replacement_and_pickle(tmp_path):
    domains = {'Integer': [4, 1, 4], 'Floating': [4.5, 1.5, 4.5],
               'Text': ['equity', 'bond', 'equity']}
    model = mx.new_model('OrderedDomains')
    try:
        for name, labels in domains.items():
            public = pd.Index(labels, name='label')
            space = model.new_space(name, refs={'source_labels': public,
                        'index_ctor': pandas_inputs.index, 'keys': keys})
            space.new_cells('public_index', 'lambda: source_labels')
            space.new_cells('prepared_labels', 'lambda: index_ctor(public_index())')
            space.new_cells('ordered', 'lambda: keys(prepared_labels())')
            space.new_cells('keyword_ordered', 'lambda: keys(table=prepared_labels())')
            space.new_cells('count', 'lambda: len(keys(prepared_labels()))')
            if name != 'Text':
                space.new_cells('total', 'def total() -> object:\n    return sum(t for t in keys(prepared_labels()))\n')
            assert space.ordered() == tuple(labels)
            assert space.keyword_ordered() == tuple(labels)
            assert space.count() == 3
            space.source_labels = public[:0]
            assert space.ordered() == () and space.count() == 0
            if name != 'Text':
                assert type(space.total()) is int and space.total() == 0
            space.source_labels = public
            assert space.ordered() == tuple(labels)
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('''from exported import mx_model as m
for name in ['Integer','Floating','Text']:
    p = getattr(m,name)
    p.ordered(); p.keyword_ordered(); p.count()
    if name != 'Text': p.total()
    fresh = getattr(type(m)(),name)
    fresh.source_labels = fresh.source_labels[:0]
    fresh.ordered(); fresh.keyword_ordered(); fresh.count()
    if name != 'Text': fresh.total()
''')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                            '--sample', 'sample.py', '--no-spec'], cwd=tmp_path,
                           capture_output=True, text=True, env=dict(os.environ))
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text()
    probe = textwrap.dedent('''
        import importlib,pickle,sys
        import pandas as pd
        from modelx_helpers.table_keys import keys
        from modelx_helpers.immutable_inputs import Table,Record
        m = importlib.import_module(sys.argv[1]).mx_model
        labels = {'Integer':(4,1,4),'Floating':(4.5,1.5,4.5),
                  'Text':('equity','bond','equity')}
        forbidden = {keys.__code__,Table.__getitem__.__code__,Record.__getitem__.__code__}
        calls = []
        def profile(frame,event,arg):
            if event == 'call' and frame.f_code in forbidden:
                calls.append(frame.f_code.co_name)
        for name,expected in labels.items():
            p = getattr(type(m)(),name)
            p.prepared_labels()  # Explicit Python snapshot boundary.
            pd.testing.assert_index_equal(p.public_index(),pd.Index(expected,name='label'),exact=True)
            if sys.argv[1].endswith('_cy'): sys.setprofile(profile)
            try:
                assert p.ordered() == expected
                assert p.keyword_ordered() == expected
                assert p.count() == 3
                if name != 'Text': assert p.total() == sum(expected)
            finally: sys.setprofile(None)
            p.source_labels = p.source_labels[:0]
            assert p.ordered() == expected  # Explicit cleanup is required.
            empty = getattr(type(m)(),name)
            empty.source_labels = empty.source_labels[:0]
            assert empty.ordered() == () and empty.keyword_ordered() == ()
            assert empty.count() == 0
            if name != 'Text': assert type(empty.total()) is int and empty.total() == 0
        warmed = type(m)()
        for name in labels: getattr(warmed,name).ordered()
        restored = pickle.loads(pickle.dumps(warmed))
        for name,expected in labels.items(): assert getattr(restored,name).ordered() == expected
        observed = []
        def replacement(*,table):
            observed.append(table)
            return (7,7)
        fresh = type(m)().Integer
        fresh.keys = replacement
        assert fresh.keyword_ordered() == (7,7)
        assert len(observed) == 1
    ''')
    for package in ['exported', 'exported_cy']:
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             capture_output=True, text=True, env=dict(os.environ))
        (tmp_path / (package + '-validation.log')).write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
