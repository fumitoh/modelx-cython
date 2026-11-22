"""audit-routes P6 (D126): every compiled root takes its lookup contexts over privately.

Items of an exported ItemSpace class are partitioned by the caller (the thread rule that
``numeric_cache.exclusively_owned`` already relies on), so a direct item keeps the round memo as
collect's reused root does. A Cython probe reads the context of the compiled class: after a
``round(x, n)`` the memo is ready on a direct item (before P6 it never was). Values equal live on
both routes.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import modelx as mx

from modelx_cython import runtime


def _build(tmp_path):
    model = mx.new_model('PrivateLookups')
    try:
        keys = (1, 2, 3)
        proj = model.new_space(
            'Projection', formula='lambda key: None', refs={'keys': keys},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        proj.new_cells('rounded', '''\
def rounded(t):
    amount = key * 0.37 + t * 0.011
    adjustment = 0.125
    total = amount + adjustment
    return round(total, min(t, 22))
''')
        proj.new_cells('value', 'lambda: sum(rounded(t) for t in range(4))')
        port = model.new_space('Portfolio', refs={'proj': proj})
        port.new_cells('direct', '''\
def direct():
    total = 0.0
    for k in proj.keys:
        item = proj[k]
        total += item.rounded(2) + item.rounded(3)
    return total
''')
        live = {'collect': proj.collect().sum.value, 'direct': port.direct(),
                'items': [proj[k].rounded(t) for k in keys for t in range(4)]}
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'Projection': {'serial_collect': 'reuse'}}}),
                                      encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'm.Projection.collect(); m.Portfolio.direct()\n'
        'for k in (1, 2, 3):\n    [m.Projection[k].rounded(t) for t in range(4)]\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                    '--spec', 'spec.py'], cwd=tmp_path, check=True)
    return live


def test_direct_items_and_collect_roots_own_their_lookup_contexts(tmp_path):
    live = _build(tmp_path)
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    contexts = sorted({line.split(':')[0].strip() for line in source.splitlines()
                       if line.strip().startswith('_v_mx_context_') and 'LookupContext' in line})
    assert contexts, 'the model has no rounding context: the test shape no longer applies'
    field = contexts[0]
    (tmp_path / 'context_probe.pyx').write_text(textwrap.dedent(f'''\
        from exported_cy._mx_classes cimport _c_Projection

        def memo_ready(_c_Projection space):
            return bool(space.{field}.round_ready)
        '''), encoding='utf-8')
    include = Path(runtime.__file__).resolve().parents[2]
    (tmp_path / 'probe_setup.py').write_text(textwrap.dedent(f'''\
        from setuptools import setup
        from Cython.Build import cythonize
        setup(ext_modules=cythonize(['context_probe.pyx'], include_path=['.', {str(include)!r}],
              compiler_directives={{'language_level': 3, 'freethreading_compatible': True}}))
        '''), encoding='utf-8')
    subprocess.run([sys.executable, 'probe_setup.py', 'build_ext', '--inplace'], cwd=tmp_path, check=True)
    probe = textwrap.dedent(f'''\
        from context_probe import memo_ready
        from exported_cy import mx_model as m
        expected = {live!r}
        p = m.Projection
        item = p[2]
        assert [p[k].rounded(t) for k in (1, 2, 3) for t in range(4)] == expected['items']
        assert memo_ready(item), 'a direct item did not keep its round memo'
        assert p.collect().sum.value == expected['collect']
        assert m.Portfolio.direct() == expected['direct']
        print('ok')
        ''')
    run = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == 'ok'
