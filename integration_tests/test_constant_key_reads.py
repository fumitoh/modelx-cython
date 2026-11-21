"""D79 (diag-vec): table reads whose key parts are build-time constants.

(1) An IntEnum member admitted at build (enum definition) used as a table key part is its module
constant (``_mx_enum_const_N``), not an attribute read of the enum class per call.
(2) A prepared read holds the codes of constant string labels next to the table handle.
The first read resolves them, a missing key raises KeyError at the read, and collect resets the holds.
Results must not change: compiled at one and four workers equals the exported model.
"""
from http import HTTPStatus
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd


def _model(tmp_path):
    keys = tuple(range(1, 41))
    model = mx.new_model('ConstKeys')
    try:
        portfolio = model.new_space('Portfolio')
        data = portfolio.new_space('Inputs', refs={
            'params': pd.DataFrame({'value': [0.5, 0.25, 0.125]}, index=[200, 201, 202]),
            'loads': pd.DataFrame({'load': [2.0, 3.0, 5.0]}, index=['ok', 'created', 'accepted']),
            'grid': pd.DataFrame({'f': [1.5, 2.5, 3.5, 4.5]}, index=pd.MultiIndex.from_tuples(
                [('ok', 'first'), ('ok', 'second'), ('created', 'first'), ('created', 'second')]))})
        data.new_cells('basis', 'lambda: params')
        data.new_cells('load_basis', 'lambda: loads')
        data.new_cells('cube', 'lambda: grid')
        projection = portfolio.new_space(
            'Projection', formula='lambda key: None', refs={'market': data, 'keys': keys, 'P': HTTPStatus},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('rate', textwrap.dedent('''\
            def rate(t):
                mix = market.basis().at[P.OK, "value"]
                load = market.load_basis().loc["created", "load"]
                g = market.cube().at[("created", "second"), "f"]
                return mix * (t + key) + load * 0.001 * t + g
            '''))
        projection.new_cells('value', textwrap.dedent('''\
            def value():
                total = 0.0
                for t in range(24):
                    total += rate(t)
                return total
            '''))
        projection.new_cells('rare', textwrap.dedent('''\
            def rare(t):
                if t > 20:
                    return market.basis().at[P.NOT_FOUND, "value"]
                return market.basis().at[P.ACCEPTED, "value"] * t
            '''))
        model.export(tmp_path / 'exported', locked_spaces=[data])
    finally:
        model.close()
    spec = {'parallel_workers': 4, 'spaces': {'Portfolio': {'spaces': {'Projection': {
        'serial_collect': 'reuse', 'parallel_collect_reuse': True, 'parallel_collect_order': 'domain'}}}}}
    (tmp_path / 'spec.py').write_text(repr(spec), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Portfolio.Projection\n'
        'p.collect()\n'
        'for key in (1, 2, 3):\n'
        '    [p[key].rate(t) for t in range(24)]; p[key].value(); [p[key].rare(t) for t in range(3)]\n',
        encoding='utf-8')


def test_constant_key_reads_use_enum_constants_and_held_codes(tmp_path):
    _model(tmp_path)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--spec', 'spec.py'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    generated = (tmp_path / 'exported_cy/_m_Portfolio/_mx_classes.py').read_text(encoding='utf-8')
    enum_read = next(line for line in generated.splitlines() if line.lstrip().startswith('mix ='))
    assert '.lcode(' in enum_read and ', 0, _mx_enum_const_' in enum_read
    assert generated.count('.hold_code(') == 3
    assert all(f'root._mx_tk_{slot} = 0' in generated for slot in range(3))
    assert 'root._mx_tp_0 = None' in generated
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        m = import_module(sys.argv[1]).mx_model
        p = m.Portfolio.Projection
        if sys.argv[1] == 'exported_cy':
            serial = type(m)()
            serial.collect_workers = 1
            print(float(serial.Portfolio.Projection.collect().sum.value), float(p.collect().sum.value))
        else:
            print(float(p.collect().sum.value), float(p.collect().sum.value))
        item = p[7]
        for attempt in range(2):
            try:
                item.rare(22)
            except KeyError as error:
                # Prepared reads and pandas both reject the missing label.
                if sys.argv[1] == 'exported':
                    assert int(error.args[0]) == 404
                print('KeyError')
            else:
                raise AssertionError('missing enum label accepted')
        print(repr([float(item.rare(t)) for t in range(5)]))
        values = [item.rate(t) for t in range(24)] + [item.value()]
        assert [item.rate(t) for t in range(24)] + [item.value()] == values
        print(repr([float(value) for value in values]))
        ''')
    out = {}
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        out[package] = run.stdout.splitlines()
    one, four = out['exported_cy'][0].split()
    assert one == out['exported'][0].split()[0], out
    assert float(four) == float(one), out
    assert out['exported_cy'][1:] == out['exported'][1:], out
