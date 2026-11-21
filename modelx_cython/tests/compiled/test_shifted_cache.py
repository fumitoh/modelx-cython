"""Sampled negative indices stay valid in both full and compact caches."""
import json
import subprocess
import sys
import textwrap

import modelx as mx
import pytest


@pytest.mark.parametrize('window', [None, (-3, 6), (0, 2)],
                         ids=['sampled', 'explicit-full', 'explicit-compact'])
def test_shifted_dense_bounds(tmp_path, window):
    model = mx.new_model('ShiftedCaches')
    try:
        for name in ('Plain', 'Locked'):
            s = model.new_space(name, refs={'seen': []})
            s.new_cells('value', textwrap.dedent('''\
                def value(t):
                    seen.append(t)
                    return t * 1.0
                '''))
            s.new_cells('previous', 'lambda t: t - 1')
            s.new_cells('legacy', 'lambda t: t * 1.0')
        model.export(tmp_path/'exported', locked_spaces=['Locked'])
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('from exported import mx_model as m\n'
        'from numpy import int64\n'
        'for s in (m.Plain, m.Locked):\n'
        '    for t in (-3, 0, 3):\n'
        '        s.value(int64(t)); s.previous(int64(t))\n'
        '    s.legacy(0); s.legacy(3)\n')
    spec = ['--no-spec']
    if window is not None:
        (tmp_path/'spec.py').write_text(repr({'spaces': {
            name: {'cells_param_size': {'t': 6},
                   'cells': {cell: {'dense_window': [window]} for cell in ('value', 'previous')}}
            for name in ('Plain', 'Locked')
        }}), encoding='utf-8')
        spec = ['--spec', 'spec.py']
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', *spec], cwd=tmp_path, check=True)
    stop = 4 if window is None else 6
    logical = (-3, stop)
    physical = window or logical
    compact = physical != logical
    report = json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    shifted = report['exported_cy._mx_classes._c_Plain._f_value']
    assert shifted['dense_offsets'] == [physical[0]]
    assert shifted['exact_spill'] is compact
    assert report['exported_cy._mx_classes._c_Plain._f_previous']['exact_spill'] is compact
    assert shifted['dense_logical_domain'] == [list(logical)]
    probe = textwrap.dedent('''\
        import sys
        from exported_cy import mx_model as m

        stop = int(sys.argv[1])
        for name in ('Plain', 'Locked'):
            s = getattr(m, name)
            indices = [-3, -1, 0, 3] + ([5] if stop == 6 else [])
            for t in indices:
                assert s.value(t) == s.value(t) == float(t)
                assert s.previous(t) == t - 1
            for cell in ('value', 'previous'):
                for t in (-4, stop, -(1 << 63)):
                    try:
                        getattr(s, cell)(t)
                    except IndexError as error:
                        assert f'{name}.{cell}' in str(error) and f't in [-3, {stop})' in str(error)
                        assert 'cells_param_size' in str(error)
                    else:
                        raise AssertionError('invalid signed index reached the formula')
            assert s.seen == indices
            try:
                s.legacy(-1)
            except IndexError as error:
                assert f'{name}.legacy' in str(error)
            else:
                raise AssertionError('unsigned sample unexpectedly admitted a negative index')
        ''')
    subprocess.run([sys.executable, '-c', probe, str(stop)],
                   cwd=tmp_path, check=True)
