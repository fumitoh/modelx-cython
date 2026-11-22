"""diag-us: an IntEnum member held in a Reference, compared with a member, is a C comparison.

SPIA_US_S `commuted_value` reads `conv = commute_disc_convention` (a Reference holding a
`DiscConvention` member, D31) and tests `conv == DiscConvention.compound` in its loop; the
comparison stayed a Python one (two attribute reads and a boxed integer per iteration).
"""
from http import HTTPStatus
import json
import subprocess
import sys
import textwrap

import modelx as mx


PREFIX = 'exported_cy._mx_classes._c_Projection._f_'


def _model(tmp_path):
    model = mx.new_model('EnumRef')
    try:
        s = model.new_space('Projection', refs={'Conv': HTTPStatus, 'conv_ref': HTTPStatus.CREATED})
        s.new_cells('factor', textwrap.dedent('''\
            def factor(t):
                conv = conv_ref
                total = 0.0
                for s in range(t + 1, 6):
                    if conv == Conv.OK:
                        total += 1.0 / (1.0 + s)
                    elif conv == Conv.CREATED:
                        total += 2.0
                    else:
                        raise ValueError("invalid")
                return total
            '''))
        s.new_cells('direct', 'lambda t: 1.0 if conv_ref == Conv.CREATED else 0.5')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model as m\n'
                                        'for t in range(5): m.Projection.factor(t); m.Projection.direct(t)\n')


def test_member_reference_compares_as_c_integer(tmp_path):
    _model(tmp_path)
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                          cwd=tmp_path, check=False, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    assert report[PREFIX + 'factor']['native_enum_comparisons'] == 2
    assert report[PREFIX + 'direct']['native_enum_comparisons'] == 1
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        p = import_module(sys.argv[1]).mx_model.Projection
        if sys.argv[2] == 'changed':
            p.conv_ref = p.Conv.OK      # a new value of the Reference: the C field is read
            assert p.factor(3) == 1.0 / 5.0 + 1.0 / 6.0 and p.direct(0) == 0.5
        else:
            assert [p.factor(t) for t in range(5)] == [2.0 * (5 - t) for t in range(5)]
            assert p.direct(0) == 1.0
        ''')
    for package in ('exported', 'exported_cy'):
        for mode in ('shipped', 'changed'):
            done = subprocess.run([sys.executable, '-c', probe, package, mode], cwd=tmp_path,
                                  capture_output=True, text=True)
            assert done.returncode == 0, package + mode + done.stdout + done.stderr
