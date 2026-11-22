"""W1.4: ``x in (E.A, E.B)`` / ``not in`` over declared enum members as C comparisons."""
from http import HTTPStatus
import json
import subprocess
import sys
import textwrap

import modelx as mx


def test_member_tuple_membership(tmp_path):
    model = mx.new_model('EnumMembership')
    try:
        s = model.new_space('Projection', refs={'Status': HTTPStatus})
        s.new_cells('kind', 'lambda t: 200 + t')
        s.new_cells('hit', 'lambda t: kind(t) in (Status.OK, Status.CREATED)')
        s.new_cells('miss', 'lambda t: kind(t) not in (Status.OK, Status.ACCEPTED, Status.CREATED)')
        s.new_cells('single', 'lambda t: kind(t) in (Status.OK,)')
        s.new_cells('single_not', 'lambda t: kind(t) not in (Status.OK,)')
        s.new_cells('local', textwrap.dedent('''\
            def local(t):
                value = kind(t)
                if value in (Status.ACCEPTED, Status.OK):
                    return 1.5
                return 0.5
            '''))
        s.new_cells('reversed_eq', 'lambda t: Status.CREATED == kind(t)')
        s.new_cells('argument', 'lambda k: k in (Status.OK, Status.NO_CONTENT)')
        s.new_cells('loose', 'lambda x: x in (Status.OK, Status.CREATED)')
        s.new_cells('literal_mix', 'lambda t: kind(t) in (Status.OK, 201)')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'p = m.Projection\n'
        'for t in range(-3, 9):\n'
        '    p.hit(t); p.miss(t); p.single(t); p.single_not(t); p.local(t); p.reversed_eq(t)\n'
        '    p.literal_mix(t)\n'
        'for k in (200, 204, 205, -1): p.argument(k)\n'
        'for x in (200, "a", 201.0, 202, None): p.loose(x)\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())

    def entry(name):
        return report[f'exported_cy._mx_classes._c_Projection._f_{name}']

    for name, count in (('hit', 2), ('miss', 3), ('single', 1), ('single_not', 1), ('local', 2),
                        ('argument', 2)):
        assert entry(name)['native_expressions'] == 1, name
        assert entry(name)['native_enum_comparisons'] == count, name
        assert entry(name)['expressions_untyped'] == [], name
    assert entry('reversed_eq')['native_enum_comparisons'] == 1
    # W1.8: an object name on the left: identity and == tests in tuple order
    assert entry('loose')['expressions_untyped'] == []
    loose = entry('loose')['expressions_object']
    assert [(e['form'], e['line']) for e in loose] == [('in', 2)], loose   # a lambda's body is line 2 of its def
    assert 'object' in loose[0]['reason']
    mixed = entry('literal_mix')['expressions_untyped']
    assert [e['form'] for e in mixed] == ['in'] and 'not a member' in mixed[0]['reason']
    probe = textwrap.dedent('''\
        import sys
        from importlib import import_module
        p = import_module(sys.argv[1]).mx_model.Projection
        for t in range(-3, 9):
            k = 200 + t
            assert p.hit(t) is (k in (200, 201)), t
            assert p.miss(t) is (k not in (200, 202, 201)), t
            assert p.single(t) is (k == 200) and p.single_not(t) is (k != 200)
            assert p.local(t) == (1.5 if k in (202, 200) else 0.5)
            assert p.reversed_eq(t) is (k == 201)
            assert p.literal_mix(t) is (k in (200, 201))
        for k in (200, 204, 205, -1):
            assert p.argument(k) is (k in (200, 204))
        for x in (200, 'a', 201.0, 202, None):
            assert p.loose(x) is (x in (200, 201))
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
