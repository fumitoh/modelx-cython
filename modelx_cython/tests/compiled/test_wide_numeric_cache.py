"""Exercise ordinary native Cell caching with more than three numeric keys."""

import json
import subprocess
import sys
import textwrap

import modelx as mx


def test_wide_numeric_cache(tmp_path):
    model = mx.new_model('WideCache')
    try:
        space = model.new_space('Policy', formula='lambda policy: None')
        space.calls = []
        space.should_fail = False
        space.new_cells('value', textwrap.dedent('''\
            def value(a, b, c, d, e, f):
                calls.append(1)
                return a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f
            '''))
        space.new_cells('recursive', textwrap.dedent('''\
            def recursive(a, b, c, d, e, f):
                if a == 0:
                    if should_fail:
                        raise ValueError('retry')
                    return b
                return recursive(a - 1, b, c, d, e, f) + 1.0
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'mx_model.Policy[0].value(1, 2.0, 3, 4.0, 5, 6.0)\n'
        'mx_model.Policy[0].recursive(1, 2.0, 3, 4.0, 5, 6.0)\n', encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'Policy': {'cells_param_size': {
        ('a', 'b', 'c', 'd', 'e', 'f'): (257, 0, 4, 0, 6, 0),
    }}}}), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py'],
                   cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    assert report['exported_cy._mx_classes._c_Policy._f_value']['cache'] == 'native-numeric-keys'
    probe = textwrap.dedent('''\
        import math
        import pickle
        from exported_cy import mx_model
        s = mx_model.Policy[0]
        before = len(s.calls)
        for k in range(257):
            assert s.value(k, 2., 3, 4., 5, 6.) == k + 90.
        for k in range(257):
            assert s.value(k, 2., 3, 4., 5, 6.) == k + 90.
        assert len(s.calls) == before + 257
        assert s.value(1, 0., 3, 4., 5, 6.) == s.value(1, -0., 3, 4., 5, 6.)
        for _ in range(2):
            assert math.isnan(s.value(1, float('nan'), 3, 4., 5, 6.))
        assert len(s.calls) == before + 260
        restored = pickle.loads(pickle.dumps(s))
        before = len(restored.calls)
        assert restored.value(250, 2., 3, 4., 5, 6.) == 340.
        assert len(restored.calls) == before
        fresh = type(mx_model)().Policy[0]
        fresh.should_fail = True
        try:
            fresh.recursive(100, 2., 3, 4., 5, 6.)
        except ValueError as error:
            assert str(error) == 'retry'
        else:
            raise AssertionError('The recursive calculation should fail before publishing')
        fresh.should_fail = False
        assert fresh.recursive(100, 2., 3, 4., 5, 6.) == 102.
        assert fresh.recursive(50, 2., 3, 4., 5, 6.) == 52.
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
