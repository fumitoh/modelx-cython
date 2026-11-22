"""Explicit native reductions preserve the exported helper boundary."""
import json
import subprocess
import sys
import textwrap

import modelx as mx


def test_builtin_minimum_of_typed_range(tmp_path):
    model = mx.new_model('BuiltinMinimum')
    try:
        s = model.new_space('Projection', refs={'values': (1.0, 2.0)})
        s.new_cells('value', 'lambda t, case: values[t]')
        s.new_cells('minimum', 'lambda n, case: min(value(t, case) for t in range(n))')
        s.new_cells('count', 'lambda: 2')
        s.new_cells('positive', 'lambda t: value(t, 0) > 0')
        s.new_cells('all_positive', 'lambda: all(positive(t) for t in range(count()))')
        s.new_cells('first', 'lambda t: t')
        s.new_cells('any_positive', '''def any_positive(t):
    first_index = first(t)
    return any(positive(u) for u in range(max(0, first_index), min(first_index + 2, count())))
''')
        s.new_cells('term', 'lambda t: value(t, 0)')
        s.new_cells('maximum', 'lambda t: max(term(u) for u in range(t, max(-1, t - 12), -3))')
        s.new_cells('total', 'lambda n: sum(value(t, 0) for t in range(n))')
        s.new_cells('float_total', '''def float_total(n):
    total = sum((value(t, 0) for t in range(n)), 0.0)
    return total
''')
        s.new_cells('shadowed', '''def shadowed(n):
    min = max
    return min(value(t, 0) for t in range(n))
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\n'
        'mx_model.Projection.minimum(2, 0)\nmx_model.Projection.shadowed(2)\n'
        'mx_model.Projection.all_positive()\nmx_model.Projection.total(2)\n'
        'mx_model.Projection.any_positive(0)\nmx_model.Projection.any_positive(2)\n'
        'mx_model.Projection.maximum(1)\nmx_model.Projection.minimum(2, 6)\n'
        'mx_model.Projection.float_total(2)\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name, count in (('minimum', 0), ('shadowed', 0), ('all_positive', 1), ('total', 1),
                        ('any_positive', 0), ('maximum', 0), ('float_total', 0)):     # max and a sum start: not lowered (R-6)
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == count
    # min over a typed range and the bounded any are general-lowering sites since R-6 5a (the report counts
    # specialised sites only): their generators are gone from the generated module, with no cold fallback
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert 'min(value(t, case) for t in range(n))' not in source
    assert 'any(positive(u) for u in range(' not in source
    probe = textwrap.dedent('''\
        import math
        import sys
        from importlib import import_module

        model = import_module(sys.argv[1]).mx_model
        s = model.Projection
        assert s.minimum(2, 0) == 1.0 and s.shadowed(2) == 2.0
        assert s.all_positive() is True and s.total(2) == 3.0
        assert s.any_positive(1) is True and s.any_positive(2) is False
        assert s.maximum(0) == 1.0 and s.maximum(1) == 2.0
        assert s.float_total(2) == 3.0 and type(s.float_total(0)) is float
        values_by_case = ((-0., 0.), (0., -0.), (float('nan'), 1.), (1., float('nan')),
                          (float('inf'), float('-inf')))
        for case, values in enumerate(values_by_case, 1):
            s.values = values
            expected = min(values)
            actual = s.minimum(2, case)
            assert math.isnan(actual) if math.isnan(expected) else actual == expected
            if actual == 0.:
                assert math.copysign(1., actual) == math.copysign(1., expected)
        try:
            min(())
        except ValueError as error:
            empty_message = str(error)
        try:
            s.minimum(0, 0)
        except ValueError as error:
            assert str(error) == empty_message
        else:
            raise AssertionError('expected empty minimum to fail')
        error = StopIteration('value')
        class Broken:
            def __getitem__(self, index):
                raise error
        s.values = Broken()
        try:
            s.minimum(2, 6)
        except RuntimeError as actual:
            assert actual.__cause__ is error
        else:
            raise AssertionError('generator StopIteration was not wrapped')
        class Override(type(s)):
            def value(self, t, case):
                return 7.5 - t
        override = Override(model)
        override._mx_copy_refs(s, s)
        assert override.minimum(2, 0) == 6.5
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
