"""Differential boundaries for composed native boolean reductions."""
import json
import subprocess
import sys
import textwrap

import modelx as mx


def test_nested_filtered_boolean_ranges(tmp_path):
    model = mx.new_model('BooleanRanges')
    try:
        s = model.new_space('Projection', refs={
            'events': [],
            'amounts': (1e16, 1.0, -1e16, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
            'rate': 0.25,
            'throw_horizon': False,
            'throw_at': -1, 'false_at': 2, 'error': StopIteration('sentinel')})
        s.new_cells('bound', '''def bound(n):
    events.append(('bound', n))
    if n == -9:
        raise error
    return n
''')
        s.new_cells('inner', '''def inner(i):
    events.append(('inner', i))
    return 3
''')
        s.new_cells('predicate', '''def predicate(i):
    events.append(('term', i))
    if i == throw_at:
        raise error
    return i != false_at
''')
        s.new_cells('every', 'lambda n: all(predicate(i) for i in range(bound(n)) if i != 1)')
        s.new_cells('some', 'lambda n: any(predicate(j) for i in range(bound(n)) if i != 1 for j in range(inner(i)) if j != 0)')
        s.new_cells('inner_step', '''def inner_step(i):
    events.append(('step', i))
    if throw_horizon:
        raise error
    return 2
''')
        s.new_cells('some_step', 'lambda n: any(predicate(j) for i in range(bound(n)) if i != 1 for j in range(0, inner(i), inner_step(i)) if j != 0)')
        s.new_cells('reverse', 'lambda n: all(predicate(i) for i in range(bound(n), -1, -1) if i != 1)')
        s.new_cells('amount', '''def amount(i):
    events.append(('amount', i))
    if i == throw_at:
        raise error
    return amounts[i]
''')
        s.new_cells('total', 'lambda n: sum((amount(i) for i in range(bound(n)) if i != 1), 0.0)')
        s.new_cells('maximum', 'lambda n: max(amount(i) for i in range(bound(n)) if i != 1)')
        s.new_cells('singleton', 'lambda n: sum((value for value in (amount(n),)), 0.0)')
        s.new_cells('inner_singleton', 'lambda n: sum((value for i in range(bound(n)) for value in (amount(i),)), 0.0)')
        s.new_cells('captured', '''def captured(n):
    factor = rate
    return sum((amount(i) * factor for i in range(bound(n))), 0.0)
''')
        s.new_cells('captured_singleton', '''def captured_singleton(n):
    factor = rate
    return sum((value * factor for value in (amount(n),)), 0.0)
''')
        s.new_cells('descending', '''def descending(n):
    if n == -10:
        return 0.0
    return max(amount(i) for i in range(n, max(-1, n - 12), -3))
''')
        s.new_cells('origin', '''def origin(n):
    events.append(('origin', n))
    return n
''')
        s.new_cells('horizon', '''def horizon():
    events.append(('horizon',))
    if throw_horizon:
        raise error
    return 8
''')
        s.new_cells('legacy_any', '''def legacy_any(n):
    first = origin(n)
    return any(predicate(i) for i in range(max(0, first), min(first + 3, horizon())))
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\np = mx_model.Projection\n'
        'p.every(8); p.some(8); p.some_step(8); p.some_step(-10); p.reverse(8)\n'
        'p.every(-10); p.some(-10); p.reverse(-10)\n'
        'p.total(8); p.total(-10); p.maximum(8)\n'
        'p.singleton(0); p.singleton(8); p.inner_singleton(8); p.inner_singleton(-10)\n'
        'p.captured(8); p.captured(-10); p.captured_singleton(0); p.captured_singleton(8)\n'
        'p.descending(8); p.descending(-10); p.legacy_any(0); p.legacy_any(-10)\n')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    for name in ('every', 'some', 'some_step', 'reverse', 'total', 'maximum', 'singleton',
                 'inner_singleton', 'captured', 'captured_singleton', 'descending', 'legacy_any'):
        assert report[f'exported_cy._mx_classes._c_Projection._f_{name}']['native_reductions'] == 0
    # the bounded any is a general-lowering site since R-6 5a (the report counts specialised sites only):
    # its generator is gone from the generated module, with no cold fallback
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert 'any(predicate(i) for i in range(' not in source
    probe = textwrap.dedent('''\
        import sys
        import pickle
        from importlib import import_module
        model = import_module(sys.argv[1]).mx_model
        def fresh():
            return pickle.loads(pickle.dumps(model)).Projection
        s = fresh()
        s.events.clear()
        assert s.every(4) is False
        assert s.events == [('bound', 4), ('term', 0), ('term', 2)]
        s = fresh()
        assert s.some(4) is True
        assert s.events == [('bound', 4), ('inner', 0), ('term', 1)]
        s = fresh()
        assert s.some_step(4) is False
        assert s.events == [('bound', 4), ('inner', 0), ('step', 0), ('term', 2),
                            ('inner', 2), ('step', 2), ('inner', 3), ('step', 3)]
        s = fresh()
        s.throw_horizon = True
        try: s.some_step(4)
        except RuntimeError as error:
            assert error.__cause__ is s.error
            assert error.__context__ is s.error
        else: raise AssertionError('inner step StopIteration not converted')
        assert s.events == [('bound', 4), ('inner', 0), ('step', 0)]
        s = fresh()
        assert s.every(0) is True
        assert s.events == [('bound', 0)]
        s = fresh()
        assert s.some(0) is False
        assert s.events == [('bound', 0)]
        s = fresh()
        assert s.reverse(4) is False
        assert s.events == [('bound', 4), ('term', 4), ('term', 3), ('term', 2)]
        s = fresh()
        s.throw_at = 0
        try:
            s.every(5)
        except RuntimeError as error:
            assert str(error) == 'generator raised StopIteration'
            assert error.__cause__ is s.error
            assert error.__context__ is s.error
        else:
            raise AssertionError('body StopIteration not converted')
        try:
            s.every(-9)
        except StopIteration as error:
            assert error is s.error
        else:
            raise AssertionError('outer bound StopIteration changed')
        s = fresh()
        s = fresh()
        assert s.total(4) == 4.0
        assert s.events == [('bound', 4), ('amount', 0), ('amount', 2), ('amount', 3)]
        s = fresh()
        assert s.total(0) == 0.0 and type(s.total(0)) is float
        assert s.events == [('bound', 0)]
        s = fresh()
        assert s.maximum(4) == 1e16
        assert s.events == [('bound', 4), ('amount', 0), ('amount', 2), ('amount', 3)]
        s = fresh()
        try:
            max(())
        except ValueError as error:
            empty_message = str(error)
        try:
            s.maximum(0)
        except ValueError as error:
            assert str(error) == empty_message
        else:
            raise AssertionError('empty maximum did not raise')
        for name in ('total', 'maximum'):
            s = fresh()
            s.throw_at = 0
            try:
                getattr(s, name)(5)
            except RuntimeError as error:
                assert error.__cause__ is s.error and error.__context__ is s.error
            else:
                raise AssertionError('numeric body StopIteration not converted')
        s = fresh()
        assert s.singleton(3) == 4.0
        assert s.events == [('amount', 3)]
        s = fresh()
        s.throw_at = 3
        try:
            s.singleton(3)
        except StopIteration as error:
            assert error is s.error
        else:
            raise AssertionError('outer singleton StopIteration changed')
        s = fresh()
        assert s.inner_singleton(3) == sum((1e16, 1.0, -1e16), 0.0)   # compensated from Python 3.12: 1.0, before: 0.0
        assert s.events == [('bound', 3), ('amount', 0), ('amount', 1), ('amount', 2)]
        s = fresh()
        s.throw_at = 0
        try:
            s.inner_singleton(3)
        except RuntimeError as error:
            assert error.__cause__ is s.error and error.__context__ is s.error
        else:
            raise AssertionError('inner singleton StopIteration not converted')
        s = fresh()
        assert s.captured(3) == sum((v * 0.25 for v in (1e16, 1.0, -1e16)), 0.0)   # 0.25 from 3.12, 0.0 before
        s = fresh()
        s.throw_at = 0
        try:
            s.captured(3)
        except RuntimeError as error:
            assert error.__cause__ is s.error and error.__context__ is s.error
        else:
            raise AssertionError('captured sum StopIteration not converted')
        s = fresh()
        s.throw_at = 3
        try:
            s.captured_singleton(3)
        except StopIteration as error:
            assert error is s.error
        else:
            raise AssertionError('captured outer singleton StopIteration changed')
        s = fresh()
        assert s.descending(6) == 1e16
        s = fresh()
        s.throw_at = 6
        try:
            s.descending(6)
        except RuntimeError as error:
            assert error.__cause__ is s.error and error.__context__ is s.error
        else:
            raise AssertionError('descending maximum StopIteration not converted')
        s = fresh()
        s.throw_at = 0
        try:
            s.legacy_any(0)
        except RuntimeError as error:
            assert error.__cause__ is s.error and error.__context__ is s.error
        else:
            raise AssertionError('bounded any StopIteration not converted')
        s = fresh()
        s.throw_horizon = True
        try:
            s.legacy_any(0)
        except StopIteration as error:
            assert error is s.error
        else:
            raise AssertionError('bounded any outer bound StopIteration changed')
        ''')
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, check=True)
