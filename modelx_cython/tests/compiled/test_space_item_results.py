"""A held ItemSpace fast path must preserve cold and unexpected results."""

import json
import re
import subprocess
import sys
import textwrap

import modelx as mx


def test_native_held_item_uses_exact_type_and_preserves_fallback(tmp_path, monkeypatch):
    (tmp_path / 'held_item_test_hook.py').write_text('callback = None\noverride = None\n', encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    import held_item_test_hook as hook

    model = mx.new_model('HeldSpaceResults')
    try:
        market = model.new_space('Market', formula='lambda shock=0: None')
        market.new_cells('rate', 'lambda t, tenor: 0.01 + 0.001 * shock + 0.0001 * t + 0.002 * tenor')
        other = model.new_space('Other', formula='lambda shock=0: None')
        other.new_cells('rate', 'lambda t, tenor: 1.01 + 0.001 * shock + 0.0001 * t + 0.002 * tenor')
        policy = model.new_space('Policy', formula='lambda key: None',
                                 refs={'Market': market, 'Other': other, 'Hook': hook})
        policy.new_cells('market', 'lambda: None if key == 2 else Other(1) if key == 3 else Market(1)')
        policy.new_cells('project', 'lambda t: market().rate(t, 1.5)')
        policy.new_cells('mixed', 'lambda: None if key == 2 else Market(1)')
        policy.new_cells('mixed_rate', 'lambda t: 0.0 if mixed() is None else mixed().rate(t, 1.5)')
        policy.new_cells('explicit', 'lambda: Market(1)')
        policy.new_cells('explicit_rate', 'lambda t: explicit().rate(t, 1.5)')
        policy.new_cells('shadowed_market', 'lambda: Market(1)')
        policy.new_cells('shadowed', '''def shadowed(t):
    _mx_cy = 1.5
    return shadowed_market().rate(t, _mx_cy)
''')
        policy.new_cells('overridable', 'lambda: Market(1) if Hook.override is None else Hook.override')
        policy.new_cells('overridden', 'lambda t: overridable().rate(t, 1.5)')
        policy.new_cells('cold_market', '''def cold_market():
    if key == 4:
        raise ValueError('getter')
    return Market(1)
''')
        policy.new_cells('tenor', '''def tenor(t):
    if key == 4:
        raise RuntimeError('argument')
    if Hook.callback is not None:
        Hook.callback()
    return 1.5
''')
        policy.new_cells('ordered', 'lambda t: cold_market().rate(t, tenor(t))')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(textwrap.dedent('''\
        from exported import mx_model as m
        for t in (0, 1, 2):
            m.Policy[1].project(t)
            m.Policy[1].mixed_rate(t)
            m.Policy[1].explicit_rate(t)
            m.Policy[1].shadowed(t)
            m.Policy[1].overridden(t)
            m.Policy[1].ordered(t)
            m.Other(1).rate(t, 1.5)
        m.Policy[2].mixed_rate(0)
        '''), encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'spaces': {'Policy': {'cells': {'explicit': {'return_type': 'object'}}}}}),
                                      encoding='utf-8')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py'],
                           cwd=tmp_path, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    source = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert 'type(self._v_market) is _c_Market' in source
    assert '_mx_cy.cast(_c_Market, _mx_pin_held_item(self._v_market)).rate(' in source
    assert 'type(self._v_mixed) is _c_Market' not in source
    assert 'type(self._v_explicit) is _c_Market' not in source
    assert 'type(self._v_shadowed_market) is _c_Market' not in source
    assert 'type(self._v_overridable) is _c_Market' in source
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    assert report['exported_cy._mx_classes._c_Policy._f_market']['return_type'] == 'object'
    c_source = (tmp_path / 'exported_cy/_mx_classes.c').read_text(encoding='utf-8')
    # The evaluated receiver is an owned temporary while the tenor Cell runs.
    pin = re.search(r'(?P<field>__pyx_t_\d+) = \(?__pyx_v_self->_v_cold_market\)?;\s*'
                    r'__Pyx_INCREF\((?P=field)\);\s*'
                    r'(?P<held>__pyx_t_\d+) = \w+__mx_pin_held_item\((?P=field)\);', c_source)
    assert pin is not None
    argument_start = c_source.index('_has_tenor[', pin.end())
    native_call = c_source.index('->__pyx_vtab)->rate(', argument_start)
    assert c_source.index(f'__Pyx_GOTREF({pin["held"]})', pin.end()) < argument_start
    assert pin['held'] in c_source[native_call:c_source.index(';', native_call)]

    probe = textwrap.dedent('''\
        import json
        import sys
        import held_item_test_hook as hook
        m = __import__(sys.argv[1]).mx_model
        values = [m.Policy[key].project(t) for key in (1, 3) for t in (0, 1, 2)]
        values += [m.Policy[key].mixed_rate(t) for key in (1, 2) for t in (0, 1, 2)]
        values += [m.Policy[1].explicit_rate(t) for t in (0, 1, 2)]
        values += [m.Policy[1].shadowed(t) for t in (0, 1, 2)]
        class Override(type(m.Market(1))):
            def rate(self, t, tenor):
                return 42.0 + t + tenor
        hook.override = Override(m.Market)
        values += [m.Policy[6].overridden(t) for t in (0, 1, 2)]
        assert values[-3:] == [43.5, 44.5, 45.5]
        policy = m.Policy[7]
        values.append(policy.ordered(0))  # fills the held target
        def drop_policy():
            del m.Policy[7]  # release/reset the receiver during the argument Cell
        hook.callback = drop_policy
        values.append(policy.ordered(1))
        hook.callback = None
        assert policy is not m.Policy[7]
        for cell, key in [('project', 2), ('ordered', 4)]:
            try:
                getattr(m.Policy[key], cell)(0)
            except (AttributeError, ValueError) as exc:
                values.append([type(exc).__name__, str(exc)])
        print(json.dumps(values))
        ''')
    results = []
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        results.append(json.loads(run.stdout))
    assert results[1] == results[0]
    assert results[0][-2:] == [['AttributeError', "'NoneType' object has no attribute 'rate'"],
                               ['ValueError', 'getter']]
