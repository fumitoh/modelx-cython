import os
import subprocess
import sys
import textwrap

import modelx as mx
import pytest


@pytest.mark.parametrize("native_cache", [False, True])
def test_numeric_dict_lookup_preserves_misses_errors_and_locked_cells(tmp_path, native_cache):
    model = mx.new_model('DictionaryLookup')
    try:
        for name in ('Plain', 'Locked'):
            space = model.new_space(name, refs={'callback': abs, 'values': {'a': 0.0, (1,): 2.0}})
            space.new_cells('number', 'lambda x=0.5: callback(x)')
            space.new_cells('text', 'lambda key="a": values[key]')
            space.new_cells('nullable', 'lambda key: None')
            space.new_cells('general', 'lambda key: values[key]')
            space.new_cells('internal_name', 'lambda _mx_cache_value: callback(_mx_cache_value)')
            space.new_cells('text_internal', 'lambda _mx_cache_value: values[_mx_cache_value]')
        model.export(tmp_path / 'exported', locked_spaces=['Locked'])
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model as m\n'
        'for s in (m.Plain, m.Locked):\n'
        '    s.number(0.5); s.text(); s.nullable("a"); s.general((1,)); s.internal_name(0.5); s.text_internal("a")\n',
        encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'spaces': {
        name: {'cells': {cell: {'native_numeric_cache': native_cache}
                         for cell in ('number', 'internal_name')}}
        for name in ('Plain', 'Locked')
    }}), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--spec', 'spec.py'],
                   cwd=tmp_path, env=dict(os.environ), check=True)
    probe = textwrap.dedent('''\
        from pathlib import Path
        from exported_cy import mx_model as m
        source = Path('exported_cy/_mx_classes.py').read_text()
        import sys
        assert ('.NumericCache()' in source) == (sys.argv[1] == 'True')
        assert '.get(key)' in source
        assert 'if key in self._v_text:' in source
        assert 'if key in self._v_general:' in source
        assert 'if key in self._v_nullable:' in source
        assert '_mx_cache_value_: object' in source
        for space in (m.Plain, m.Locked):
            calls = []
            failure = LookupError('retry')
            def callback(x):
                calls.append(x)
                if len(calls) == 1:
                    raise failure
                return x
            space.callback = callback
            try:
                space.number(x=-1.5)
            except LookupError as error:
                assert error is failure
            else:
                raise AssertionError('missing error')
            assert space.number(-1.5) == space.number(-1.5) == -1.5
            assert calls == [-1.5, -1.5]
            assert space.number(0.0) == space.number(0.0) == 0.0
            assert calls == [-1.5, -1.5, 0.0]
            assert space.internal_name(_mx_cache_value=3.5) == space.internal_name(3.5) == 3.5
            assert space.text_internal('a') == 0.0
            assert space.general((1,)) == 2.0
            assert space.text() == space.text(key='a') == 0.0
            class Label(str):
                pass
            try:
                space.text(Label('a'))
            except TypeError:
                pass
            else:
                raise AssertionError('changed Cython string boundary')
            space.values['fresh'] = 3.25
            assert space.text('fresh') == space.text('fresh') == 3.25
            assert space.nullable('missing') is None
            assert space.nullable('missing') is None
            space.values = {'a': 9.0}
            assert space.text('a') == 0.0
        ''')
    subprocess.run([sys.executable, '-c', probe, str(native_cache)], cwd=tmp_path,
                   env=dict(os.environ), check=True)
