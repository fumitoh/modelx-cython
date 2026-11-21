"""Numeric cache keys retain the existing compiled argument semantics."""
import subprocess
import sys
import textwrap

import modelx as mx


def test_numeric_keys(tmp_path):
    model = mx.new_model('NumericalKeys')
    try:
        for name in ('Plain', 'Locked'):
            s = model.new_space(name, refs={'seen': []})
            s.new_cells('value', textwrap.dedent('''\
                def value(a, b):
                    seen.append((a, b))
                    if a != a:
                        return float(len(seen))
                    return -0.0 if a == 0.0 else a + b
                '''))
            s.new_cells('integer', 'lambda a: 1152921504606846979')
        model.export(tmp_path/'exported', locked_spaces=['Locked'])
    finally:
        model.close()
    (tmp_path/'sample.py').write_text(textwrap.dedent('''\
        from exported import mx_model as m
        for s in (m.Plain, m.Locked):
            s.value(1.5, 1)
            s.value(0.0, 1)
            s.value(0.0, 2)
            s.integer(1.5)
        '''))
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
                    '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, check=True)
    probe = textwrap.dedent('''\
        import math, sys
        from concurrent.futures import ThreadPoolExecutor
        from exported_cy import mx_model as m
        cache_type = next(module.NumericCache for name, module in tuple(sys.modules.items())
                          if name.startswith('exported_cy.') and hasattr(module, 'NumericCache'))
        try:
            cache_type(1 << 62)
        except (MemoryError, OverflowError):
            pass
        else:
            raise AssertionError('oversized native allocation was not rejected')
        for s in (m.Plain, m.Locked):
            assert math.copysign(1.0, s.value(-0.0, 1)) == -1.0
            assert math.copysign(1.0, s.value(0.0, True)) == -1.0
            assert len(s.seen) == 1
            assert s.value(1.5, 1) == s.value(1.5, True) == 2.5
            assert len(s.seen) == 2
            nan = float('nan')
            assert s.value(nan, 1) != s.value(nan, 1)
            assert s.integer(2.5) == 1152921504606846979
            with ThreadPoolExecutor(4) as pool:
                values = list(pool.map(lambda i: s.value(float(i), 2), range(1, 1001)))
            assert values == [float(i) + 2 for i in range(1, 1001)]
            assert s.value(999.0, 2) == 1001.0
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
