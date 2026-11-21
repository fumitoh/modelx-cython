"""The prepared table's integer lookup preserves Python key conversion."""

import math
import re
import subprocess
import sys

import modelx as mx
import pandas as pd


def test_builtin_int_key_uses_native_lookup_only_for_proven_int64(tmp_path):
    model = mx.new_model('IntegerKeys')
    try:
        data = model.new_space('Data')
        data.rates = pd.DataFrame({'rate': [3.0, 4.0, 1.0, 2.0]},
                                  index=pd.Index([-(2**63), 0, 20, 21], dtype='int64'))
        data.new_cells('table', 'lambda: rates')
        calc = model.new_space('Calc', refs={'data': data, 'convert': math.sin})
        calc.new_cells('offset', 'lambda t: 20 + t')
        calc.new_cells('native_key', '''def native_key(t):
    a, b = offset(t), offset(t)
    table = data.table()
    return table.loc[int(a), 'rate'] if int(b) in table.index else 0.0
''')
        calc.new_cells('float_key', 'lambda x: data.table().loc[int(x), "rate"]')
        calc.new_cells('string_key', 'lambda x: data.table().loc[int(x), "rate"]')
        calc.new_cells('shadowed', '''def shadowed(x):
    int = convert
    return int(x) in data.table().index
''')
        calc.new_cells('shadowed_asof', '''def shadowed_asof(x):
    int = convert
    return int(data.table().index.asof(x))
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text('''from exported import mx_model as model
calc = model.Calc
for t in (0, 1):
    calc.native_key(t)
    calc.float_key(20.5 + t)
    calc.string_key(str(20 + t))
    calc.shadowed(t)
    calc.shadowed_asof(20 + t)
calc.native_key(2)
''', encoding='utf-8')
    built = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                            '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr

    source = (tmp_path / 'exported_cy' / '_mx_classes.py').read_text(encoding='utf-8')
    for key in ('a', 'b'):
        assert re.search(rf'\.kcode\([^\n]+_mx_cy\.cast\(_mx_cy\.longlong, int\({key}\)\)', source)
    assert '_mx_cy.cast(_mx_cy.longlong, int(x))' not in source

    checked = subprocess.run([sys.executable, '-c', '''
import math
from exported import mx_model as live
from exported_cy import mx_model as compiled

for model in (live, compiled):
    calc = model.Calc
    assert [calc.native_key(t) for t in (0, 1, 2)] == [1.0, 2.0, 0.0]
    assert calc.float_key(20.5) == calc.string_key('20') == 1.0
    assert [calc.shadowed(x) for x in (0, 1)] == [True, False]
    assert calc.shadowed_asof(20) == math.sin(20)
    for cell, value in ((calc.float_key, float('nan')), (calc.float_key, float(2**70)),
                        (calc.string_key, str(2**70))):
        try:
            cell(value)
        except (ValueError, OverflowError, KeyError):
            pass
        else:
            raise AssertionError(f'{cell} accepted {value!r} as a key')
'''], cwd=tmp_path, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
