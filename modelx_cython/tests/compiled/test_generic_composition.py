"""Compiler features compose without narrowing legal Python values/providers."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from http import HTTPStatus

import modelx as mx


def compile_model(path, sample):
    (path/'sample.py').write_text(sample, encoding='utf-8')
    subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported', '--no-spec'],
                   cwd=path, check=True)


def test_generator_literals_preserve_integer_width_and_sign(tmp_path):
    values = {'minimum': -(1 << 63), 'maximum': (1 << 63)-1,
              'above': 1 << 63, 'below': -(1 << 63)-1, 'large': 1 << 80}
    low, high = -(1 << 63), (1 << 63)-1
    ranges = {'positive_overshoot': (high-1, high, 2),
              'negative_overshoot': (low+1, low, -2),
              'positive_safe': (high-2, high, 2), 'negative_safe': (low+2, low, -2),
              'empty': (high-1, high-2, 2)}
    model = mx.new_model('LiteralDomains')
    try:
        space = model.new_space('Values', refs={'Status':HTTPStatus, 'stride': 2})
        for name, value in values.items():
            space.new_cells(name, f'lambda: sum((k * 1.0 for k in ({value},)), 0.0)')
        space.new_cells('step', f'lambda: sum((k * 1.0 for k in range(0, 2, {1 << 80})), 0.0)')
        space.new_cells('filtered', f'lambda: sum((k * 1.0 for k in ({1 << 63},) if k > 0), 0.0)')
        for name, args in ranges.items():
            space.new_cells(name, f'lambda: sum((k * 1.0 for k in range{args!r}), 0.0)')
        space.new_cells('dynamic_stride', f'lambda epoch=0: sum((k * 1.0 for k in range({high-1}, {high}, stride)), 0.0)')
        space.new_cells('authored_enum_prefix', '''def authored_enum_prefix():
    _mx_enum_const_0 = -123
    if _mx_enum_const_0 < 0:
        return Status.CONTINUE
    return 0
''')
        space.new_cells('assigned', '''def assigned(epoch=0):
    total = sum((k * 1.0 for k in range(3)), 0.0)
    return str(total)
''')
        space.new_cells('weighted', '''def weighted(epoch=0):
    factor = 1.5
    total = sum((k * factor for k in range(3)), 0.0)
    return total
''')
        model.export(tmp_path/'exported')
    finally:
        model.close()
    names = (*values, *ranges, 'dynamic_stride', 'step', 'filtered', 'authored_enum_prefix', 'assigned', 'weighted')
    compile_model(tmp_path, 'from exported import mx_model\n'
                  + '\n'.join(f'mx_model.Values.{name}()' for name in names)
                  + '\nfor epoch in (1,2): mx_model.Values.assigned(epoch); mx_model.Values.weighted(epoch)\n')
    expected = {name: float(value) for name, value in values.items()}
    expected.update({name: sum(k * 1.0 for k in range(*args)) for name, args in ranges.items()})
    expected['dynamic_stride'] = float(high-1)
    expected.update(step=0.0, filtered=float(1 << 63), authored_enum_prefix=100,
                    assigned='3.0', weighted=4.5)
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-B', '-c',
            f'from {package} import mx_model\n'
            f'expected={expected!r}\n'
            'for name,value in expected.items():\n'
            '    actual=getattr(mx_model.Values,name)()\n'
            '    assert actual==value,(name,actual,value)\n'], cwd=tmp_path, check=True, timeout=30)
    admission = json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    # 'assigned' .. 'empty' are sums with a start argument, not lowered since R-6; the value oracles stay.
    for name in ('above', 'below', 'large', 'step', 'filtered',
                 'positive_overshoot', 'negative_overshoot', 'dynamic_stride',
                 'assigned', 'weighted', 'positive_safe', 'negative_safe', 'empty'):
        assert admission[f'exported_cy._mx_classes._c_Values._f_{name}']['native_reductions'] == 0
    for package in ('exported', 'exported_cy'):
        script=f'''from {package} import mx_model
s=mx_model.Values
s.stride=0
try: s.dynamic_stride(1)
except ValueError as error: assert str(error)=='range() arg 3 must not be zero'
else: raise AssertionError('zero range step did not raise')
'''
        subprocess.run([sys.executable,'-B','-c',script],cwd=tmp_path,check=True)
