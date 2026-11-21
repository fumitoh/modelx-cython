"""Check the registered scalar math provider on values, errors and rebinding."""

import math
import json
import subprocess
import sys
import textwrap

import modelx as mx
import pytest
from modelx_helpers import numeric_math


@pytest.mark.parametrize("namespace", [math, numeric_math])
def test_registered_scalar_math(tmp_path, namespace):
    model = mx.new_model('ScalarMath')
    try:
        values = model.new_space('Values', refs={'numbers': namespace})
        for name in ('erf', 'exp', 'isinf', 'log', 'sqrt'):
            values.new_cells(name, f'lambda x: numbers.{name}(x)')
        values.new_cells('composed', 'def composed(x):\n    root = numbers.sqrt(x)\n    return numbers.exp(numbers.log(root) / x)\n')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'v.erf(1.5); v.exp(1.5); v.isinf(1.5); v.log(1.5); v.sqrt(1.5); v.composed(1.5)\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    for name in ('erf', 'exp', 'isinf', 'log', 'sqrt'):   # W1.4: bound at build time (the build report)
        helpers = report[f'exported_cy._mx_classes._c_Values._f_{name}']['native_helpers']
        assert any(h.endswith(f'.m_{name}') for h in helpers), (name, helpers)
    probe = textwrap.dedent('''\
        import math
        from types import SimpleNamespace
        from exported_cy import mx_model
        v = mx_model.Values
        assert v.composed(2.5) == math.exp(math.log(math.sqrt(2.5)) / 2.5)
        for name in ('erf', 'exp', 'log', 'sqrt'):
            expected = getattr(math, name)
            for arg in (0.5, 1.0, 2.5, 10.0, math.inf, math.nan):
                left = getattr(v, name)(arg)
                right = expected(arg)
                assert left == right or (math.isnan(left) and math.isnan(right))
        assert v.exp(-math.inf) == 0.0 and v.erf(-math.inf) == -1.0
        assert v.isinf(math.inf) is True and v.isinf(-math.inf) is True
        assert v.isinf(math.nan) is False and v.isinf(2.5) is False
        for name, arg, error in (('sqrt', -1.0, ValueError), ('log', 0.0, ValueError),
                                 ('sqrt', -math.inf, ValueError), ('log', -math.inf, ValueError),
                                 ('exp', 1000.0, OverflowError)):
            try:
                getattr(v, name)(arg)
            except error:
                pass
            else:
                raise AssertionError(name)
        v.numbers = SimpleNamespace(log=lambda x: x + 1.0)
        assert v.log(30.0) == 31.0
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
