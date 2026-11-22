"""Implicit parentheses and comments survive native expression admission."""
import subprocess
import sys

import modelx as mx


def test_multiline_comparisons_keep_python_and_native_divisibility(tmp_path):
    model = mx.new_model('MultilineExpressions')
    try:
        space = model.new_space('Values')
        space.new_cells('value', 'lambda n: n * 1.0')
        space.new_cells('limit', 'lambda: 1.0')
        space.new_cells('within', '''def within(n):
    return bool(
        value(n)  # this expression borrows the call's parentheses
        <= limit()
    )
''')
        space.new_cells('divisible', '''def divisible(n, size):
    return bool(
        n % size  # the operator continues on the next line
        == 0
    )
''')
        model.export(tmp_path/'exported')
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model
for n in (0,1,2):
    mx_model.Values.within(n)
    mx_model.Values.divisible(n,2)
''', encoding='utf-8')
    subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported', '--no-spec'],
                   cwd=tmp_path, check=True)
    for package in ('exported', 'exported_cy'):
        subprocess.run([sys.executable, '-B', '-c', f'''from {package} import mx_model
s=mx_model.Values
assert [s.within(n) for n in (0,1,2)]==[True,True,False]
assert [s.divisible(n,2) for n in (0,1,2)]==[True,False,True]
try: s.divisible(2,0)
except ZeroDivisionError: pass
else: raise AssertionError('zero divisor accepted')
'''], cwd=tmp_path, check=True, timeout=30)
