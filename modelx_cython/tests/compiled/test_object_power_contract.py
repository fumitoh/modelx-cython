"""Object-result compatibility stays separate from c8's numeric execution."""
import os
import subprocess
import sys

import modelx as mx


def test_native_input_object_power_through_normal_cli(tmp_path):
    model = mx.new_model('ObjectPower')
    try:
        p = model.new_space('Values', refs={'base': 1.25, 'exponent': 2.0})
        # The power probe passes the full signed exponent range through number.
        p.new_cells('number', 'lambda n: base', is_cached=False)
        p.new_cells('power_value', 'lambda n: exponent')
        p.new_cells('integer_power', '''def integer_power(n) -> object:
    return number(n) ** n
''', is_cached=False)
        p.new_cells('local_power', '''def local_power(n) -> object:
    b, e = number(n), power_value(n)
    return b ** e
''')
        p.new_cells('object_power', '''def object_power(b, e) -> object:
    return b ** e
''')
        p.new_cells('integer_value', 'lambda n: n')
        p.new_cells('raw_minimum', '''def raw_minimum(n) -> object:
    return min(integer_value(n), number(n))
''')
        model.export(tmp_path/'exported')
    finally:
        model.close()
    (tmp_path/'sample.py').write_text('''from exported import mx_model
p=mx_model.Values
p.integer_power(2);p.local_power(2)
p.object_power(2,3);p.object_power(1.25,2.)
p.raw_minimum(1)
''')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
        '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, env=dict(os.environ),
        capture_output=True, text=True)
    (tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode == 0, build.stdout+build.stderr
    probe = '''import importlib,itertools,math,struct,sys
m=importlib.import_module(sys.argv[1]).mx_model
def outcome(operation):
 try:
  value=operation()
  if type(value) is float:
   return ('float','nan' if math.isnan(value) else struct.pack('d',value))
  if type(value) is complex:
   return ('complex',*(('nan' if math.isnan(x) else struct.pack('d',x)) for x in (value.real,value.imag)))
  return (type(value),value)
 except Exception as error:return ('error',type(error),error.args)
bases=[-1e308,-2.,-1.,-1e-308,-0.,0.,1e-308,.5,1.,2.,1e308,float('nan'),float('inf'),-float('inf')]
exponents=[-1e308,-3.,-2.5,-1.,-.5,-0.,0.,.5,1.,2.,3.,1e308,float('nan'),float('inf'),-float('inf')]
for b,e in itertools.product(bases,exponents):
 p=type(m)().Values;p.base=b;p.exponent=e
 assert outcome(lambda:p.local_power(2))==outcome(lambda:b**e),(b,e)
for b,e in itertools.product(bases,[-(1<<63),-1000,-3,-1,0,1,2,3,1000,(1<<63)-1]):
 p=type(m)().Values;p.base=b
 assert outcome(lambda:p.integer_power(e))==outcome(lambda:b**e),(b,e)
p=type(m)().Values
assert p.raw_minimum(1)==1 and type(p.raw_minimum(1)) is int
for b,e in [(2,1000),(2,-3),(-1.,.5),(complex(1,2),3),(1<<100,2)]:
 assert outcome(lambda:p.object_power(b,e))==outcome(lambda:b**e)
class Custom:
 def __pow__(self,other):return (self,other,type(other))
for e in [True,2,2.,object()]:
 b=Custom();assert p.object_power(b,e)==b**e
class Reflected:
 def __rpow__(self,other):return (other,type(other))
for b in [True,2,2.,object()]:
 e=Reflected();assert p.object_power(b,e)==b**e
'''
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             env=dict(os.environ), capture_output=True, text=True)
        (tmp_path/(package+'-validation.log')).write_text(run.stdout+run.stderr)
        assert run.returncode == 0, run.stdout+run.stderr
