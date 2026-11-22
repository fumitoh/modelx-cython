"""Differential execution of defects found by the pre-coverage audit."""
import functools
from http import HTTPStatus
import json
import math
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import modelx as mx
import numpy as np
from modelx_cython.cli import create_setup


def test_pickle_restores_distinct_multidimensional_native_array_shapes(tmp_path):
    (tmp_path/'array_state.py').write_text('''import cython
@cython.cclass
class Arrays:
    small: cython.double[2][2]
    large: cython.double[5][2]
    def __init__(self):
        for i in range(2):
            for j in range(2): self.small[i][j] = 10*i+j
            for j in range(5): self.large[i][j] = 100*i+j
    def values(self):
        return ([self.small[i][j] for i in range(2) for j in range(2)],
                [self.large[i][j] for i in range(2) for j in range(5)])
''')
    create_setup('array_state',[Path('array_state.py')],tmp_path/'setup.py')
    built=subprocess.run([sys.executable,'-B','setup.py','build_ext','--inplace'],
                         cwd=tmp_path,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert built.returncode==0,built.stdout
    assert 'warning C4048' not in built.stdout,built.stdout
    subprocess.run([sys.executable,'-B','-c','''import pickle
from array_state import Arrays
original=Arrays()
assert pickle.loads(pickle.dumps(original)).values()==original.values()
'''],cwd=tmp_path,check=True)


def compile_model(tmp_path, sample, integer_checks=None):
    (tmp_path/'sample.py').write_text(sample,encoding='utf-8')
    spec = ['--no-spec']
    if integer_checks is not None:
        (tmp_path/'spec.py').write_text(repr({'integer_checks': integer_checks}),encoding='utf-8')
        spec = ['--spec','spec.py']
    subprocess.run([sys.executable,'-B','-m','modelx_cython','exported',*spec],
                   cwd=tmp_path,check=True)


def test_scope_overrides_aliases_and_arithmetic(tmp_path):
    from modelx_helpers.typed_inputs import lookup, sum_f64

    model=mx.new_model('AuditContracts')
    try:
        foreign=SimpleNamespace(Status=SimpleNamespace(OK=999),
            lookup=functools.partial(float,7), sum_f64=sum_f64,
            term=functools.partial(float), count=functools.partial(int,2))
        s=model.new_space('Values',refs={'Status':HTTPStatus,
            'other':foreign,
            'lookup':lookup,'sum_f64':sum_f64,'numbers':math,
            'downward':math.floor,'round_alias':round})
        s.new_cells('term','lambda k: 1.0 + k')
        s.new_cells('coded','lambda t, code: t + code')
        s.new_cells('call_code','lambda t: coded(t, Status.OK)').is_cached=False
        s.new_cells('flag','lambda t: t > 0')
        s.new_cells('compare_flag','lambda t: flag(t) == Status.OK').is_cached=False
        s.new_cells('count','lambda: 2')
        s.new_cells('nested_enum','''def nested_enum():
    def nested(self):
        return self.Status.OK
    return nested(other)
''')
        s.new_cells('nested_helper','''def nested_helper():
    def nested(self):
        return self.lookup()
    return nested(other)
''')
        s.new_cells('nested_sum','''def nested_sum():
    def nested(self):
        return self.sum_f64(self.term(k) for k in range(self.count()))
    return nested(other)
''')
        s.new_cells('matches','lambda t: t == Status.OK').is_cached=False
        s.new_cells('condition','lambda t: 1.0 if t == Status.OK else 0.0').is_cached=False
        s.new_cells('compound','lambda t: 1.0 if t == Status.OK or t < 0 else 0.0').is_cached=False
        s.new_cells('comparison_value','lambda t: (t == Status.OK,)').is_cached=False
        s.new_cells('aliases','lambda x: numbers.floor(x) + downward(x) + round_alias(x)').is_cached=False
        s.new_cells('long_expression','''def long_expression(t):
    a = t + t + t + t + t + t + t + t
    b = t * (t - 1)
    return a / 12.0 + b
''').is_cached=False
        s.new_cells('hygienic','''def hygienic():
    _mx_reduce_0_function = 1.0
    return sum((term(k) + _mx_reduce_0_function for k in range(count()) if k != 1), 0.0)
''')
        s.new_cells('reassigned_tail','''def reassigned_tail(t):
    a = t * 2
    a = t * 3
    return a / 12.0 + a
''').is_cached=False
        # not an IntEnum class: an ordinary Reference (read per evaluation, rebindable)
        virtual=model.new_space('VirtualValues',refs={'Status':SimpleNamespace(OK=HTTPStatus.OK)})
        for name in ('matches','condition','compound','comparison_value','coded',
                     'call_code','flag','compare_flag'):
            cell=virtual.new_cells(name,s.cells[name].formula.source)
            cell.is_cached=s.cells[name].is_cached
        model.export(tmp_path/'exported')
    finally:model.close()
    compile_model(tmp_path,'from exported import mx_model\ns=mx_model.Values\n'
        's.term(0);s.count();s.nested_enum();s.nested_helper();s.nested_sum();s.matches(200);'
        's.condition(200);s.compound(200);s.comparison_value(200);'
        's.call_code(1);s.coded(1,201);'
        's.compare_flag(1);'
        's.aliases(1.25);s.long_expression(2);s.hygienic();s.reassigned_tail(2)\n'
        'v=mx_model.VirtualValues\nv.matches(200);v.condition(200);v.compound(200);'
        'v.comparison_value(200);v.call_code(1);v.coded(1,201);v.compare_flag(1)\n',
        integer_checks='checked')     # the arithmetic checks below are the checked mode's (owner's rule)
    admission=json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    info=lambda name:admission[f'exported_cy._mx_classes._c_Values._f_{name}']
    assert info('nested_sum')['native_reductions']==0
    assert info('hygienic')['native_reductions']==0      # a sum with a start argument is not lowered (R-6)
    script='''import math
from http import HTTPStatus
from types import SimpleNamespace
from PACKAGE import mx_model
s=mx_model.Values
assert s.nested_enum()==999
assert s.nested_helper()==7.0
assert s.nested_sum()==1.0
assert s.hygienic()==2.0
assert s.matches(200) is True and s.matches(201) is False
assert s.call_code(1)==201
class OverrideDependency(type(s)):
    def coded(self, t, code): return 1000+t+code
override=OverrideDependency(mx_model)
override._mx_copy_refs(s,s)
assert override.call_code(1)==1201
if type(s).__module__.startswith('exported_cy'):
    for parent in (type(s),type('Middle',(type(s),),{})):
        for name in ('Status',):
            invalid=type('ChangedSchema',(parent,),{name:object()})
            try: invalid(mx_model)
            except TypeError as actual:
                assert name in str(actual) and 'rebuild' in str(actual)
            else: raise AssertionError('changed compiled definition was accepted')
    try: s.Status=SimpleNamespace(OK=201)
    except AttributeError: pass
    else: raise AssertionError('an IntEnum Reference was rebindable in the compiled model')
    assert s.Status is HTTPStatus
v=mx_model.VirtualValues
v.Status=SimpleNamespace(OK=201)
assert v.matches(200) is False and v.matches(201) is True
assert v.condition(200)==0.0 and v.condition(201)==1.0
assert v.call_code(1)==202
class BoolMember:
    def __eq__(self, value): return type(value) is bool and value
v.Status=SimpleNamespace(OK=BoolMember())
assert v.compare_flag(1) is True
class EqualityValue:
    def __bool__(self): raise AssertionError('unexpected comparison coercion')
equality_value=EqualityValue()
class ReturnsObject:
    def __eq__(self, value): return equality_value
v.Status=SimpleNamespace(OK=ReturnsObject())
assert v.comparison_value(200)[0] is equality_value
class CountTruth:
    calls=0
    def __bool__(self):
        self.calls+=1
        return self.answer
truth=CountTruth()
class CountEquality:
    def __eq__(self, value): return truth
v.Status=SimpleNamespace(OK=CountEquality())
for answer in (False,True):
    truth.answer=answer
    truth.calls=0
    assert v.compound(200)==float(answer)
    assert truth.calls==1
class EqualTo200:
    def __eq__(self, value): return value == 200
v.Status=SimpleNamespace(OK=EqualTo200())
assert v.matches(200) is True and v.matches(201) is False
failure=ValueError('enum comparison')
class BrokenMember:
    def __eq__(self, value): raise failure
v.Status=SimpleNamespace(OK=BrokenMember())
try:
    v.matches(200)
except ValueError as actual:
    assert actual is failure
else:
    raise AssertionError('subclass comparison exception lost')
for x in (-4.75,-0.0,1.25,17.5):
    assert s.aliases(x)==math.floor(x)*2+round(x)
s.downward=lambda x:10
assert s.aliases(1.25)==12
for t in (-(1<<32),-(1<<25),-1,0,1,(1<<25),(1<<32)):
    if abs(t)==(1<<32) and type(s).__module__.startswith('exported_cy'):
        # t*(t-1) leaves the signed range: the exact fallback is gone (checked mode), it raises
        try:
            s.long_expression(t)
        except OverflowError as error:
            assert 'Values.long_expression' in str(error),error
        else:
            raise AssertionError('long_expression: an intermediate beyond the signed range did not raise')
    else:
        assert s.long_expression(t)==(t+t+t+t+t+t+t+t)/12.0+t*(t-1)
    assert s.reassigned_tail(t)==(t*3)/12.0+t*3
'''
    for package in ('exported','exported_cy'):
        subprocess.run([sys.executable,'-B','-c',script.replace('PACKAGE',package)],cwd=tmp_path,check=True)


def test_observed_wide_integers_and_buffer_formats_fall_back_to_objects(tmp_path):
    model=mx.new_model('ObservedRepresentations')
    try:
        s=model.new_space('Values',refs={'wide':2**80,
            'a32':np.array([1.25,2.5],dtype=np.float32),
            'i32':np.array([11,22],dtype=np.int32),
            'swapped':np.array([1.25,2.5],dtype='>f8'),
            'zero':np.array(1.25),
            'native':np.array([1.25,2.5],dtype=np.float64)})
        for name in ('a32','i32','swapped','native'):
            s.new_cells(name+'_array','lambda: '+name)
            s.new_cells(name+'_at','lambda k: '+name+'_array()[k]')
        s.new_cells('echo','lambda x: x')
        s.new_cells('zero_array','lambda: zero')
        s.new_cells('zero_at','lambda: zero_array()[()]')
        s.new_cells('reference','lambda: wide')
        model.export(tmp_path/'exported')
    finally:model.close()
    compile_model(tmp_path,'from exported import mx_model\ns=mx_model.Values\n'
        's.echo(2**80);s.echo(-2**80);s.reference();s.zero_at()\n'
        'for k in (0,1):s.a32_at(k);s.i32_at(k);s.swapped_at(k);s.native_at(k)\n'
        'import json\n'
        'with open("formats.json","w") as f:json.dump({"swapped_native":s.swapped.dtype.isnative},f)\n')
    admission=json.loads((tmp_path/'exported_cy/_mx_admission.json').read_text())
    for name in ('a32_array','i32_array','zero_array','echo','reference'):
        assert admission[f'exported_cy._mx_classes._c_Values._f_{name}']['return_type']=='object'
    # Protocol 4 normalizes endianness during export; protocol 5 preserves it.
    formats=json.loads((tmp_path/'formats.json').read_text())
    swapped_type=admission['exported_cy._mx_classes._c_Values._f_swapped_array']['return_type']
    assert swapped_type==('const double[:]' if formats['swapped_native'] else 'object')
    for package in ('exported','exported_cy'):
        subprocess.run([sys.executable,'-B','-c',f'''from {package} import mx_model
s=mx_model.Values
for value in (2**80,-2**80,2**100):assert s.echo(value)==value
assert s.reference()==2**80
assert s.zero_at()==1.25
assert s.a32_at(1)==s.swapped_at(1)==s.native_at(1)==2.5
assert s.i32_at(1)==22
'''],cwd=tmp_path,check=True)


def test_prepared_helper_resources_do_not_break_model_pickling(tmp_path):
    from modelx_helpers.immutable_inputs import Table
    from modelx_helpers.typed_inputs import lookup

    model=mx.new_model('HelperSerialization')
    try:
        s=model.new_space('Values',refs={'lookup':lookup,
            'table':Table((0,1),((0.5,),(1.5,)),('value',))})
        s.new_cells('value',"lambda k: lookup(table, k, 'value')")
        model.export(tmp_path/'exported')
    finally:model.close()
    compile_model(tmp_path,'from exported import mx_model\n'
        'mx_model.Values.value(0);mx_model.Values.value(1)\n')
    subprocess.run([sys.executable,'-B','-c','''import pickle
from exported_cy import mx_model
assert mx_model.Values.value(0)==0.5
restored=pickle.loads(pickle.dumps(mx_model))
assert restored.Values.value(0)==0.5
assert restored.Values.value(1)==1.5
'''],cwd=tmp_path,check=True)
