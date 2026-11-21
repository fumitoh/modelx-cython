"""Native numeric extrema retain c8's values and Cython evaluation mechanics."""
import os
import subprocess
import sys
import textwrap

import modelx as mx


def test_ordered_builtin_extrema_native_contract(tmp_path):
    model = mx.new_model('OrderedArguments')
    try:
        space = model.new_space('Values', refs={'values': (1.0, 2.0, 3.0),
            'events': [], 'failures': (None, None, None)})
        space.new_cells('term', '''def term(index):
    events.append(index)
    if failures[index] is not None:
        raise failures[index]
    return values[index]
''')
        space.new_cells('maximum', 'lambda: max(term(0), term(1), term(2))')
        space.new_cells('minimum', 'lambda: min(term(0), term(1), term(2))')
        space.new_cells('fractional_share', '''def fractional_share():
    q = term(0)
    monthly = 1.0 - (1.0 - q) ** (1.0 / 12.0)
    return min(1.0, monthly * term(1))
''')
        space.new_cells('integer_power', '''def integer_power():
    q = term(0)
    return q ** 2
''')
        space.new_cells('nested_fractional_power', '''def nested_fractional_power():
    q = term(0) ** (2 ** -1)
    return min(1.0, q)
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\nm=mx_model.Values\nm.maximum();m.minimum();m.fractional_share();m.integer_power();m.nested_fractional_power()\n')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported',
        '--sample', 'sample.py', '--no-spec'], cwd=tmp_path, capture_output=True,
        text=True, env=dict(os.environ))
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    probe = textwrap.dedent('''
        import importlib,itertools,math,struct,sys
        model=importlib.import_module(sys.argv[1]).mx_model
        def fresh(values,failures=(None,None,None)):
            result=type(model)().Values
            result.values=values;result.events=[];result.failures=failures
            return result
        samples=[float('nan'),float('inf'),float('-inf'),0.0,-0.0,1.0,-1.0]
        for q in (0.0, 0.25, 1.0):
            space=fresh((q, 0.5, 0.0))
            assert space.fractional_share()==min(1.0,(1.0-(1.0-q)**(1.0/12.0))*0.5)
            assert space.integer_power()==q**2
            assert space.nested_fractional_power()==min(1.0,q**0.5)
        space=fresh((-1.0,0.5,0.0))
        try:space.nested_fractional_power()
        except TypeError:pass
        else:raise AssertionError('nested fractional comparison did not raise')
        # The local power may return complex; its comparison must raise the
        # original TypeError instead of being rejected by Cython or made real.
        space=fresh((2.0,0.5,0.0))
        try:space.fractional_share()
        except TypeError:pass
        else:raise AssertionError('complex comparison did not raise')
        for name,operation in [('maximum',max),('minimum',min)]:
            for values in itertools.product(samples,repeat=3):
                space=fresh(values)
                actual=getattr(space,name)();expected=operation(values)
                assert sorted(space.events)==[0,1,2],(name,space.events)
                if math.isnan(expected):assert math.isnan(actual)
                else:assert struct.pack('d',actual)==struct.pack('d',expected)
            first,second=ValueError('first'),LookupError('second')
            space=fresh((1.0,2.0,3.0),(first,second,None))
            try:getattr(space,name)()
            except (ValueError,LookupError) as error:assert error is first or error is second
            else:raise AssertionError('argument exception was lost')
            # The user permits a different first-error order. Value selection
            # and ordinary argument exceptions remain part of the contract.
            assert space.events and space.events[-1] in (0,1)
    ''')
    for package in ['exported', 'exported_cy']:
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path,
                             capture_output=True, text=True, env=dict(os.environ))
        (tmp_path / (package + '-validation.log')).write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
