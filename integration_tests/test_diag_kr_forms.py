"""diag-kr: forms the build left to Python objects, compiled natively, equal to the exported model.

- integer locals in loop bodies, from min/max/abs/int() and compared in tests (``s = t - u``,
  ``d = abs(s)``, ``a = min(max(age(t), lo), hi)``, ``x = age(t)``, ``range(s, t)`` bounds,
  ``t < n + 2 * age(t)``): C integers under D76 (checked here as regressions: no object local);
- a table local bound in several branches to the same held table Cell holds the Cell's handle,
  resolved at each binding, and its columns (Cancer_KR_S claims ``tbl = data.care_table()`` per kind);
- a ``for k in range(...)`` target with integer bounds is an integer table key part
  (Cancer_KR_S ``for k in range(1, 7): tbl.loc[k, "c"]``), not an object key;
- ``min``/``max`` of a generator over an integer vector cell (Cancer_KR_S tier_share), with
  Python's empty-iterable ValueError, and a vector sum whose element holds ``max(0.0, ...)``
  (CI_KR_S claims) are C loops; an element of an integer vector and ``T.index.asof(a)`` at an
  integer ``a`` are integer keys; a returned ``[... for s in range(..)] + list(range(..))`` is
  built into one integer vector without Python lists (CI_KR_S ci_cohort_ids); a vector rebuilt
  from itself (``val = [v * g for v in val]``, Dep_FR_S dep_cohorts) is a typed vector;
- ``for x in (E.A, E.B, ...)`` over declared enum members is a loop over the member constants
  with a C long long target when every read of the target is a comparison operand or a long long
  cell argument (Child_KR_S benefit_pp); a target read any other way (``is``) stays an object.
Compiled equals the exported model on every call (value, type, or exception).
"""
from http import HTTPStatus
import json
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd

PREFIX = 'exported_cy._mx_classes._c_Values._f_'


def _formula(source, name):
    """The generated text of a cell's formula (the dense formula when the cell has one)."""
    start = source.index(f'def _f_{name}(')
    body = source[start:]
    end = body.find('\n    @', 10)
    body = body[:end] if end > 0 else body
    marker = '_mx_dense_formula_'
    if marker in body and len(body) < 400:
        index = body[body.index(marker):].split('(')[0]
        dense = source[source.index(f'def {index}('):]
        stop = dense.find('\n@', 10)
        return dense[:stop] if stop > 0 else dense
    return body


def _build(tmp_path):
    model = mx.new_model('DiagKR')
    try:
        rates = pd.DataFrame({'rate': [1.5, 2.5, 3.5, 4.5]}, index=[1, 2, 3, 4])
        s = model.new_space('Values', refs={'rates': rates, 'Status': HTTPStatus})
        s.new_cells('table', 'lambda: rates')
        s.new_cells('age', 'lambda t: 3 + t // 4')
        s.new_cells('flow', 'lambda s: 1.0 + 0.25 * s')
        s.new_cells('horizon', 'lambda: 21')
        s.new_cells('guarantee', 'lambda: 2')
        # a vector cell: range loop targets are C integers in a module with vector cells (W1.6),
        # as in the models this test stands for
        s.new_cells('vec', 'lambda: [0.5 * k for k in range(4)]')
        s.new_cells('surv', 'lambda s, u: 1.0 / (1.0 + s + u)')
        s.new_cells('pay', textwrap.dedent('''\
            def pay(t):
                if t < 0:
                    return 0.0
                total = 0.0
                for u in range(0, min(t, horizon() - 1) + 1):
                    s = t - u
                    if u < guarantee():
                        w = 1.0
                    else:
                        w = surv(s, s + 4 * (u // 4))
                    total += flow(s) * w
                return total
            '''))
        s.new_cells('cohort', textwrap.dedent('''\
            def cohort(t, s):
                d = abs(s)
                if d <= 0 or d > t:
                    return 0.0
                return flow(d) / flow(t)
            '''))
        s.new_cells('clamped', textwrap.dedent('''\
            def clamped(t):
                a = min(max(age(t), 4), 6)
                return flow(a) + float(a)
            '''))
        s.new_cells('direct', textwrap.dedent('''\
            def direct(t):
                x = age(t)
                if x > 5:
                    return flow(x) * 2.0
                return flow(x)
            '''))
        s.new_cells('branched', textwrap.dedent('''\
            def branched(t):
                if t % 2 == 0:
                    tbl = table()
                    return float(tbl.loc[1, "rate"]) * t
                if t % 3 == 0:
                    tbl = table()
                    return float(tbl.loc[2, "rate"]) + t
                return 0.0
            '''))
        s.new_cells('keyed', textwrap.dedent('''\
            def keyed(t):
                tbl = table()
                total = 0.0
                for k in range(1, 5):
                    total = total + float(tbl.loc[k, "rate"]) * flow(t + k)
                return total
            '''))
        s.new_cells('grad', textwrap.dedent('''\
            def grad(t, k):
                if k == 1:
                    s = t - 3
                    if s < 0:
                        return 0.0
                    cohort = flow(s)
                else:
                    s = t - 2
                    if s < 0:
                        return 0.0
                    cohort = grad(s, k - 1)
                factor = 1.0
                for u in range(s, t):
                    factor = factor * flow(u)
                return cohort * factor
            '''))
        s.new_cells('scaled', textwrap.dedent('''\
            def scaled(t):
                n = 4
                val = [0.0] * n
                for z in range(n):
                    val[z] = flow(t) * z
                if t % 2 == 0:
                    g = 1.0 + 0.5
                    val = [v * g for v in val]
                return val
            '''))
        s.new_cells('within', textwrap.dedent('''\
            def within(t):
                n = 3 * age(t)
                if t < n + 2 * age(t):
                    return flow(n - 1)
                return 0.0
            '''))
        s.new_cells('ages', 'lambda: [1, 2, 3, 4]')
        s.new_cells('above', textwrap.dedent('''\
            def above(t):
                ages_ = ages()
                a = min(max(t - 2, ages_[0]), ages_[-1])
                lo = table().index.asof(a)
                hi = min(g for g in ages_ if g >= a)
                if hi == lo:
                    return float(table().loc[lo, "rate"])
                return float(table().loc[hi, "rate"]) * (a - lo) / (hi - lo)
            '''))
        s.new_cells('empty_min', 'lambda t: max(g for g in ages() if g > t)')
        s.new_cells('capped', textwrap.dedent('''\
            def capped(t):
                loan = flow(t) * 0.5
                return sum(flow(s) * max(0.0, flow(t) - loan) for s in ages())
            '''))
        s.new_cells('ids', 'lambda t: [-s for s in range(1, min(max(t, 0), 3) + 1)] + list(range(1, max(t, 0) + 1))')
        s.new_cells('over_ids', 'lambda t: sum(flow(abs(s)) * 0.5 for s in ids(t))')
        s.new_cells('weight', 'lambda code: 0.5 if code == Status.OK else 0.25 * (code - 200)')
        s.new_cells('members', textwrap.dedent('''\
            def members(t):
                total = 0.0
                for code in (Status.OK, Status.CREATED, Status.ACCEPTED):
                    if code == Status.CREATED:
                        total += 2.0 * t
                    total += weight(code) * flow(t)
                return total
            '''))
        s.new_cells('identity', textwrap.dedent('''\
            def identity(t):
                total = 0.0
                for code in (Status.OK, Status.CREATED):
                    if code is Status.OK:
                        total += 1.0
                    total += weight(code) * t
                return total
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'v.vec(); v.ages()\n'
        'for t in range(12):\n'
        '    v.pay(t); v.clamped(t); v.direct(t); v.keyed(t); v.members(t); v.identity(t)\n'
        '    v.above(t); v.capped(t); v.ids(t); v.over_ids(t); v.grad(t, 1); v.grad(t, 2); v.scaled(t); v.within(t); v.branched(t)\n'
        '    try:\n        v.empty_min(t)\n    except ValueError:\n        pass\n'
        '    [v.cohort(t, s) for s in range(-t - 1, t + 2)]\n', encoding='utf-8')
    done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py',
                           '--no-spec'], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return ((tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8'),
            json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text()))


def test_forms_compile_natively_and_equal_the_interpreter(tmp_path):
    source, report = _build(tmp_path)
    pay = _formula(source, 'pay')
    assert 's: object' not in pay and 'u: object' not in pay, pay
    cohort = _formula(source, 'cohort')
    assert 'd: object' not in cohort, cohort
    clamped = _formula(source, 'clamped')
    assert 'a: object' not in clamped, clamped
    direct = _formula(source, 'direct')
    assert 'x: object' not in direct, direct
    keyed = _formula(source, 'keyed')
    assert '.kcode(' in keyed and 'cast(object, k)' not in keyed, keyed
    assert not any(e['kind'] == 'table-key-object' for e in report[PREFIX + 'keyed']['slow_paths'])
    members = _formula(source, 'members')
    assert 'code: _mx_cy.longlong' in members and 'for code in (_mx_enum_const_' in members, members
    identity = _formula(source, 'identity')
    assert 'code: _mx_cy.longlong' not in identity, identity
    above = _formula(source, 'above')
    assert 'cast(object' not in above and f'ValueError("{_empty_min_message()}")' in above, above
    assert report[PREFIX + 'capped']['vectors']['lowered'] >= 1, report[PREFIX + 'capped']['vectors']
    assert not report[PREFIX + 'capped']['slow_paths'], report[PREFIX + 'capped']['slow_paths']
    grad = _formula(source, 'grad')
    assert 's: object' not in grad, grad      # assigned in both branches, read by range(s, t)
    branched = _formula(source, 'branched')
    assert branched.count('.f64_1(') == 2 and '.col("rate")' not in branched, branched
    within = _formula(source, 'within')
    assert 'n: object' not in within, within
    scaled = _formula(source, 'scaled')
    assert 'val: ' in scaled and 'F64Vec' in scaled and 'vec_of(val)' not in scaled, scaled
    ids = _formula(source, 'ids')
    assert 'ivec_of(' not in ids and '.data[' in ids, ids
    probe = textwrap.dedent('''\
        import json, sys
        from importlib import import_module
        v = import_module(sys.argv[1]).mx_model.Values
        out = []
        def call(f, *args):
            try:
                r = f(*args)
                out.append([type(r).__name__, repr(r)])
            except Exception as e:
                out.append(['raise', type(e).__name__])
        for t in range(12):
            for f in (v.pay, v.clamped, v.direct, v.keyed, v.members, v.identity, v.above, v.capped,
                      v.empty_min):
                call(f, t)
            # an empty vector sum is int 0 live and 0.0 compiled (D24): compared as a float
            out.append(repr(float(v.over_ids(t))))
            out.append(list(v.ids(t)))
            out.append([repr(x) for x in v.scaled(t)])
            call(v.within, t)
            call(v.branched, t)
            call(v.grad, t, 1)
            call(v.grad, t, 2)
            for s in range(-t - 1, t + 2):
                call(v.cohort, t, s)
        print(json.dumps(out))
        ''')
    results = []
    for package in ('exported', 'exported_cy'):
        run = subprocess.run([sys.executable, '-c', probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, package + run.stdout + run.stderr
        results.append(json.loads(run.stdout))
    assert results[0] == results[1], [(a, b) for a, b in zip(*results) if a != b][:10]


def _empty_min_message():
    """The running interpreter's min() message for an empty iterable (its text changed in 3.12)."""
    try:
        min(())
    except ValueError as error:
        return str(error)
