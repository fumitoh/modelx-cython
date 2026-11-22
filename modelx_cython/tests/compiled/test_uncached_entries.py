"""D139: a compiled call to an uncached Cell on a proven receiver is a direct C call.

`surv(s, t)` and `tick(t)` are uncached (``is_cached = False``). A caller in the same class (collect
route) and a caller holding a typed item local (direct route) call the Cell's private entry behind the
same exact-receiver test as a cached Cell's entry: no cache lookup, no storage, no readiness flag. The
entry checks integer arguments as the cached entries do, and Python callers keep the public method.
`surv` recurses through itself and the cached `rate`; the cached `inforce` recurses through `surv`
(mixed cached/uncached recursion). Each call of `tick` evaluates it: `ticks(n)` calls it n times and
appends n entries to a shared log, a second `ticks(n)` is a cache hit, and each Python call evaluates
again. Compiled values equal the exported model at one and four workers and on the direct route.
"""
import json
import subprocess
import sys
import textwrap

import modelx as mx

PREFIX = 'exported_cy._mx_classes._c_Projection.'


def _build(tmp_path):
    keys = tuple(range(1, 9))
    model = mx.new_model('UncachedEntries')
    try:
        projection = model.new_space(
            'Projection', formula='lambda key: None', refs={'keys': keys, 'log': []},
            collect_over={'key': keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        projection.new_cells('rate', 'lambda t: 0.01 * ((t + key) % 7)')
        projection.new_cells('surv', textwrap.dedent('''\
            def surv(s, t):
                if t <= s:
                    return 1.0
                return surv(s, t - 1) * (1.0 - rate(t - 1))
            '''), is_cached=False)
        projection.new_cells('inforce', textwrap.dedent('''\
            def inforce(t):
                if t == 0:
                    return 1.0
                return inforce(t - 1) * surv(t - 1, t)
            '''))
        # D139b: an uncached reader of a dense Cell at its own argument takes the proven inline hit
        projection.new_cells('df', 'lambda t: 1.0 / (1.0 + 0.01 * key) ** t')
        projection.new_cells('discount', 'lambda t: surv(0, t) * df(t)', is_cached=False)
        projection.new_cells('tick', textwrap.dedent('''\
            def tick(t):
                log.append(t)
                return 0.5 * t
            '''), is_cached=False)
        projection.new_cells('ticks', textwrap.dedent('''\
            def ticks(n):
                total = 0.0
                for i in range(n):
                    total += tick(i)
                return total
            '''))
        projection.new_cells('value', textwrap.dedent('''\
            def value():
                total = 0.0
                for t in range(12):
                    for s in range(0, t + 1, 1 + key % 3):
                        total += surv(s, t) * (t + 1)
                    total += inforce(t) + discount(t)
                return total
            '''))
        portfolio = model.new_space('Portfolio', refs={'projection': projection})
        portfolio.new_cells('direct', textwrap.dedent('''\
            def direct():
                total = 0.0
                for key in projection.keys:
                    item = projection[key]
                    for t in range(12):
                        total += item.surv(0, t) + item.inforce(t)
                return total
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'p = mx_model.Projection\n'
        'p.collect(); mx_model.Portfolio.direct()\n'
        'for key in (1, 2, 3):\n'
        '    p[key].value(); p[key].ticks(5); p[key].tick(2)\n'
        '    [p[key].surv(s, t) for s in range(12) for t in range(12)]\n'
        '    [p[key].df(t) + p[key].rate(t) for t in range(30)]\n', encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)


PROBE = textwrap.dedent('''\
    import json
    import sys
    from importlib import import_module
    m = import_module(sys.argv[1]).mx_model
    p = m.Projection
    out = {}
    if sys.argv[1] == 'exported_cy':
        serial = type(m)()
        serial.collect_workers = 1
        out['collect1'] = serial.Projection.collect().sum.value
        out['collect1_again'] = serial.Projection.collect().sum.value
    else:
        out['collect1'] = out['collect1_again'] = p.collect().sum.value
    out['collect4'] = p.collect().sum.value
    out['direct'] = m.Portfolio.direct()
    item = p[5]
    out['surv'] = [item.surv(s, t) for s in range(12) for t in range(12)]
    out['inforce'] = [item.inforce(t) for t in range(12)]
    out['value'] = item.value()
    out['discount'] = [item.discount(t) for t in (0, 5, 11, 12, 20)]   # 12 and 20: outside the domain
    log = item.log
    del log[:]
    counts = [item.ticks(5)]
    counts.append(len(log))
    item.ticks(5)                       # a cache hit: tick is not evaluated
    counts.append(len(log))
    for _ in range(3):
        item.tick(4)                    # each Python call evaluates the uncached Cell
    counts.append(len(log))
    out['counts'] = counts
    out['log'] = list(log)
    errors = []
    for call in (lambda: item.surv(0, 1.5), lambda: item.surv(True, 2), lambda: item.rate(1.5)):
        try:
            call()
            errors.append(None)
        except Exception as error:
            errors.append(type(error).__name__)
    out['errors'] = errors
    print(json.dumps(out))
    ''')


def _run(tmp_path, module):
    done = subprocess.run([sys.executable, '-c', PROBE, module], cwd=tmp_path,
                          capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_uncached_cells_take_a_direct_entry_and_equal_live(tmp_path):
    _build(tmp_path)
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))

    # the build report: no cache for an uncached Cell, and a private entry for each admitted Cell
    for name in ('surv', 'tick'):
        assert admission[PREFIX + '_f_' + name]['cache'] == 'none'
    assert all(admission[PREFIX + '_f_' + name]['inline_entry'] for name in ('surv', 'tick', 'rate', 'discount'))

    live, compiled = _run(tmp_path, 'exported'), _run(tmp_path, 'exported_cy')
    for key in ('collect1', 'collect1_again', 'collect4', 'direct', 'value'):
        assert abs(compiled[key] - live[key]) <= 1e-7 + 1e-10 * abs(live[key]), key
    for key in ('surv', 'inforce', 'discount'):
        assert all(abs(a - b) <= 1e-7 + 1e-10 * abs(b) for a, b in zip(compiled[key], live[key])), key
    assert compiled['counts'] == live['counts'] == [5.0, 5, 5, 8]
    assert compiled['log'] == live['log'] == [0, 1, 2, 3, 4, 4, 4, 4]
    # integer arguments are checked as for a cached Cell
    assert compiled['errors'][0] == compiled['errors'][2] is not None
