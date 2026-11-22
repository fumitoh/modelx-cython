"""Collect and direct cell calls in one session do not interfere (D108).

A reused-root collect (serial, and parallel reuse at four workers) runs each policy on a private
root instance; direct calls (``Projection[i].cell()``) use the per-policy items. This checks, in
every order and from a cell that itself calls collect, that collect returns the live result,
that direct items keep their cached values (no re-evaluation: formula evaluations are counted)
and that zero-argument shared cells of the Data Space are not cleared by collect.
"""
import subprocess
import sys
import textwrap

import modelx as mx


def _build(tmp_path):
    model = mx.new_model("CollectDirect")
    try:
        keys = tuple(range(6))
        data = model.new_space("Data", refs={"events": []})
        data.new_cells("table", '''def table():
    events.append(("table",))
    return tuple(0.01 * (k + 1) for k in range(6))
''')
        proj = model.new_space(
            "Projection", formula="lambda key: None",
            refs={"keys": keys, "data": data, "events": data.events, "fail_key": -1},
            collect_over={"key": keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        proj.new_cells("rate", "lambda: data.table()[key]")
        proj.new_cells("part", '''def part(t):
    events.append(("part", key, t))
    if key == fail_key:
        raise LookupError(key)
    if t == 0:
        return 100.0 + key
    return part(t - 1) * (1 + rate()) + t
''')
        proj.new_cells("value", "lambda: part(6)")
        proj.new_cells("pair", "lambda: (key, part(2))")
        port = model.new_space("Portfolio", refs={"proj": proj})
        port.new_cells("total", "lambda: proj.collect().sum.value")
        model.export(tmp_path / "exported", locked_spaces=["Data"])
    finally:
        model.close()
    spec = {"parallel_workers": 4, "spaces": {"Projection": {
        "serial_collect": "reuse", "parallel_collect_reuse": True}}}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\n"
        "m.Projection.collect(); m.Portfolio.total()\n"
        "for k in range(6):\n    m.Projection[k].value(); m.Projection[k].pair()\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                    "--spec", "spec.py"], cwd=tmp_path, check=True)


PROBE = textwrap.dedent('''\
    import math
    import sys
    from exported import mx_model as live
    from exported_cy import mx_model as compiled
    workers = int(sys.argv[1])
    expected = live.Projection.collect().sum.value
    direct = {k: live.Projection[k].value() for k in range(6)}
    pairs = {k: live.Projection[k].pair() for k in range(6)}

    def same(total):
        # four workers merge chunk totals in domain order: last-bit differences are documented
        return total == expected if workers == 1 else abs(total - expected) <= 4 * math.ulp(expected)

    def fresh():
        m = type(compiled)()
        m.collect_workers = workers
        m.Data.events.clear()
        return m

    def parts(m, key):
        return sum(1 for e in m.Data.events if e[0] == "part" and e[1] == key)

    def tables(m):
        return sum(1 for e in m.Data.events if e == ("table",))

    # (a) direct calls for some policies, then collect
    m = fresh()
    p = m.Projection
    held = {k: (p[k].value(), p[k].pair()) for k in (0, 3)}
    before = {k: parts(m, k) for k in (0, 3)}
    shared = m.Data.table()
    assert same(m.Projection.collect().sum.value)
    for k in (0, 3):
        n = parts(m, k)
        assert p[k].value() == held[k][0] == direct[k]
        assert p[k].pair() is held[k][1], "a direct item's cached object was recomputed"
        assert parts(m, k) == n, "a direct item re-evaluated after collect"
    assert m.Data.table() is shared and tables(m) == 1, "collect cleared a shared Data cell"

    # (b) collect, then direct calls
    m = fresh()
    assert same(m.Projection.collect().sum.value)
    collected = len(m.Data.events)
    for k in (5, 0, 2):            # 5 is the last policy collect evaluated
        assert m.Projection[k].value() == direct[k]
        assert m.Projection[k].pair() == pairs[k]
    assert len(m.Data.events) > collected, "a direct call read values left by collect"
    assert tables(m) == 1

    # (c) a cell that calls collect, between direct calls of the same Space
    m = fresh()
    p = m.Projection
    held = {k: (p[k].value(), p[k].pair()) for k in (1, 4)}
    assert same(m.Portfolio.total())
    for k in (1, 4):
        n = parts(m, k)
        assert p[k].pair() is held[k][1] and p[k].value() == direct[k]
        assert parts(m, k) == n, "a direct item re-evaluated after a collect inside a cell"
    assert p[2].value() == direct[2]
    assert same(m.Portfolio.total())      # cached
    assert tables(m) == 1

    # (d) repeated collects and direct calls interleaved; Data table evaluated once
    m = fresh()
    for _ in range(2):
        assert same(m.Projection.collect().sum.value)
        assert m.Projection[3].value() == direct[3]
    assert tables(m) == 1
    # (e) a failing first (or later) policy, then collect again and direct calls
    for failing in (0, 1, 5):
        m = fresh()
        held = m.Projection[3].pair()
        m.Projection.fail_key = failing
        try:
            m.Projection.collect()
        except LookupError as error:
            assert error.args == (failing,)
        else:
            raise AssertionError("failure swallowed")
        m.Projection.fail_key = -1
        assert same(m.Projection.collect().sum.value)
        assert [m.Projection[k].value() for k in range(6)] == [direct[k] for k in range(6)]
        assert tables(m) == 1
    print("ok", workers)
''')


def test_collect_and_direct_calls_do_not_interfere(tmp_path):
    _build(tmp_path)
    for workers in (1, 4):
        run = subprocess.run([sys.executable, "-c", PROBE, str(workers)], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        assert run.stdout.strip() == f"ok {workers}"
