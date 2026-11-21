"""audit-routes P3: the export's generic collect loop builds and frees each policy root through the
class's generated construction and release (D116), as the evaluator and the direct route do.

The generic loop runs when the evaluator cannot: under a profile or trace function, inside another
collect, for non-native coordinates, or with workers but no parallel reuse. Its value equals live,
each policy root is freed by its reference count (no cyclic garbage of the Space class), a failing
policy frees its root too, and direct items are not touched.
"""
import subprocess
import sys
import textwrap

import modelx as mx


def _build(tmp_path):
    model = mx.new_model("CollectFallback")
    try:
        keys = tuple(range(5))
        data = model.new_space("Data")
        data.new_cells("table", "lambda: tuple(0.01 * (k + 1) for k in range(5))")
        proj = model.new_space(
            "Projection", formula="lambda key: None",
            refs={"keys": keys, "data": data, "fail_key": -1},
            collect_over={"key": keys}, collect_outputs='lambda: {"sum": {"value": value()}}')
        proj.new_cells("rate", "lambda: data.table()[key]")
        proj.new_cells("part", '''def part(t):
    if key == fail_key:
        raise LookupError(key)
    if t == 0:
        return 100.0 + key
    return part(t - 1) * (1 + rate()) + t
''')
        proj.new_cells("value", "lambda: part(6)")
        model.export(tmp_path / "exported", locked_spaces=["Data"])
    finally:
        model.close()
    spec = {"parallel_workers": 1, "spaces": {"Projection": {"serial_collect": "reuse"}}}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\n"
        "m.Projection.collect()\n"
        "for k in range(5):\n    m.Projection[k].value()\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                    "--spec", "spec.py"], cwd=tmp_path, check=True)


PROBE = textwrap.dedent('''\
    import gc
    import sys
    from exported import mx_model as live
    from exported_cy import mx_model as compiled
    expected = live.Projection.collect().sum.value
    direct = {k: live.Projection[k].value() for k in range(5)}
    kind = type(compiled.Projection)

    def alive():
        return sum(1 for o in gc.get_objects() if type(o) is kind)

    def generic(call):
        # a profile function makes collect_from_space skip the evaluator (the export's generic loop)
        sys.setprofile(lambda *args: None)
        try:
            return call()
        finally:
            sys.setprofile(None)

    # collect results are cached per base, so each case runs on a fresh model
    gc.collect()
    gc.disable()
    m = type(compiled)()
    p = m.Projection
    item = p[2]
    held = item.value()
    before = alive()
    total = generic(lambda: p.collect().sum.value)
    assert total == expected, (total, expected)
    assert alive() == before, ("policy roots left to the cyclic GC", alive() - before)
    # direct items are other instances: untouched
    assert p[2] is item and item.value() == held == direct[2]
    assert [p[k].value() for k in range(5)] == [direct[k] for k in range(5)]

    m = type(compiled)()
    before = alive()
    assert m.Projection.collect().sum.value == expected     # the evaluator, same value
    assert alive() == before

    # a failing policy on the generic loop: the error propagates and its root is freed too
    m = type(compiled)()
    m.Projection.fail_key = 3
    before = alive()
    try:
        generic(lambda: m.Projection.collect())
    except LookupError as error:
        assert error.args == (3,)
    else:
        raise AssertionError("failure swallowed")
    assert alive() == before, ("failed policy root left to the cyclic GC", alive() - before)
    gc.enable()
    print("ok")
''')


def test_generic_collect_frees_roots_through_shared_release(tmp_path):
    _build(tmp_path)
    run = subprocess.run([sys.executable, "-c", PROBE], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == "ok"
