"""D18: parallel collect reuse merges chunk totals in domain order by default.

Without ``"parallel_collect_order"`` the generated options ask for domain order, so the
four-worker total does not depend on which chunk finishes first. ``"completion"`` keeps
the round-2 behaviour on request.
"""
import subprocess
import sys
import textwrap

import modelx as mx


def test_domain_order_is_the_default_and_completion_stays_available(tmp_path):
    model = mx.new_model("DomainDefault")
    try:
        values = (1e16, 1.0, -1e16, 1.0, 2.0, 3.0, 4.0, -0.0)
        keys = tuple(range(len(values)))
        for name in ("Default", "Completion"):
            space = model.new_space(
                name, formula="lambda key: None", refs={"keys": keys, "values": values},
                collect_over={"key": keys}, collect_outputs='lambda: {"sum": {"value": value()}}',
            )
            space.new_cells("value", "lambda: values[key] + 0.0")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    spec = {"parallel_workers": 4, "spaces": {
        "Default": {"serial_collect": "reuse", "parallel_collect_reuse": True},
        "Completion": {"serial_collect": "reuse", "parallel_collect_reuse": True,
                       "parallel_collect_order": "completion"},
    }}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Default.collect(); mx_model.Completion.collect()\n",
        encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--spec", "spec.py"], cwd=tmp_path, check=True)
    probe = textwrap.dedent("""\
        import itertools
        import sys
        from concurrent import futures

        gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)
        before = gil_enabled()
        from exported_cy import mx_model as m
        assert gil_enabled() == before

        assert m.Default._mx_collect_options_v2()[3] is True
        assert m.Completion._mx_collect_options_v2()[3] is False
        if before:
            def no_pool(*args, **kwargs):
                raise AssertionError("GIL-enabled collect must run serially")
            futures.ThreadPoolExecutor = no_pool
        totals = {"Default": set(), "Completion": set()}
        for order in itertools.permutations(range(4)):
            futures.as_completed = lambda tasks, order=order: (tasks[i] for i in order)
            for name in totals:
                fresh = type(m)()
                totals[name].add(getattr(fresh, name).collect().sum.value)
        if before:
            assert totals == {"Default": {10.0}, "Completion": {10.0}}, totals
        else:
            # Chunks [1e16, 1], [-1e16, 1], [2, 3], [4, -0] give one domain-ordered total.
            assert totals["Default"] == {9.0}, totals
            assert len(totals["Completion"]) > 1, totals
    """)
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True, timeout=120)
