import subprocess
import sys
import textwrap

import modelx as mx


def test_parallel_collect_reuse_merges_completed_chunks_and_retries(tmp_path):
    model = mx.new_model("PrivateCollect")
    try:
        values = (1e16, 1.0, -1e16, 1.0, 2.0, 3.0, 4.0, -0.0)
        keys = tuple(range(len(values)))
        for name in ("Reused", "Ordered", "Ordinary"):
            space = model.new_space(
                name, formula="lambda key: None", refs={"keys": keys, "values": values, "adjust": float},
                collect_over={"key": keys}, collect_outputs='lambda: {"sum": {"value": value()}}',
            )
            space.new_cells("part", "lambda t: values[key] + t")
            space.new_cells("value", "lambda: adjust(part(0))")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    spec = {"parallel_workers": 4, "spaces": {
        "Reused": {"serial_collect": "reuse", "parallel_collect_reuse": True,
                   "parallel_collect_order": "completion"},
        "Ordered": {"serial_collect": "reuse", "parallel_collect_reuse": True,
                    "parallel_collect_order": "domain"},
    }}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Reused.collect(); mx_model.Ordered.collect(); mx_model.Ordinary.collect()\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported",
                    "--sample", "sample.py", "--spec", "spec.py"], cwd=tmp_path, check=True)
    assert (tmp_path / 'exported/_mx_sys.py').read_bytes() == (tmp_path / 'exported_cy/_mx_sys.py').read_bytes()
    probe = textwrap.dedent("""\
        from concurrent import futures
        import sys
        from threading import Barrier, Lock, get_ident

        gil = getattr(sys, "_is_gil_enabled", lambda: True)
        before = gil()
        from exported_cy import mx_model as m
        assert gil() == before

        pool = futures.ThreadPoolExecutor
        starts = Barrier(4, timeout=15)
        lock = Lock()
        threads = set()
        class Observer(pool):
            def __init__(self, *args, **kwargs):
                assert not before, "GIL-enabled collect must run serially"
                super().__init__(*args, **kwargs)

            def submit(self, fn, /, *args, **kwargs):
                assert args[-1] is not None
                def run():
                    with lock:
                        threads.add(get_ident())
                    starts.wait()
                    return fn(*args, **kwargs)
                return super().submit(run)

        def completion_order(tasks):
            for index in (2, 0, 1, 3):
                yield tasks[index]

        from exported_cy import _mx_sys
        assert _mx_sys._mx_collect_adapter_version == 2
        if before:
            futures.ThreadPoolExecutor = Observer
        assert m.Ordinary.collect().sum.value == 10.0
        futures.ThreadPoolExecutor = Observer
        futures.as_completed = completion_order
        serial_model = type(m)()
        serial_model.collect_workers = 1
        assert serial_model.Ordered.collect().sum.value == 10.0
        assert m.Ordered.collect().sum.value == (10.0 if before else 9.0)
        failure = LookupError("retry")
        def broken(x):
            if x == -1e16:
                raise failure
            return x
        m.Reused.adjust = broken
        try:
            m.Reused.collect()
        except LookupError as error:
            assert error is failure
        else:
            raise AssertionError("failure swallowed")
        first_failure = ValueError("first domain chunk")
        later_failure = ValueError("later domain chunk")
        def multiple_failures(x):
            if x == 1e16:
                raise first_failure
            if x == -1e16:
                raise later_failure
            return x
        def reversed_failures(tasks):
            for index in (2, 1, 0, 3):
                yield tasks[index]
        futures.as_completed = reversed_failures
        m.Reused.adjust = multiple_failures
        try:
            m.Reused.collect()
        except ValueError as error:
            assert error is first_failure
        else:
            raise AssertionError("first domain failure lost")
        futures.as_completed = completion_order
        m.Reused.adjust = float
        result = m.Reused.collect()
        assert result.sum.value == (10.0 if before else 8.0)
        assert m.Reused.collect() is result
        if before:
            assert not threads
        else:
            assert len(threads) >= 4
        try:
            del m.Reused[3]
        except KeyError:
            pass
        else:
            raise AssertionError("collect left a public ItemSpace")
        assert m.Reused[3].value() == 1.0
        assert gil() == before
    """)
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True, timeout=90)


def test_obsolete_collect_export_requests_reexport(tmp_path):
    model = mx.new_model("Obsolete")
    try:
        keys = (0, 1)
        projection = model.new_space(
            "Projection", formula="lambda key: None", refs={"keys": keys},
            collect_over={"key": keys}, collect_outputs='lambda: {"sum": {"value": value()}}',
        )
        projection.new_cells("value", "lambda: float(key)")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    runtime = tmp_path / "exported/_mx_sys.py"
    source = runtime.read_text(encoding="utf-8")
    marker = "_mx_collect_adapter_version = 2"
    assert marker in source
    runtime.write_text(source.replace(marker, "_mx_collect_adapter_version = 1"), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model\nmx_model.Projection.collect()\n", encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
         "--no-spec", "--translate-only"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "without export protocol 2" in result.stderr and "Re-export the model" in result.stderr
