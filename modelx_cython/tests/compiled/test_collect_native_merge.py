"""collect_outputs with several outputs: a ``sum`` output written ``{k: cell(k) for k in D}`` (D a
Reference holding integers) or ``cell()`` is summed natively; any other output, and the ``individual``
section, is merged as the export merges it. Results, errors and the order of keys equal the exported
model's, and the build report says which outputs are native and why the others are not."""
import importlib
import json
import math
import subprocess
import sys
import textwrap

import modelx as mx


def test_native_and_python_outputs_equal_the_export(tmp_path, monkeypatch):
    (tmp_path / "collect_arithmetic.py").write_text(textwrap.dedent("""\
        class LeftFold:
            def __init__(self, value):
                self.value = value
            def __add__(self, right):
                return LeftFold(f"({self.value}+{right.value})")
            def __repr__(self):
                return self.value

        def keyword_echo(**kwargs):
            return kwargs["self"]
        """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    arithmetic = importlib.import_module("collect_arithmetic")
    model = mx.new_model("NativeMerge")
    try:
        keys = (3, 1, 4, 5, 9, 2)
        months = (0, 1, 2, 5, 3)            # not contiguous, not sorted: positions in D, not values
        empty = ()
        common = {"keys": keys, "months": months, "empty": empty, "LeftFold": arithmetic.LeftFold,
                  "keyword_echo": arithmetic.keyword_echo}
        outputs = {
            "Mixed": ('lambda: {"individual": {"state": {t: cf(t) for t in months}}, '
                      '"sum": {"pv": pv(), "cf": {t: cf(t) for t in months}, '
                      '"none": {t: cf(t) for t in empty}}}'),
            "MixedPython": ('lambda: {"sum": {"pv": pv(), "keys": [key], '
                           '"fold": LeftFold(str(key))}}'),
            "RootLocal": ('lambda: {"individual": {"state": {root: cf(root) for root in months}}, '
                          '"sum": {"pv": pv()}}'),
            "RootUnused": ('lambda: {"individual": {"state": [cf(0) for root in months]}, '
                           '"sum": {"pv": pv()}}'),
            "Keyword": ('lambda: {"individual": {"state": keyword_echo(self=3)}, '
                        '"sum": {"pv": pv()}}'),
            "Duplicate": 'lambda: {"sum": {"pv": pv(), "pv": 1}}',
            "CellDomain": 'lambda: {"sum": {"cf": {t: cf(t) for t in domain()}}}',
            "MissingKey": 'lambda: {"sum": {"pv": pv(), "cf": {t: cf(t) for t in range(size())}}}',
        }
        for name, collect_outputs in outputs.items():
            space = model.new_space(name, formula="lambda key: None", refs=dict(common),
                                    collect_over={"key": keys}, collect_outputs=collect_outputs)
            space.new_cells("cf", "lambda t: key / 3.0 + t * 0.1 + (1e16 if key == 4 and t == 0 else 0.0)")
            space.new_cells("pv", "lambda: sum(cf(t) for t in months) / 7.0")
            space.new_cells("domain", "lambda: months")
            space.new_cells("size", "lambda: 2 if key == 5 else 3")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    spec = {"spaces": {name: {"serial_collect": "reuse"} for name in outputs}}
    for name in ("Mixed", "MixedPython"):
        spec["spaces"][name].update(parallel_collect_reuse=True, parallel_collect_order="completion")
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(textwrap.dedent("""\
        from exported import mx_model as m
        m.Mixed.collect(); m.CellDomain.collect()
        m.MixedPython.collect(); m.RootLocal.collect(); m.RootUnused.collect(); m.Keyword.collect(); m.Duplicate.collect()
        for key in m.MissingKey.keys:
            m.MissingKey[key].pv(); m.MissingKey[key].cf(0); m.MissingKey[key].size()
        """), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                    "--spec", "spec.py"], cwd=tmp_path, check=True)
    probe = textwrap.dedent("""\
        import json, sys
        package = sys.argv[1]
        Model = type(__import__(package).mx_model)
        def plain(value):
            if hasattr(value, '_fields'):
                return {k: plain(v) for k, v in value._fields.items()}
            if isinstance(value, dict):
                return [[repr(k), plain(v)] for k, v in value.items()]
            return repr(value)
        out = {}
        for workers in (1, 4):
            m = Model()
            m.collect_workers = workers
            out[f'mixed{workers}'] = plain(m.Mixed.collect())
            out[f'python{workers}'] = plain(m.MixedPython.collect())
            out[f'root{workers}'] = plain(m.RootLocal.collect())
            out[f'unused{workers}'] = plain(m.RootUnused.collect())
            out[f'keyword{workers}'] = plain(m.Keyword.collect())
            out[f'duplicate{workers}'] = plain(m.Duplicate.collect())
            out[f'cell{workers}'] = plain(m.CellDomain.collect())
            try:
                m.MissingKey.collect()
                out[f'missing{workers}'] = 'no error'
            except Exception as error:
                out[f'missing{workers}'] = [type(error).__name__, str(error)]
        out['native'] = {name: getattr(m, name)._mx_collect_options_v2()[1] is not None
                         if hasattr(getattr(m, name), '_mx_collect_options_v2') else None
                         for name in ('Mixed', 'MixedPython', 'RootLocal', 'RootUnused', 'Keyword',
                                      'Duplicate', 'CellDomain', 'MissingKey')}
        if out['native']['Mixed']:
            out['mixed_parallel'] = m.Mixed._mx_collect_options_v2()[2:4]
            out['python_parallel'] = m.MixedPython._mx_collect_options_v2()[2]
        print(json.dumps(out))
        """)
    results = {}
    for package in ("exported", "exported_cy"):
        done = subprocess.run([sys.executable, "-c", probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        results[package] = json.loads(done.stdout)
    live, compiled = results["exported"], results["exported_cy"]
    assert compiled["native"] == {"Mixed": True, "MixedPython": True, "RootLocal": False, "RootUnused": False,
                                  "Keyword": True, "Duplicate": False, "CellDomain": False, "MissingKey": True}
    assert compiled["mixed_parallel"] == [True, True]  # private parallel reuse, domain-order merge
    assert compiled["python_parallel"] is False        # generic worker results are folded per policy
    for key in ("mixed1", "cell1", "missing1", "python1", "python4", "root1", "root4",
                "unused1", "unused4", "keyword1", "keyword4", "duplicate1", "duplicate4"):
        assert compiled[key] == live[key], key
    assert compiled["python4"]["sum"]["keys"] == "[3, 1, 4, 5, 9, 2]"
    assert compiled["python4"]["sum"]["fold"] == "(((((3+1)+4)+5)+9)+2)"
    assert compiled["duplicate4"]["sum"]["pv"] == "6"
    # 4 workers: the individual section in domain order and with the serial values
    assert compiled["mixed4"]["individual"] == live["mixed1"]["individual"]
    assert compiled["missing4"] == live["missing4"] == live["missing1"]
    assert compiled["missing1"][0] == "ValueError" and "structure changed" in compiled["missing1"][1]
    mixed = compiled["mixed1"]
    assert [entry[0] for entry in mixed["sum"]["cf"]] == ["0", "1", "2", "5", "3"]
    assert mixed["sum"]["none"] == []
    parallel_sum = compiled["mixed4"]["sum"]
    assert parallel_sum["none"] == []
    assert math.isclose(float(parallel_sum["pv"]), float(mixed["sum"]["pv"]), rel_tol=1e-12)
    assert [key for key, _ in parallel_sum["cf"]] == [key for key, _ in mixed["sum"]["cf"]]
    for (_, got), (_, want) in zip(parallel_sum["cf"], mixed["sum"]["cf"]):
        assert math.isclose(float(got), float(want), rel_tol=1e-12)
    report = json.loads((tmp_path / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))
    def leaves(space):
        entry = report[f"exported_cy._mx_classes._c_{space}._f_collect_outputs"]
        return {(leaf["mode"], leaf["output"]): (leaf["native"], leaf.get("reason", ""))
                for leaf in entry["collect_outputs"]}
    mixed_leaves = leaves("Mixed")
    assert {key: native for key, (native, _) in mixed_leaves.items()} == {
        ("individual", ""): False, ("sum", "pv"): True, ("sum", "cf"): True,
        ("sum", "none"): True}
    assert leaves("MixedPython")[("sum", "pv")][0] and not leaves("MixedPython")[("sum", "fold")][0]
    assert "Reference" in leaves("CellDomain")[("sum", "cf")][1]
    assert leaves("MissingKey")[("sum", "pv")][0] and not leaves("MissingKey")[("sum", "cf")][0]


def test_collect_with_other_parameters_is_native_per_argument_set(tmp_path):
    """Valuation(a, key, b).collect(a, b): the other parameters reach the compiled evaluator (protocol 2
    extension); each call has its own accumulators (allocated and freed in that call), and the collect
    cache is keyed by the arguments, so two argument sets never share a total. One item and many argument
    sets is the per-call cost case."""
    model = mx.new_model("CollectArguments")
    try:
        book = tuple(range(1, 9))
        for name, keys in (("Book", book), ("One", (7,)), ("Reduce", book)):
            months = tuple(range(12))
            space = model.new_space(name, formula="lambda a, key, b=-1: None",
                                    refs={"keys": keys, "months": months}, collect_over={"key": keys},
                                    collect_outputs='lambda: {"sum": {"pv": pv(), "cf": {t: cf(t) for t in months}}}')
            space.new_cells("cf", "lambda t: key * 0.37 + a / 7.0 + b * 0.011 + t / 3.0")
            space.new_cells("pv", "lambda: sum(cf(t) for t in months) / (1.0 + a / 100.0)")
            if name == "Reduce":
                space.new_cells("need", "lambda: a + key + b")
        space = model.new_space(
            "ArgumentDomain", formula="lambda key, periods: None",
            refs={"keys": book}, collect_over={"key": book},
            collect_outputs='lambda: {"sum": {"pv": pv(), "cf": {t: cf(t) for t in periods}}}')
        space.new_cells("pv", "lambda: key * 0.25")
        space.new_cells("cf", "lambda t: pv() + t")
        model.export(tmp_path / "exported")
    finally:
        model.close()
    spec = {"spaces": {name: {"serial_collect": "reuse", "parallel_collect_reuse": True} for name in ("Book", "One")}}
    spec["spaces"]["Reduce"] = {"serial_collect": "reduce",
                                "cells_param_size": {"t": 12}, "param_bound_checks": {"t": "need"}}
    spec["spaces"]["ArgumentDomain"] = {"serial_collect": "reuse", "parallel_collect_reuse": True}
    (tmp_path / "spec.py").write_text(repr(spec), encoding="utf-8")
    (tmp_path / "sample.py").write_text(
        "from exported import mx_model as m\n"
        "for a in (0, 12, 24):\n"
        "    for b in (-1, 0, 6):\n"
        "        m.Book.collect(a, b); m.One.collect(a, b); m.Reduce.collect(a, b)\n"
        "        for k in m.Reduce.keys:\n"
        "            m.Reduce[a, k, b].need()\n"
        "m.ArgumentDomain.collect((2, 0, 1))\n",
        encoding="utf-8")
    subprocess.run([sys.executable, "-m", "modelx_cython", "exported", "--sample", "sample.py",
                    "--spec", "spec.py"], cwd=tmp_path, check=True)
    probe = textwrap.dedent("""\
        import json, sys
        Model = type(__import__(sys.argv[1]).mx_model)
        def plain(r):
            return [repr(r.sum.pv), [[k, repr(v)] for k, v in r.sum.cf.items()]]
        out = {}
        for workers in (1, 4):
            m = Model()
            m.collect_workers = workers
            out[f'book{workers}'] = [plain(m.Book.collect(a, b)) for a in (0, 12, 24) for b in (-1, 0, 6)]
            out[f'one{workers}'] = [plain(m.One.collect(a, b)) for a in range(0, 300, 12) for b in range(-1, 7)]
            out[f'argument_domain{workers}'] = plain(m.ArgumentDomain.collect((1, 2)))
            if workers == 1:
                out['reduce'] = [plain(m.Reduce.collect(a, b)) for a in (0, 12, 24) for b in (-1, 0, 6)]
        out['native'] = [hasattr(getattr(m, n), '_mx_collect_options_v2') and getattr(m, n)._mx_collect_options_v2()[1] is not None
                         for n in ('Book', 'One', 'Reduce', 'ArgumentDomain')]
        if out['native'][2]:
            try:
                Model().Reduce.collect(25, 6)
            except IndexError as error:
                out['reduce_bound'] = str(error)
        print(json.dumps(out))
        """)
    results = {}
    for package in ("exported", "exported_cy"):
        done = subprocess.run([sys.executable, "-c", probe, package], cwd=tmp_path, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        results[package] = json.loads(done.stdout)
    live, compiled = results["exported"], results["exported_cy"]
    assert compiled["native"] == [True, True, True, True]
    assert compiled["book1"] == live["book1"] and compiled["one1"] == live["one1"] == compiled["one4"]
    assert compiled["reduce"] == live["reduce"]
    assert "policy 8 needs t for need() = 39" in compiled["reduce_bound"]
    for workers in (1, 4):
        assert compiled[f"argument_domain{workers}"] == live[f"argument_domain{workers}"]
    report = json.loads((tmp_path / "exported_cy/_mx_admission.json").read_text(encoding="utf-8"))
    leaves = report["exported_cy._mx_classes._c_ArgumentDomain._f_collect_outputs"]["collect_outputs"]
    assert {leaf["output"]: leaf["native"] for leaf in leaves} == {"pv": True, "cf": False}
    assert len({json.dumps(r) for r in compiled["book1"]}) == 9            # nine argument sets, nine totals
    # 4 workers: chunk totals merge in the parallel order, so the last bits may differ from the serial sum
    for got, want in zip(compiled["book4"], live["book1"]):
        assert math.isclose(float(got[0]), float(want[0]), rel_tol=1e-12)
        assert [k for k, _ in got[1]] == [k for k, _ in want[1]]
        for (_, x), (_, y) in zip(got[1], want[1]):
            assert abs(float(x) - float(y)) <= 1e-12 * abs(float(y))
