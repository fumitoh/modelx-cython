"""Verify input replacement in live, exported, and compiled models."""

import subprocess
import sys
import textwrap


def test_new_input_scenario_after_collect(tmp_path):
    def write_script(name, source):
        (tmp_path / name).write_text(textwrap.dedent(source), encoding="utf-8")

    def run(*args):
        subprocess.run([sys.executable, *args], cwd=tmp_path, check=True, timeout=180)

    write_script("scenario_inputs.py", """\
        from enum import Enum

        import pandas as pd


        class Basis(Enum):
            STANDARD = 0.0
            ADJUSTED = 100.0


        def assign_inputs(model, amounts, multiplier, category):
            model.Data.benefit_amounts = pd.DataFrame(
                {"amount": amounts}, index=pd.Index([1, 2], name="policy_id")
            )
            model.Data.multiplier = multiplier
            model.Data.category = category


        def replace_inputs(model):
            assign_inputs(model, [11.0, 21.0], 3.0, "ADJUSTED")


        def check_results(model, expected):
            projection = model.Projection
            assert projection.collect().sum.benefit == sum(expected)
            assert (projection[1].benefit(), projection[2].benefit()) == expected
        """)

    write_script("build_model.py", """\
        import modelx as mx
        from scenario_inputs import Basis, assign_inputs, check_results, replace_inputs

        model = mx.new_model("Benefits")
        try:
            data = model.new_space("Data")
            assign_inputs(model, [10.0, 20.0], 2.0, "STANDARD")
            data.new_cells(
                "benefits",
                'lambda: benefit_amounts',
            )
            policy_ids = (1, 2)
            projection = model.new_space(
                "Projection", formula="lambda policy_id: None",
                refs={"data": data, "Basis": Basis, "policy_ids": policy_ids},
                collect_over={"policy_id": policy_ids},
                collect_outputs='lambda: {"sum": {"benefit": benefit()}}',
            )
            projection.new_cells(
                "benefit",
                'lambda: float(data.benefits().at[policy_id, "amount"])'
                ' * data.multiplier + Basis[data.category].value',
            )
            check_results(model, (20.0, 40.0))
            model.export("benefits_nomx", locked_spaces=["Data"])
            replace_inputs(model)
            check_results(model, (133.0, 163.0))
        finally:
            model.close()
        """)
    run("build_model.py")

    write_script("sample.py", """\
        from benefits_nomx import mx_model
        from scenario_inputs import check_results, replace_inputs

        check_results(mx_model, (20.0, 40.0))
        scenario = type(mx_model)()
        replace_inputs(scenario)
        check_results(scenario, (133.0, 163.0))
        """)
    run("-m", "modelx_cython", "benefits_nomx", "--sample", "sample.py", "--no-spec")

    write_script("run_scenario.py", """\
        from importlib import import_module
        from importlib.machinery import ExtensionFileLoader
        import sys

        from scenario_inputs import check_results, replace_inputs

        package, mode = sys.argv[1:]
        model = import_module(package).mx_model
        if package.endswith("_cy"):
            classes = import_module(f"{package}._mx_classes")
            assert isinstance(classes.__loader__, ExtensionFileLoader)

        if mode == "fresh-instance":
            check_results(model, (20.0, 40.0))
            scenario = type(model)()
            replace_inputs(scenario)
            check_results(scenario, (133.0, 163.0))
            check_results(model, (20.0, 40.0))
        else:
            replace_inputs(model)
            check_results(model, (133.0, 163.0))
        """)
    for package in ("benefits_nomx", "benefits_nomx_cy"):
        run("run_scenario.py", package, "fresh-instance")
        run("run_scenario.py", package, "fresh-process")
