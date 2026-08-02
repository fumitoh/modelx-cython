import sys
import os
import subprocess
import pathlib
import shutil
import pytest


@pytest.fixture
def sample_dir(tmp_path_factory, request):
    sample = request.param
    dst = tmp_path_factory.mktemp("temp") / "samples" / sample
    shutil.copytree(pathlib.Path(__file__).parent / "samples" / sample, dst)
    return dst


@pytest.mark.parametrize("sample_dir, model", [["tradlife_a", "TradLife_A"],
                                               ["tradlife_a_ex1", "TradLife_A_EX1"]],
                         indirect=["sample_dir"])
def test_mx2cy_with_annuallife(sample_dir, model):
    import lifelib
    import modelx as mx

    work_dir = sample_dir
    lifelib.create('annuallife', work_dir / 'annuallife')

    mx.read_model(work_dir / 'annuallife' / model).export(work_dir / (model + '_nomx'))
    del mx.get_models()[model]

    # The exported model loads input.xlsx from its parent directory
    shutil.copy(work_dir / 'annuallife' / 'input.xlsx', work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            "--spec", str(work_dir / "spec.py"),
            "--sample", str(work_dir / "sample.py")]

    assert subprocess.run(argv, env=env).returncode == 0

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0
