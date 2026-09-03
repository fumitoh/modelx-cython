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


# The Spaces of TradLife_A that every Projection[i] shares
TRADLIFE_A_SHARED = ['InputData', 'Economic', 'Assumptions', 'PolicyAttrs',
                     'CommTable']


def export_has_locked_spaces():
    """Whether modelx is new enough to export locked Spaces (v0.33.0)"""
    import inspect
    import modelx as mx
    return "locked_spaces" in inspect.signature(
        mx.core.model.Model.export).parameters


@pytest.mark.parametrize("sample_dir, model, locked",
                         [["tradlife_a", "TradLife_A", False],
                          ["tradlife_a", "TradLife_A", True],
                          ["tradlife_a_ex1", "TradLife_A_EX1", False]],
                         indirect=["sample_dir"])
def test_mx2cy_with_annuallife(sample_dir, model, locked):
    """Translate and compile TradLife_A; with ``locked`` the shared Spaces
    are exported with ``locked_spaces`` and assert_cy.py also runs the
    compiled model from a thread pool."""
    import lifelib
    import modelx as mx

    if locked and not export_has_locked_spaces():
        pytest.skip("modelx export() has no locked_spaces parameter")

    work_dir = sample_dir
    lifelib.create('annuallife', work_dir / 'annuallife')

    kwargs = {'locked_spaces': TRADLIFE_A_SHARED} if locked else {}
    mx.read_model(work_dir / 'annuallife' / model).export(
        work_dir / (model + '_nomx'), **kwargs)
    del mx.get_models()[model]

    # The exported model loads input.xlsx from its parent directory
    shutil.copy(work_dir / 'annuallife' / 'input.xlsx', work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--spec", str(work_dir / "spec.py"),
            "--sample", str(work_dir / "sample.py")]

    assert subprocess.run(argv, env=env).returncode == 0

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0
