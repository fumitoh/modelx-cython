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


@pytest.mark.parametrize("sample_dir, model", [["basicterm_s", "BasicTerm_S"],
                                               ["basicterm_sc", "BasicTerm_SC"]], indirect=["sample_dir"])
@pytest.mark.parametrize("target", ["mx2cy", pytest.param("main", marks=pytest.mark.skip(reason="Skipping 'main' target"))])
def test_mx2cy_with_lifelib(sample_dir, target, model):
    import lifelib
    import modelx as mx

    work_dir = sample_dir # / "basicterm_s"
    lifelib.create('basiclife', work_dir / 'basiclife')
    mx.read_model(work_dir / 'basiclife' / model).export(work_dir / (model + '_nomx'))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
         "--spec", str(work_dir / "spec.py"),
         "--sample", str(work_dir / "sample.py")]

    if target == "mx2cy":
        assert subprocess.run(argv, env=env).returncode == 0
    elif target == "main":
        from modelx_cython.cli import main
        assert main(argv[1:], sys.stdout, sys.stderr) == 0
    
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0

    result = subprocess.run(
        [sys.executable, str(work_dir / "benchmark.py")],
        env=env,
        capture_output=True,
        text=True
    )
    print(result.stdout.strip())
    assert result.returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["ref_space", "RefSpace"],
                                               ["nested_params", "NestedParams"],
                                               ["duplicated_params", "DuplicatedParams"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("target", ["mx2cy", pytest.param("main", marks=pytest.mark.skip(reason="Skipping 'main' target"))])
def test_mx2cy_with_ref_space(sample_dir, target, model):
    import modelx as mx

    work_dir = sample_dir
    mx.read_model(work_dir / model).export(work_dir / (model + '_nomx'))
    del mx.get_models()[model]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            "--spec", str(work_dir / "spec.py"),
            "--sample", str(work_dir / "sample.py")]

    if target == "mx2cy":
        assert subprocess.run(argv, env=env).returncode == 0
    elif target == "main":
        from modelx_cython.cli import main
        assert main(argv[1:], sys.stdout, sys.stderr) == 0

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["no_spec", "NoSpec"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("target", ["mx2cy", pytest.param("main", marks=pytest.mark.skip(reason="Skipping 'main' target"))])
@pytest.mark.parametrize("allow_spec", [True, False])
def test_no_spec(sample_dir, target, model, allow_spec):
    import modelx as mx

    work_dir = sample_dir
    mx.read_model(work_dir / model).export(work_dir / (model + '_nomx'))
    del mx.get_models()[model]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py")]

    if allow_spec:
        argv.append("--allow-spec")

    if target == "mx2cy":
        assert subprocess.run(argv, env=env).returncode == int(not allow_spec)
    elif target == "main":
        from modelx_cython.cli import main
        if allow_spec:
            assert main(argv[1:], sys.stdout, sys.stderr) == int(not allow_spec)
        else:
            with pytest.raises(FileNotFoundError):
                main(argv[1:], sys.stdout, sys.stderr)

    if allow_spec:
        assert subprocess.run(
            [sys.executable, str(work_dir / "assert_cy.py")],
            env=env
        ).returncode == 0
