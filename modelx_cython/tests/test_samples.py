import logging
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


def generate_nomx(work_dir: pathlib.Path, model: str):
    import modelx as mx
    mx.read_model(work_dir / model).export(work_dir / (model + '_nomx'))
    del mx.get_models()[model]


def get_env(work_dir: pathlib.Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.mark.parametrize("sample_dir, model", [["ref_space", "RefSpace"],
                                               ["nested_params", "NestedParams"],
                                               ["duplicated_params", "DuplicatedParams"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("target", ["mx2cy", pytest.param("main", marks=pytest.mark.skip(reason="Skipping 'main' target"))])
def test_mx2cy_with_ref_space(sample_dir, target, model):

    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

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
@pytest.mark.parametrize("no_spec", [True, False])
def test_no_spec(sample_dir, target, model, no_spec):

    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py")]

    if no_spec:
        argv.append("--no-spec")

    if target == "mx2cy":
        assert subprocess.run(argv, env=env).returncode == int(not no_spec)
    elif target == "main":
        from modelx_cython.cli import main
        if no_spec:
            assert main(argv[1:], sys.stdout, sys.stderr) == int(not no_spec)
        else:
            with pytest.raises(FileNotFoundError):
                main(argv[1:], sys.stdout, sys.stderr)

    if no_spec:
        assert subprocess.run(
            [sys.executable, str(work_dir / "assert_cy.py")],
            env=env
        ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["varying_types_of_args", "VaryingArgTypes"]],
                         indirect=["sample_dir"])
def test_varying_arg_types(sample_dir, model, caplog):
    """int and float numbers are given to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--no-spec"]

    assert (result := subprocess.run(argv + ['--log-level', 'INFO'], env=env, capture_output=True, text=True)).returncode == 0
    assert "varying types given to argument 'i' in VaryingArgTypes_nomx._mx_classes._c_Space1._f_foo: int 1, float 2.0" in result.stderr
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["varying_integral_types_of_args", "VaryingIntegralArgTypes"]],
                         indirect=["sample_dir"])
def test_varying_integral_arg_types(sample_dir, model):
    """int and np.int64 numbers are passed to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--no-spec"]

    assert subprocess.run(argv, env=env).returncode == 0
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0

    assert "cdef long long[3] _v_foo" in (work_dir / (model + "_nomx_cy") / "_mx_classes.pxd").read_text()


@pytest.mark.parametrize("sample_dir, model, spec", [["deep_recursion", "DeepRecursion", ""],
                                               ["index_range", "IndexRange", "--no-spec"]],
                         indirect=["sample_dir"])
def test_deep_recursion_and_index_range(sample_dir, model, spec):
    """int and np.int64 numbers are passed to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py")]

    if spec:
        argv.append(spec)

    assert subprocess.run(argv, env=env, cwd=work_dir).returncode == 0
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env,
        capture_output=True,
        text=True
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["various_types", "VariousTypes"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("sample, assertion", [['sample.py', 'assert_cy.py'],
                                               ['no_sample.py', 'assert_cy_no_sample.py']])
def test_various_types(sample_dir, model, sample, assertion):
    """int and np.int64 numbers are passed to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / sample),
            "--no-spec"]

    assert subprocess.run(argv, env=env, cwd=work_dir).returncode == 0
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env,
        capture_output=True,
        text=True
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["array_size", "ArraySize"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("spec", [["--no-spec"], ["--spec", "spec_large.py"], ["--spec", "spec_small.py"]])
def test_array_size(sample_dir, model, spec):
    """int and np.int64 numbers are passed to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            # "--sample", str(work_dir / "sample.py"),
            "--log-level", "INFO"]
    argv += spec

    assert (result:= subprocess.run(argv, env=env, cwd=work_dir,  capture_output=True, text=True)).returncode == 0

    if spec[-1] == "spec_small.py":
        assert "INFO:modelx_cython.builder:Specified max size of 6 for cells parameter i in _c_Space1 is replaced by 11 from ArraySize_nomx_cy._mx_classes._c_Space1._f_foo" in result.stderr
        assert "INFO:modelx_cython.builder:Specified max size of 3 for cells parameter i in _c_Space1 is replaced by 6 from ArraySize_nomx_cy._mx_classes._c_Space1._f_bar" in result.stderr
        assert "INFO:modelx_cython.builder:Specified max size of 6 for cells parameter j in _c_Space1 is replaced by 11 from ArraySize_nomx_cy._mx_classes._c_Space1._f_bar" in result.stderr

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env,
        capture_output=True,
        text=True
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["size_spec_change", "SizeSpecChange"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("spec", ["spec_old.py", "spec_new.py"])
def test_varying_arg_types(sample_dir, model, spec):
    """int and float numbers are given to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = ["mx2cy", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--spec", str(work_dir / spec)]

    assert (result := subprocess.run(argv, env=env, capture_output=True, text=True)).returncode == 0
    assert subprocess.run([sys.executable, str(work_dir / "assert_cy_old.py")], env=env).returncode == 0
    assert subprocess.run([sys.executable, str(work_dir / "assert_cy_new.py")], env=env
                          ).returncode == (1 if spec == 'spec_old.py' else 0)