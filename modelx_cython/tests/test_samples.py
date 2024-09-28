import sys
import os
import subprocess
import pathlib
import shutil
import pytest


@pytest.fixture
def copy_samples(tmp_path_factory):
    dst = tmp_path_factory.mktemp("temp") / "samples"
    shutil.copytree(pathlib.Path(__file__).parent / "samples", dst)
    return dst

@pytest.mark.parametrize("target", ["mx2cy", "main"])
def test_mx2cy_with_basicterm_s(copy_samples, target):
    import lifelib
    import modelx as mx

    work_dir = copy_samples / "basicterm_s"
    lifelib.create('basiclife', work_dir / 'basiclife')
    mx.read_model(work_dir / 'basiclife' / 'BasicTerm_S').export(work_dir / 'BasicTerm_S_nomx')

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

    argv = ["mx2cy", str(work_dir / "BasicTerm_S_nomx"),
         "--spec", str(work_dir / "spec.py"),
         "--sample", str(work_dir / "sample.py")]

    if target == "mx2cy":
        assert subprocess.run(argv, env=env).returncode == 0
    elif target == "main":
        from modelx_cython.cli import main
        assert main(argv[1:], sys.stdout, sys.stderr) == 0
    
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_basicterm_s.py")],
        env=env
    ).returncode == 0
