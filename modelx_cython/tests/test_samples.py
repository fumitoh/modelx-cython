import sys
import os
import subprocess
import pathlib
import shutil
import pytest


@pytest.fixture(scope="session")
def copy_samples(tmp_path_factory):
    dst = tmp_path_factory.mktemp("temp") / "samples"
    shutil.copytree(pathlib.Path(__file__).parent / "samples", dst)
    return dst


def test_basicterm_s(copy_samples):

    work_dir = copy_samples / "basicterm_s"    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")
    assert subprocess.run(
        ["mx2cy", str(work_dir / "BasicTerm_S_nomx"), 
         "--spec", str(work_dir / "spec.py"),
         "--sample", str(work_dir / "sample.py")], env=env).returncode == 0
    
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_basicterm_s.py")],
        env=env
    ).returncode == 0
