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


# The model lives under products/<product>/ in the library, and reads its
# input CSVs from the directory above itself.
ULSG_PRODUCT = "guaranteed_ul"


@pytest.mark.parametrize("sample_dir, model", [["uslib_ulsg", "ULSG_US_S"]],
                         indirect=["sample_dir"])
def test_mx2cy_with_uslib(sample_dir, model):
    """``**`` on a typed double compiles, on the model that reported it.

    ULSG_US_S is the universal-life model of issue #49.  Its
    ``inv_return_mth`` derives a monthly credited rate from an annual one
    and floors it at the guarantee::

        max((1 + crediting_rate_ann(t)) ** (1 / 12) - 1, guar_rate_mth())

    Until :mod:`modelx_cython.powers` rewrote such powers, that failed to
    cythonize with "complex types are unordered" and the model could not
    be translated under any configuration of mx2cy.  Seven of its
    formulas raise a base to a fractional exponent, over all three shapes
    the operand classifier resolves — a cells call, a reference and a
    parameter — while ``inflation_factor`` raises one to an integer
    exponent and must come through untouched.
    """
    import lifelib
    import modelx as mx

    work_dir = sample_dir
    try:
        lifelib.create('uslib', work_dir / 'uslib')
    except KeyError:
        pytest.skip("this lifelib has no uslib library")

    product = work_dir / 'uslib' / 'products' / ULSG_PRODUCT
    if not (product / model).is_dir():
        pytest.skip(f"lifelib has no uslib/products/{ULSG_PRODUCT}/{model}")

    mx.read_model(product / model).export(work_dir / (model + '_nomx'))
    del mx.get_models()[model]

    # The exported model reads its input CSVs from its parent directory
    for csv in product.glob("*.csv"):
        shutil.copy(csv, work_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--no-spec"]

    # this is the assertion issue #49 is about: before the rewrite,
    # cythonize failed with "complex types are unordered"
    assert subprocess.run(argv, env=env).returncode == 0

    translated = work_dir / (model + "_nomx_cy")
    src = (translated / "_mx_classes.py").read_text(encoding="utf-8")

    # every fractional power is rewritten, over all three operand shapes
    assert ("_mx_sys._mx_pow((1 + self.crediting_rate_ann(t)), (1 / 12))"
            in src)                                     # cells call
    assert "_mx_sys._mx_pow((1 + self.guar_rate_ann), (1 / 12))" in src  # ref
    assert "_mx_sys._mx_pow((1 - self.mort_rate(t)), (1 / 12))" in src
    # an integer exponent never reaches the complex path, so it is left
    # exactly as modelx exported it
    assert ("return (1 + self.inflation_rate) ** (self.policy_year(t) - 1)"
            in src)
    # nothing is left for Cython to evaluate on double complex
    c_src = (translated / "_mx_classes.c").read_text(encoding="utf-8")
    assert "__Pyx_c_pow_double" not in c_src
    assert "__Pyx_SoftComplexToDouble" not in c_src

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0
