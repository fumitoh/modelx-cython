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

    # Both ordinary and exact-operand memo entries preserve this power
    # contract. Match the operands, not the physical cache entry's spelling.
    import ast
    powers = [node for node in ast.walk(ast.parse(src)) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.value.id == '_mx_sys'
              and node.func.attr in ('_mx_pow', '_mx_memo_pow')]
    class PublicOperands(ast.NodeTransformer):
        def exact_guard(self,node):
            if isinstance(node,ast.Name):
                return node.id.startswith(('_mx_exact_receiver','_mx_inbounds_exact'))
            if isinstance(node,ast.BoolOp):
                return all(self.exact_guard(value) for value in node.values)
            return (isinstance(node,ast.Compare) and len(node.ops)==1
                and isinstance(node.ops[0],ast.Is) and isinstance(node.left,ast.Call)
                and isinstance(node.left.func,ast.Name) and node.left.func.id=='type'
                and len(node.left.args)==1 and isinstance(node.left.args[0],ast.Name)
                and node.left.args[0].id=='self')
        def visit_IfExp(self,node):
            # Read the preserved public operand behind a generated exact-type
            # dispatch; authored conditional arithmetic is not normalized away.
            return self.visit(node.orelse) if self.exact_guard(node.test) else self.generic_visit(node)
        def visit_Call(self,node):
            # The virtual branch of that dispatch converts the public call to the Cell's C
            # return type (``_mx_cy.cast(_mx_cy.double, ...)``, D59); read the call itself.
            if (isinstance(node.func,ast.Attribute) and node.func.attr=='cast'
                    and isinstance(node.func.value,ast.Name) and node.func.value.id=='_mx_cy'
                    and len(node.args)==2 and isinstance(node.args[0],ast.Attribute)
                    and isinstance(node.args[0].value,ast.Name) and node.args[0].value.id=='_mx_cy'):
                return self.visit(node.args[1])
            return self.generic_visit(node)
    def has_power(base):
        expected = [ast.dump(ast.parse(expr, mode='eval').body) for expr in (base, '1 / 12')]
        return any([ast.dump(PublicOperands().visit(arg)) for arg in call.args[-2:]] == expected
                   for call in powers)
    assert has_power('1 + self.crediting_rate_ann(t)')
    assert has_power('1 + self.guar_rate_ann')
    assert has_power('1 - self.mort_rate(t)')
    # Integer-exponent power retains c8's ordinary Cython numeric operation.
    assert "return (1 + self.inflation_rate) ** (self.policy_year(t) - 1)" in src
    # nothing is left for Cython to evaluate on double complex
    c_src = (translated / "_mx_classes.c").read_text(encoding="utf-8")
    assert "__Pyx_c_pow_double" not in c_src
    assert "__Pyx_SoftComplexToDouble" not in c_src

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0
