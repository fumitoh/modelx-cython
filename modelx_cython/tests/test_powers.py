import libcst as cst
import pytest

from modelx_cython.powers import (
    KIND_FLOAT, KIND_INT, KIND_UNKNOWN, OperandKind, kind_of_type_expr,
    rewrite_powers,
)
from modelx_cython.usage import CellsResolver

CLS = "Model_nomx._mx_classes._c_Space1"
RATE = CLS + "._f_rate"
COUNT = CLS + "._f_count"
ARR = CLS + "._f_arr"


@pytest.fixture
def kind():
    """A classifier over one class with a float cells, an int cells, an
    array cells, a float ref and an int ref, and parameters ``t`` (int)
    and ``x`` (float)."""
    resolver = CellsResolver(
        cells_by_class={CLS: {"rate": RATE, "count": COUNT, "arr": ARR}},
        refs_by_class={CLS: {"rate_ref": "", "n_ref": "", "table": ""}},
        spaces_by_class={CLS: {}},
    )
    return OperandKind(
        resolver,
        cells_kinds={RATE: KIND_FLOAT, COUNT: KIND_INT, ARR: KIND_UNKNOWN},
        ref_kinds={CLS: {"rate_ref": KIND_FLOAT, "n_ref": KIND_INT,
                         "table": KIND_UNKNOWN}},
        cls_fqname=CLS,
        param_kinds={"t": KIND_INT, "x": KIND_FLOAT},
    )


@pytest.mark.parametrize("expr, expected", [
    # literals
    ("0.5", KIND_FLOAT),
    ("2", KIND_INT),
    # true division is what makes the ``x ** (1 / 12)`` idiom work: both
    # operands are integer literals but the value is a float
    ("1 / 12", KIND_FLOAT),
    ("t / 12", KIND_FLOAT),
    # promotion
    ("1 + x", KIND_FLOAT),
    ("1 + t", KIND_INT),
    ("2 * t + 1", KIND_INT),
    ("-t", KIND_INT),
    ("-x", KIND_FLOAT),
    ("(1 - x)", KIND_FLOAT),
    ("t ** 2", KIND_INT),
    ("x ** 2", KIND_FLOAT),
    # resolved through the model
    ("self.rate(t)", KIND_FLOAT),
    ("self.count(t)", KIND_INT),
    ("1 + self.rate(t)", KIND_FLOAT),
    ("self.rate_ref", KIND_FLOAT),
    ("self.n_ref", KIND_INT),
    # nothing is assumed about anything else
    ("self.arr(t)", KIND_UNKNOWN),
    ("self.table", KIND_UNKNOWN),
    ("self.arr(t)[0]", KIND_UNKNOWN),
    ("self.other.rate(t)", KIND_UNKNOWN),
    ("local", KIND_UNKNOWN),
    ("len(y)", KIND_UNKNOWN),
    ("self.np.exp(x)", KIND_UNKNOWN),
    ("x + local", KIND_UNKNOWN),
    ("t & 1", KIND_UNKNOWN),
    ('"s"', KIND_UNKNOWN),
])
def test_operand_kind(kind, expr, expected):
    assert kind.kind(cst.parse_expression(expr)) == expected


@pytest.mark.parametrize("type_expr, expected", [
    ("double", KIND_FLOAT),
    ("long long", KIND_INT),
    # a memoryview must not match on its element type, and bint is left
    # unknown rather than treated as an integer
    ("const double[:, :]", KIND_UNKNOWN),
    ("bint", KIND_UNKNOWN),
    ("object", KIND_UNKNOWN),
    ("str", KIND_UNKNOWN),
])
def test_kind_of_type_expr(type_expr, expected):
    assert kind_of_type_expr(type_expr) == expected


class _FakeCells:
    """Stands in for CombinedCellsInfo: only the parameter types and the
    fqname are read by :func:`rewrite_powers`."""

    fqname = CLS + "._f_target"
    params = ("t", "x")

    def has_args(self):
        return True

    def has_typeinfo(self):
        return True

    def get_argtype_expr(self, arg, c_style=False):
        return "long long" if arg == "t" else "double"


def _rewrite(body: str) -> str:
    resolver = CellsResolver(
        cells_by_class={CLS: {"rate": RATE, "arr": ARR}},
        refs_by_class={CLS: {}},
        spaces_by_class={CLS: {}},
    )
    func = cst.parse_statement(f"def _f_target(self, t, x):\n    {body}\n")
    out = rewrite_powers(
        func, resolver,
        cells_kinds={RATE: KIND_FLOAT, ARR: KIND_UNKNOWN},
        ref_kinds={CLS: {}},
        cls_fqname=CLS,
        cells=_FakeCells(),
    )
    return cst.Module([]).code_for_node(out)


@pytest.mark.parametrize("body", [
    # the shape from issue #49: without the rewrite Cython rejects this
    # with "complex types are unordered"
    "return max((1 + self.rate(t)) ** (1 / 12) - 1, 0.003)",
    "return 1 - (1 - self.rate(t)) ** (1 / 12)",
    "return x ** 0.5",
    "return x ** (t / 12)",
    "return 2 ** (1 / 12)",
])
def test_floating_exponent_is_rewritten(body):
    assert "_mx_sys._mx_pow(" in _rewrite(body)
    assert "**" not in _rewrite(body)


@pytest.mark.parametrize("body", [
    # an integer exponent never reaches Cython's complex path, so it is
    # left alone -- this is what keeps 2 ** -3 equal to 0.125
    "return 2 ** -3",
    "return 2 ** -t",
    "return t ** 2",
    "return x ** 2",
    "return x ** t",
    # an operand with no sampled C type is never rewritten through
    "return self.arr(t) ** 0.5",
    "return self.arr(t)[0] ** 0.5",
    "return local ** 0.5",
    "return x ** local",
    # a comprehension or lambda can shadow a parameter, so nothing
    # inside one is rewritten
    "return sum(x ** (1 / 12) for x in range(3))",
    "return [x ** (1 / 12) for x in range(3)]",
    "return (lambda x: x ** (1 / 12))(2.0)",
])
def test_left_alone(body):
    out = _rewrite(body)
    assert "_mx_pow" not in out
    assert "**" in out


def test_nested_powers_are_both_rewritten():
    out = _rewrite("return (x ** 0.5) ** (1 / 12)")
    assert out.count("_mx_sys._mx_pow(") == 2
    assert "**" not in out


def test_without_a_resolver_nothing_is_rewritten():
    func = cst.parse_statement(
        "def _f_target(self, t, x):\n    return x ** 0.5\n")
    out = rewrite_powers(func, None, {}, {}, CLS, _FakeCells())
    assert out is func
