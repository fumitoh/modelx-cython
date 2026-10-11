import libcst as cst
import pytest

from modelx_cython.powers import (
    KIND_FLOAT, KIND_INT, KIND_UNKNOWN, OperandKind, build_kind_maps,
    build_math_refs, kind_of_type_expr, rewrite_math_calls, rewrite_powers,
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
    # true division yields a float only for operands that are numbers at
    # all: an unknown operand must win over the Divide shortcut, or an
    # array divided by an int would count as a floating exponent
    ("local / 12", KIND_UNKNOWN),
    ("self.arr(t) / 12", KIND_UNKNOWN),
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

    fqname = formula_fqname = CLS + "._f_target"
    params = ("t", "x")

    def has_args(self):
        return True

    def has_typeinfo(self):
        return True

    def get_argtype_expr(self, arg, c_style=False, formula=False):
        # the C-style spelling is what kind_of_type_expr matches, so a
        # caller that forgot c_style=True must not silently pass
        assert c_style, "powers must ask for the C-style type expression"
        # the formula's types, which can differ from the public method's
        assert formula, "powers must ask for the formula's types"
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
    # a construct that binds names of its own can shadow a parameter, and
    # the parameter's declared C type would not apply to the shadowing
    # name, so nothing inside one is rewritten
    "return sum(x ** (1 / 12) for x in range(3))",
    "return [x ** (1 / 12) for x in range(3)]",
    "return {x ** (1 / 12) for x in range(3)}",
    "return {i: x ** (1 / 12) for i, x in enumerate([2.0])}",
    "return (lambda x: x ** (1 / 12))(2.0)",
    "def inner(x):\n        return x ** (1 / 12)\n    return inner(local)",
    "def inner(t):\n        return 2 ** (t / 12)\n    return inner(local)",
    "class C:\n        x = 3.0\n        v = x ** (1 / 12)\n    return C.v",
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


def test_an_integer_base_is_cast_explicitly():
    """``_mx_pow`` takes doubles; narrowing the base implicitly costs an
    MSVC C4244 warning on a build that was warning-free before."""
    assert "_mx_sys._mx_pow(_mx_cy.cast(_mx_cy.double, t), (1 / 12))" in (
        _rewrite("return t ** (1 / 12)"))
    # a base that is already floating needs no cast
    out = _rewrite("return x ** (1 / 12)")
    assert "_mx_sys._mx_pow(x, (1 / 12))" in out
    assert "cast" not in out


class _FakeCellsInfo:
    def __init__(self, fqname, rettype):
        self.fqname = fqname
        self._rettype = rettype

    def get_rettype_expr(self, c_style=False):
        assert c_style
        return self._rettype


class _FakeRefInfo:
    def __init__(self, type_expr):
        self._type_expr = type_expr

    def get_type_expr(self, c_style=False):
        assert c_style
        return self._type_expr


class _FakeClassInfo:
    def __init__(self, fqname, cells, refs):
        self.fqname = fqname
        self.cells = cells
        self.refs = refs


class _FakeModuleInfo:
    def __init__(self, classes):
        self.classes = classes


def test_build_kind_maps():
    """The maps must key cells by fqname and refs by class fqname, and
    must ask for the C-style spelling of every type."""
    info = _FakeModuleInfo({
        "_c_Space1": _FakeClassInfo(
            CLS,
            cells={"rate": _FakeCellsInfo(RATE, "double"),
                   "count": _FakeCellsInfo(COUNT, "long long"),
                   "arr": _FakeCellsInfo(ARR, "const double[:]")},
            refs={"rate_ref": _FakeRefInfo("double"),
                  "n_ref": _FakeRefInfo("long long"),
                  "table": _FakeRefInfo("object")},
        )
    })
    cells_kinds, ref_kinds = build_kind_maps({"Model_nomx._mx_classes": info})

    assert cells_kinds == {RATE: KIND_FLOAT, COUNT: KIND_INT,
                           ARR: KIND_UNKNOWN}
    assert ref_kinds == {CLS: {"rate_ref": KIND_FLOAT, "n_ref": KIND_INT,
                               "table": KIND_UNKNOWN}}


def test_kind_maps_feed_the_classifier():
    """build_kind_maps' output is what OperandKind consumes, so a ref
    typed through it must classify the same way."""
    info = _FakeModuleInfo({
        "_c_Space1": _FakeClassInfo(
            CLS,
            cells={"rate": _FakeCellsInfo(RATE, "double")},
            refs={"rate_ref": _FakeRefInfo("double")},
        )
    })
    cells_kinds, ref_kinds = build_kind_maps({"Model_nomx._mx_classes": info})
    kind = OperandKind(
        CellsResolver({CLS: {"rate": RATE}}, {CLS: {"rate_ref": ""}},
                      {CLS: {}}),
        cells_kinds, ref_kinds, CLS, {},
    )
    assert kind.kind(cst.parse_expression("self.rate(0)")) == KIND_FLOAT
    assert kind.kind(cst.parse_expression("self.rate_ref")) == KIND_FLOAT


# --------------------------------------------------------------------
# math.exp/log/pow under the spec's "use_libm"


def _rewrite_math(body: str, math_refs=None) -> str:
    resolver = CellsResolver(
        cells_by_class={CLS: {"rate": RATE, "arr": ARR}},
        refs_by_class={CLS: {"math": "", "np": ""}},
        spaces_by_class={CLS: {}},
    )
    func = cst.parse_statement(f"def _f_target(self, t, x):\n    {body}\n")
    out = rewrite_math_calls(
        func, resolver,
        cells_kinds={RATE: KIND_FLOAT, ARR: KIND_UNKNOWN},
        ref_kinds={CLS: {}},
        math_refs={CLS: {"math"}} if math_refs is None else math_refs,
        cls_fqname=CLS,
        cells=_FakeCells(),
    )
    return cst.Module([]).code_for_node(out)


@pytest.mark.parametrize("body, expected", [
    ("return self.math.exp(x)", "return _mx_sys._mx_exp(x)"),
    ("return self.math.exp(-self.rate(t) * t)",
     "return _mx_sys._mx_exp(-self.rate(t) * t)"),
    ("return self.math.log(1 + x)", "return _mx_sys._mx_log(1 + x)"),
    ("return self.math.pow(x, 0.5)", "return _mx_sys._mx_math_pow(x, 0.5)"),
    # an integer argument is cast explicitly, as math converts it
    ("return self.math.exp(t)",
     "return _mx_sys._mx_exp(_mx_cy.cast(_mx_cy.double, t))"),
    ("return self.math.pow(2, t)",
     "return _mx_sys._mx_math_pow(_mx_cy.cast(_mx_cy.double, 2), "
     "_mx_cy.cast(_mx_cy.double, t))"),
    # nested calls are both rewritten
    ("return self.math.exp(self.math.log(x))",
     "return _mx_sys._mx_exp(_mx_sys._mx_log(x))"),
])
def test_math_calls_are_rewritten(body, expected):
    assert expected in _rewrite_math(body)


@pytest.mark.parametrize("body", [
    "return self.math.log(x, 2.0)",         # two-argument log
    "return self.math.exp(x=x)",            # keyword
    "return self.math.pow(*[x, 2.0])",      # starred
    "return self.math.exp(local)",          # unknown argument
    "return self.math.exp(self.arr(t)[0])",
    "return self.math.sqrt(x)",             # not lowered
    "return self.np.exp(x)",                # not the math module
    "return math.exp(x)",                   # not through a reference
    "return self.other.math.exp(x)",
    "return [self.math.exp(x) for x in range(3)]",  # x may be rebound
])
def test_math_calls_left_alone(body):
    assert "_mx_sys" not in _rewrite_math(body)


def test_math_calls_need_use_libm():
    """Without math_refs (no "use_libm" in the spec) nothing changes"""
    assert "_mx_sys" not in _rewrite_math("return self.math.exp(x)", {})
    assert "_mx_sys" not in _rewrite_math(
        "return self.math.exp(x)", {CLS + "_other": {"math"}})


def test_build_math_refs():
    import math
    import types
    from modelx_cython.builder import CombinedRefInfo
    from modelx_cython.tracer import RuntimeRefInfo

    refs = {name: CombinedRefInfo("M._mx_classes", "_c_S", name,
                                  RuntimeRefInfo.init_mxobj(value, "M"))
            for name, value in [("math", math), ("m", math), ("np", types),
                                ("x", 1.0)]}
    cls_info = types.SimpleNamespace(fqname="M._mx_classes._c_S", refs=refs)
    info = types.SimpleNamespace(classes={"_c_S": cls_info})
    assert build_math_refs({"M._mx_classes": info}) == {
        "M._mx_classes._c_S": {"math", "m"}}


# --------------------------------------------------------------------
# '**' and math calls under use_libm, applied to each other


@pytest.mark.parametrize("body, expected", [
    # the power of a math call that becomes a C double is rewritten too:
    # left alone, it would go on Cython's complex path, which max()
    # cannot compare
    ("return max(self.math.exp(x) ** 0.5, 1.0)",
     "return max(_mx_sys._mx_pow(_mx_sys._mx_exp(x), 0.5), 1.0)"),
    ("return 2 ** self.math.log(x)",
     "return _mx_sys._mx_pow(_mx_cy.cast(_mx_cy.double, 2), "
     "_mx_sys._mx_log(x))"),
    # a math call on a rewritten power is rewritten
    ("return self.math.exp(x ** 2.0)",
     "return _mx_sys._mx_exp(_mx_sys._mx_pow(x, 2.0))"),
    ("return self.math.exp(x ** 2.0) ** 0.5",
     "return _mx_sys._mx_pow(_mx_sys._mx_exp(_mx_sys._mx_pow(x, 2.0)), 0.5)"),
])
def test_powers_and_math_calls_together(body, expected):
    resolver = CellsResolver(
        cells_by_class={CLS: {"rate": RATE}},
        refs_by_class={CLS: {"math": ""}},
        spaces_by_class={CLS: {}},
    )
    func = cst.parse_statement(f"def _f_target(self, t, x):\n    {body}\n")
    args = dict(cells_kinds={RATE: KIND_FLOAT}, ref_kinds={CLS: {}},
                cls_fqname=CLS, cells=_FakeCells())
    math_refs = {CLS: {"math"}}
    func = rewrite_powers(func, resolver, math_refs=math_refs, **args)
    func = rewrite_math_calls(func, resolver, math_refs=math_refs, **args)
    assert expected in cst.Module([]).code_for_node(func)


def test_powers_without_use_libm_leave_math_calls_unknown():
    """Without use_libm the math call stays a Python object, and a power
    on it is left as exported"""
    out = _rewrite("return max(self.math.exp(x) ** 0.5, 1.0)")
    assert "self.math.exp(x) ** 0.5" in out


# --------------------------------------------------------------------
# Parameters a formula binds again

import types

from modelx_cython.builder import CombinedCellsInfo
from modelx_cython.parser import LexicalCellsInfo, collect_param_rebinds
from modelx_cython.powers import demote_rebound_params
from modelx_cython.tracer import ReturnTypeInfo


def make_rebound(body, arg_types, has_formula_def=True):
    """Module infos with one cells ``target(t, x)`` whose formula is
    ``body``, traced with ``arg_types``"""
    params = list(arg_types)
    src = f"def _f_target(self, {', '.join(params)}):\n" + "".join(
        "    " + line + "\n" for line in body.splitlines())
    lx = LexicalCellsInfo("Model_nomx._mx_classes", "_c_Space1", "target",
                          params)
    rt = types.SimpleNamespace(ret_type=ReturnTypeInfo(float),
                               arg_types=arg_types,
                               max_args={p: 3 for p, t in arg_types.items()
                                         if t is int})
    cells = CombinedCellsInfo(None, lx, rt, {})
    cells.has_formula_def = has_formula_def
    cells.param_rebinds = collect_param_rebinds(cst.parse_statement(src),
                                                params)
    cls_info = types.SimpleNamespace(fqname=CLS, cells={"target": cells})
    infos = {"Model_nomx._mx_classes": types.SimpleNamespace(
        classes={"_c_Space1": cls_info})}
    resolver = CellsResolver(cells_by_class={CLS: {"rate": RATE}},
                             refs_by_class={CLS: {}},
                             spaces_by_class={CLS: {}})
    demote_rebound_params(infos, resolver, {RATE: KIND_FLOAT}, {CLS: {}})
    return cells


@pytest.mark.parametrize("body, arg_types, kept", [
    # bound to values provably of the parameter's C type: kept
    ("t = t - 1\nreturn t", {"t": int}, True),
    ("t += 1\nt //= 2\nt = -t % 3\nreturn t", {"t": int}, True),
    ("x = x / 100\nx = t\nx = self.rate() * 2\nreturn x",
     {"t": int, "x": float}, True),
    ("x **= 0.5\nreturn x", {"x": float}, True),
    # not provably: object
    ("t = t / 100\nreturn t", {"t": int}, False),
    ("t /= 2\nreturn t", {"t": int}, False),         # compiles, truncates
    ("t = 2 ** -t\nreturn t", {"t": int}, False),    # a double in Cython
    ("t **= 2\nreturn t", {"t": int}, False),
    ("t = max(t, 0.5)\nreturn t", {"t": int}, False),
    ("t = self.rate()\nreturn t", {"t": int}, False),
    ("x = None\nreturn x", {"x": float}, False),
    ("for x in range(3):\n    pass\nreturn x", {"x": float}, False),
    ("b = not b\nreturn b", {"b": bool}, False),
    ("s = s + 'a'\nreturn s", {"s": str}, False),
])
def test_demote_rebound_params(body, arg_types, kept):
    cells = make_rebound(body, arg_types)
    param = list(cells.param_rebinds)[-1]
    traced = cells.get_argtype_expr(param, c_style=True)
    assert traced != "object"
    expected = traced if kept else "object"
    assert cells.get_argtype_expr(param, c_style=True, formula=True) \
        == expected


def test_demote_rebound_params_until_nothing_changes():
    """x is demoted for t, which is demoted only once x is: the check is
    repeated"""
    cells = make_rebound("x = None\nt = x\nreturn t", {"t": float, "x": float})
    assert cells.get_argtype_expr("x", c_style=True, formula=True) == "object"
    assert cells.get_argtype_expr("t", c_style=True, formula=True) == "object"
    # the public method of a cached cells keeps the types
    assert cells.get_argtype_expr("t", c_style=True) == "double"


def test_demote_rebound_params_uncached_cells_everywhere():
    cells = make_rebound("t = t / 2\nreturn t", {"t": int},
                         has_formula_def=False)
    assert cells.get_argtype_expr("t", c_style=True) == "object"
