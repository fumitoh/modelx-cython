import libcst as cst
import pytest

from modelx_cython.powers import (
    KIND_FLOAT, KIND_INT, KIND_UNKNOWN, OperandKind, build_kind_maps,
    kind_of_type_expr, rewrite_powers,
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
    ("t ** +2", KIND_INT),
    ("t ** -0", KIND_INT),
    ("2 ** -1", KIND_UNKNOWN),
    ("2 ** t", KIND_UNKNOWN),
    ("2 ** (2 ** -1)", KIND_UNKNOWN),
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

    def __init__(self, return_type="object"):
        self.return_type = return_type

    def has_args(self):
        return True

    def has_typeinfo(self):
        return True

    def get_rettype_expr(self, c_style=False):
        assert c_style
        return self.return_type

    def get_argtype_expr(self, arg, c_style=False, formula=False):
        # the C-style spelling is what kind_of_type_expr matches, so a
        # caller that forgot c_style=True must not silently pass
        assert c_style, "powers must ask for the C-style type expression"
        # the formula's types, which can differ from the public method's
        assert formula, "powers must ask for the formula's types"
        return "long long" if arg == "t" else "double"


def _rewrite(body: str, *, builtin_float=False, return_type="object") -> str:
    resolver = CellsResolver(
        cells_by_class={CLS: {"rate": RATE, "mort_rate": RATE, "lapse_rate": RATE,
                              "transfer_rate": RATE, "arr": ARR}},
        refs_by_class={CLS: {}},
        spaces_by_class={CLS: {}},
    )
    func = cst.parse_statement(f"def _f_target(self, t, x):\n    {body}\n")
    out = rewrite_powers(
        func, resolver,
        cells_kinds={RATE: KIND_FLOAT, ARR: KIND_UNKNOWN},
        ref_kinds={CLS: {"discount_rate": KIND_FLOAT}},
        cls_fqname=CLS,
        cells=_FakeCells(return_type),
        builtin_float=builtin_float,
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


def test_annotated_float_locals_are_native_only_for_the_builtin():
    body = "base: float = x\n    years: float = t / 12.0\n    return base ** years"
    assert "_mx_sys._mx_pow(base, years)" in _rewrite(body, builtin_float=True)
    assert "base ** years" in _rewrite(body)


def test_float_local_rebinding_does_not_prove_the_power():
    body = "base: float = x\n    base = self.arr(t)\n    return base ** 0.5"
    assert "base ** 0.5" in _rewrite(body, builtin_float=True)


@pytest.mark.parametrize("local, cell, zero_guard", [
    ("q", "mort_rate", ""),
    ("w", "lapse_rate", "    if w <= 0.0:\n        return 0.0\n"),
    ("th", "transfer_rate", "    if th <= 0.0:\n        return 0.0\n"),
])
def test_double_cell_local_feeds_native_return_power(local, cell, zero_guard):
    body = (f"{local} = self.{cell}(t)\n"
            f"{zero_guard}"
            f"    if {local} >= 1.0:\n"
            "        return 1.0 if self.is_anniv(t) else 0.0\n"
            f"    return 1.0 - (1.0 - {local}) ** (1.0 / 12.0)")
    out = _rewrite(body, return_type="double")
    assert f"_mx_sys._mx_pow((1.0 - {local}), (1.0 / 12.0))" in out
    assert out.count(f"self.{cell}(t)") == 1
    assert f"if {local} >= 1.0:" in out
    assert "**" not in out
    assert "**" in _rewrite(body)


def test_double_reference_local_feeds_native_return_power():
    body = ("j = self.discount_rate\n"
            "    if j == 0.0:\n        return 0.0\n"
            "    return (1.0 - (1.0 + j) ** (-t / 12.0)) / ((1.0 + j) ** (1.0 / 12.0) - 1.0)")
    out = _rewrite(body, return_type="double")
    assert out.count("_mx_sys._mx_pow(") == 2
    assert "**" not in out
    assert "**" in _rewrite(body)


@pytest.mark.parametrize("body", [
    "base = self.rate(t)\n    root = base ** 0.5\n    return root",
    "base = self.rate(t)\n    return max(0.0, base ** 0.5)",
    "base = self.rate(t)\n    return self.arr(t) + base ** 0.5",
    "base = self.rate(t)\n    return base ** -t",
    "base = self.rate(t)\n    base = self.arr(t)\n    return base ** 0.5",
    "base = self.arr(t)\n    return base ** 0.5",
    "base = x\n    return base ** 0.5",
    "if t:\n        return base ** 0.5\n    base = self.rate(t)\n    return 0.0",
    "if t:\n        base = self.rate(t)\n    return base ** 0.5",
    "self = other\n    base = self.rate(t)\n    return base ** 0.5",
    "base = self.rate(t)\n    return sum(base ** 0.5 for base in (x,))",
])
def test_cell_local_power_keeps_unproved_uses(body):
    out = _rewrite(body, return_type="double")
    assert "_mx_sys._mx_pow(" not in out
    assert "**" in out


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


def test_assigned_local_kinds_from_every_assignment():
    """D59: a local takes a kind only when every binding of it has that kind."""
    import libcst as cst
    from modelx_cython.powers import _assigned_local_kinds, KIND_FLOAT, KIND_INT

    class _NoResolve:
        def resolve_chain(self, *args):
            return None, None

    source = (
        'def f(self, t, i):\n'
        '    v = 1.0 / (1.0 + i)\n'
        '    total = 0.0\n'
        '    k = 1\n'
        '    mixed = 0\n'
        '    step = 3 if t else 6\n'
        '    either = 1 if t else 0.5\n'
        '    while True:\n'
        '        s = t + (k - 1) * step\n'
        '        if s > 40:\n'
        '            break\n'
        '        total += v ** ((s - t) / 12.0)\n'
        '        mixed += 0.5\n'
        '        k += 1\n'
        '    for j in range(3):\n'
        '        looped = j\n'
        '    return total\n')
    body = cst.parse_module(source).body[0].body.body
    kinds = _assigned_local_kinds(body, ('t', 'i'), _NoResolve(), {}, {}, 'm.C',
                                  {'t': KIND_INT, 'i': KIND_FLOAT})
    assert kinds == {'v': KIND_FLOAT, 'total': KIND_FLOAT, 'k': KIND_INT, 'step': KIND_INT,
                     's': KIND_INT, 'j': KIND_INT, 'looped': KIND_INT}, kinds


@pytest.mark.parametrize("body, count", [
    # an accumulator started at a double: the power is an operand of its arithmetic
    ("v = 1.0 / (1.0 + self.rate(t))\n    total = 0.0\n    k = 1\n"
     "    while k < 5:\n        total += (1.0 / 4) * v ** ((k - t) / 12.0)\n        k += 1\n    return total", 1),
    # inside a typed range loop, with the bound from a parameter
    ("v = self.rate(t)\n    total = 0.0\n    for j in range(t):\n        total = total + v ** (j / 12.0)\n"
     "    return total", 1),
])
def test_power_into_a_double_accumulator_is_native(body, count):
    """D62: the power's value flows into a local that is a double without it."""
    out = _rewrite(body)
    assert out.count("_mx_sys._mx_pow(") == count, out
    assert "**" not in out


@pytest.mark.parametrize("body", [
    # the only double assignment is the power itself: Python's semantics stay
    "v = self.rate(t)\n    root = v ** 0.5\n    return root",
    # the accumulator also takes a value of unknown kind
    "v = self.rate(t)\n    total = 0.0\n    total += v ** 0.5\n    total += self.arr(t)\n    return total",
    # the base is unknown
    "total = 0.0\n    total += self.arr(t) ** 0.5\n    return total",
    # the accumulator is an integer
    "v = self.rate(t)\n    total = 0\n    total += v ** 0.5\n    return total",
])
def test_power_outside_a_double_sink_keeps_python_semantics(body):
    out = _rewrite(body)
    assert "_mx_sys._mx_pow(" not in out and "**" in out, out


def test_power_accumulated_with_max_of_doubles_is_native():
    """DIA_US_S annuity_factor: ``total += c * v ** e * max(cert, lf)``."""
    body = ("v = 1.0 / (1.0 + self.rate(t))\n    cert = self.rate(t) if t else 0.0\n    lf = self.rate(t)\n"
            "    total = 0.0\n    k = 1\n    while k < 5:\n"
            "        total += (1.0 / 4) * v ** ((k - t) / 12.0) * max(cert, lf)\n        k += 1\n    return total")
    out = _rewrite(body)
    assert out.count("_mx_sys._mx_pow(") == 1 and "**" not in out, out
    assert "**" in _rewrite(body.replace("max(cert, lf)", "max(cert, self.arr(t))"))
    assert "**" in _rewrite("max = self.rate\n    " + body)


def test_range_loop_bound_from_int_of_a_number():
    """DIA_US_S annuity_factor (range spelling): ``n = int(...)``; ``for k in range(1, n + 1)``."""
    body = ("v = self.rate(t)\n    n = int((40 - t) // 3)\n    total = 0.0\n"
            "    for k in range(1, n + 1):\n        total += v ** ((3 * k) / 12.0) * max(v, 0.5)\n    return total")
    out = _rewrite(body)
    assert out.count("_mx_sys._mx_pow(") == 1 and "**" not in out, out
    assert "**" in _rewrite("int = self.arr\n    " + body)
def test_local_kinds_through_declarations_and_compiler_casts():
    """diag-us (SPIA_US_S commuted_value): a loop variable declared before its loop
    (``s: _mx_cy.longlong``) and a value guarded by the overflow cast keep their kinds,
    so the power in one branch of the accumulator's operand is native."""
    body = ("n = 12\n    j = self.rate(t)\n    total = 0.0\n    s: _mx_cy.longlong\n"
            "    for s in range(t + 1, n):\n"
            "        yrs = ((s - t) / 12.0 if t >= 0 else _mx_cy.cast(_mx_cy.double, (_mx_cy.cast(object, s) - t) / 12.0))\n"
            "        if t > 3:\n            v = (1.0 + j) ** (-yrs)\n        else:\n            v = max(0.0, 1.0 - j * yrs)\n"
            "        total += self.rate(s) * v\n    return total")
    out = _rewrite(body)
    assert out.count("_mx_sys._mx_pow(") == 1 and "**" not in out, out
    # without the declaration rule or the cast rule the chain breaks: the power stays Python's
    assert "**" in _rewrite(body.replace("_mx_cy.cast(_mx_cy.double, ", "self.arr(").replace(
        "/ 12.0))\n", "/ 12.0))\n", 1))


def test_formula_local_kinds_reads_the_formula_body():
    import libcst as cst
    from modelx_cython.powers import formula_local_kinds, KIND_INT, KIND_FLOAT

    class _NoResolve:
        def resolve_chain(self, *args):
            return None, None

    node = cst.parse_module('def f(self, t):\n    k = t - 3\n    s: _mx_cy.longlong\n'
                            '    for s in range(k):\n        pass\n    q = 0.5 ** k\n    return q\n').body[0]
    kinds = formula_local_kinds(node, ('t',), _NoResolve(), {}, {}, 'm.C', {'t': KIND_INT})
    assert kinds == {'k': KIND_INT, 's': KIND_INT, 'q': KIND_FLOAT}, kinds
    assert formula_local_kinds(node, ('t',), None, {}, {}, 'm.C', {'t': KIND_INT}) == {}
