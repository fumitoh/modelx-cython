"""Regression tests added by review of the bondportfolio/mx2cy-improvements
changes.  Each test demonstrates a defect and fails on the reviewed code."""


def _rewrite_powers_then_math(body):
    """Both passes, in the order ModuleTransformer._rewrite_powers runs
    them, with use_libm on: ``x`` is a C double parameter."""
    import libcst as cst
    from modelx_cython.powers import rewrite_math_calls, rewrite_powers
    from modelx_cython.usage import CellsResolver

    cls = "Model_nomx._mx_classes._c_Space1"

    class Cells:
        fqname = formula_fqname = cls + "._f_target"
        params = ("x",)

        def has_args(self):
            return True

        def has_typeinfo(self):
            return True

        def get_argtype_expr(self, arg, c_style=False, formula=False):
            return "double"

    resolver = CellsResolver(cells_by_class={cls: {}},
                             refs_by_class={cls: {"math": ""}},
                             spaces_by_class={cls: {}})
    node = cst.parse_statement(f"def _f_target(self, x):\n    {body}\n")
    # as ModuleTransformer._rewrite_powers calls it under use_libm
    node = rewrite_powers(node, resolver, {}, {cls: {}}, cls, Cells(),
                          math_refs={cls: {"math"}})
    node = rewrite_math_calls(node, resolver, {}, {cls: {}},
                              {cls: {"math"}}, cls, Cells())
    return cst.Module([]).code_for_node(node)


def test_use_libm_power_of_math_call_is_real():
    """With use_libm, ``self.math.exp(x)`` becomes a C double, so a ``**``
    on it that is left alone goes on Cython's complex path: the model
    ``max(math.exp(x) ** 0.5, 1.0)`` compiles without use_libm and fails
    with "complex types are unordered" with it.  The power has to be
    rewritten as well (or the math call left alone)."""
    out = _rewrite_powers_then_math("return max(self.math.exp(x) ** 0.5, 1.0)")
    assert ("_mx_sys._mx_pow(_mx_sys._mx_exp(x), 0.5)" in out
            or "self.math.exp(x) ** 0.5" in out), out


def test_use_libm_math_call_on_rewritten_power():
    """``math.exp(x ** 2.0)``: the power is rewritten first, and the math
    pass then sees ``_mx_sys._mx_pow(...)`` as an unknown operand, leaves
    the exp a Python call and logs that its argument is not provably a C
    number."""
    out = _rewrite_powers_then_math("return self.math.exp(x ** 2.0)")
    assert "_mx_sys._mx_exp(_mx_sys._mx_pow(x, 2.0))" in out, out
