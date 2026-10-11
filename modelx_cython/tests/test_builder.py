import types

import numpy as np
import pytest

from modelx_cython.builder import (
    CombinedCellsInfo,
    CombinedRefInfo,
    MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS,
)
from modelx_cython.parser import LexicalCellsInfo
from modelx_cython.tracer import ReturnTypeInfo
from modelx_cython.usage import UsageVerdict

FQ = "M._mx_classes._c_S._f_arr"


def make_lx():
    return LexicalCellsInfo("M._mx_classes", "_c_S", "arr", [])


def make_cells(spec=None, usage=None, ndim=1, value_type=np.float64,
               is_array=True):
    rt = types.SimpleNamespace(
        ret_type=ReturnTypeInfo(value_type, is_array=is_array, ndim=ndim),
        arg_types={},
    )
    cells = CombinedCellsInfo(None, make_lx(), rt, spec or {})
    if usage is not None:
        cells.usage = usage
    return cells


def test_rettype_without_verdict_keeps_memoryview():
    assert make_cells().get_rettype_expr(c_style=True) == "const double[:]"
    assert make_cells(ndim=2).get_rettype_expr(c_style=True) \
        == "const double[:, :]"
    assert make_cells().get_rettype_expr() == "_mx_cy.const[_mx_cy.double][:]"


def test_rettype_safe_verdict_keeps_memoryview():
    cells = make_cells(usage=UsageVerdict(FQ, 1, True, True))
    assert cells.get_rettype_expr(c_style=True) == "const double[:]"


def test_rettype_unsafe_verdict_falls_back_to_object():
    cells = make_cells(usage=UsageVerdict(FQ, 1, False, True))
    assert cells.get_rettype_expr(c_style=True) == "object"


def test_rettype_poisoned_verdict_falls_back_to_object():
    # unresolvable call sites mark unsafe without marking internal uses;
    # the unsafe flag must win over the no-internal-uses policy
    cells = make_cells(usage=UsageVerdict(FQ, 1, False, False))
    assert cells.get_rettype_expr(c_style=True) == "object"


def test_rettype_no_internal_uses_follows_policy():
    cells = make_cells(usage=UsageVerdict(FQ, 1, True, False))
    expected = ("const double[:]" if MEMORYVIEW_FOR_EXTERNAL_ONLY_ARRAYS
                else "object")
    assert cells.get_rettype_expr(c_style=True) == expected


def test_spec_memoryview_overrides_unsafe_verdict():
    cells = make_cells(spec={"return_type": "memoryview"},
                       usage=UsageVerdict(FQ, 1, False, True))
    assert cells.get_rettype_expr(c_style=True) == "const double[:]"


def test_spec_object_overrides_all():
    cells = make_cells(spec={"return_type": "object"},
                       usage=UsageVerdict(FQ, 1, True, True))
    assert cells.get_rettype_expr(c_style=True) == "object"


@pytest.mark.parametrize("kwargs", [
    {"is_array": False, "value_type": float},   # scalar return
    {"ndim": 0},                                # 0-d array
])
def test_spec_memoryview_invalid_return_raises(kwargs):
    with pytest.raises(ValueError,
                       match="requires a real-valued numpy array return"):
        make_cells(spec={"return_type": "memoryview"}, **kwargs)


def test_spec_memoryview_without_typeinfo_ignored(caplog):
    cells = CombinedCellsInfo(None, make_lx(), None,
                              {"return_type": "memoryview"})
    assert cells.get_rettype_expr(c_style=True) == "object"


# --------------------------------------------------------------------
# CombinedRefInfo


def make_ref(module, mx_class, cls="_c_Foo", name="bar", type_=object):
    rt = types.SimpleNamespace(type_=type_, mx_class=mx_class)
    return CombinedRefInfo(module, cls, name, rt_info=rt)


def test_ref_class_in_own_module_is_relative():
    ref = make_ref("Pkg._mx_classes", "Pkg._mx_classes._c_Bar")
    assert ref.is_relative
    assert ref.decl_type_expr == "_c_Bar"
    assert ref.get_type_expr(c_style=True) == "_c_Bar"


def test_ref_class_in_other_module_is_absolute():
    ref = make_ref("Pkg._mx_classes", "Pkg._m_Bar._mx_classes._c_Qux")
    assert not ref.is_relative
    assert ref.decl_type_expr == "Pkg._m_Bar._mx_classes._c_Qux"


def test_ref_class_in_prefix_sharing_module_is_absolute():
    """The relative test must hold on a dotted boundary.

    A module whose name merely extends this module's name is a
    different module, so a class defined there is declared by its full
    path.  Comparing the raw prefix instead marks it relative and cuts
    the wrong number of leading characters off, leaving the truncated
    ``ar._mx_classes._c_Qux`` in the generated declaration.
    """
    ref = make_ref("Pkg._m_Foo", "Pkg._m_FooBar._mx_classes._c_Qux")
    assert not ref.is_relative
    assert ref.decl_type_expr == "Pkg._m_FooBar._mx_classes._c_Qux"


def test_ref_holding_no_space_has_no_decl_type():
    ref = make_ref("Pkg._mx_classes", "", type_=int)
    assert not ref.is_relative
    assert ref.decl_type_expr == ""
    assert ref.get_type_expr(c_style=True) == "long long"


def test_ref_never_sampled_declares_object():
    ref = CombinedRefInfo("Pkg._mx_classes", "_c_Foo", "bar", rt_info=None)
    assert ref.type_ is None
    assert ref.mx_class == ""
    assert not ref.is_relative
    assert ref.get_type_expr(c_style=True) == "object"


# --------------------------------------------------------------------
# Arrays that are not real-valued


@pytest.mark.parametrize("value_type", [np.str_, np.bytes_, np.object_,
                                        np.bool_, np.complex128])
def test_rettype_non_real_array_is_object(value_type):
    """An array of strings is not a str, and no other non-real array
    is its element type or a memoryview either"""
    cells = make_cells(value_type=value_type)
    assert cells.get_rettype_expr(c_style=True) == "object"
    assert cells.get_rettype_expr() == "object"


def test_rettype_spec_bool_on_array_is_object():
    """A bool array does not coerce to a bint memoryview"""
    cells = make_cells(spec={"return_type": "bool"})
    assert cells.get_rettype_expr(c_style=True) == "object"


def test_rettype_0d_real_array_is_object():
    cells = make_cells(ndim=0)
    assert cells.get_rettype_expr(c_style=True) == "object"


def test_rettype_str_scalar_is_str():
    cells = make_cells(value_type=np.str_, is_array=False, ndim=0)
    assert cells.get_rettype_expr(c_style=True) == "str"


# --------------------------------------------------------------------
# Spec "param_type"


def make_param_cells(spec, arg_types, max_args=None, params=("t", "x")):
    lx = LexicalCellsInfo("M._mx_classes", "_c_S", "foo", list(params))
    rt = types.SimpleNamespace(
        ret_type=ReturnTypeInfo(float),
        arg_types=arg_types,
        max_args=max_args if max_args is not None else {},
    )
    return CombinedCellsInfo(None, lx, rt, spec)


def test_param_type_overrides_traced_types():
    cells = make_param_cells(
        {"param_type": {"x": "float", "t": "object"}},
        {"t": int, "x": int}, {"t": 3, "x": 4})
    assert cells.get_argtype_expr("x", c_style=True) == "double"
    assert cells.get_argtype_expr("x") == "_mx_cy.double"
    assert cells.get_argtype_expr("t", c_style=True) == "object"
    assert not cells.is_arrayable()     # no longer all-integral


def test_param_type_leaves_unnamed_params_traced():
    cells = make_param_cells({"param_type": {"x": "int"}},
                             {"t": int, "x": int}, {"t": 3, "x": 4})
    assert cells.get_argtype_expr("t", c_style=True) == "long long"
    assert cells.is_arrayable()


def test_param_type_int_without_traced_max_is_not_arrayable():
    """An argument traced as float has no maximum to size an array"""
    cells = make_param_cells({"param_type": {"x": "int"}},
                             {"t": int, "x": float}, {"t": 3})
    assert cells.get_argtype_expr("x", c_style=True) == "long long"
    assert cells.is_int_args
    assert not cells.is_arrayable()


@pytest.mark.parametrize("spec, match", [
    ({"param_type": {"y": "int"}}, "has no parameter named 'y'"),
    ({"param_type": {"x": "double"}}, "'double' is not one of"),
    ({"param_type": {"x": "memoryview"}}, "'memoryview' is not one of"),
    ({"param_type": ["x"]}, "expected a dict"),
])
def test_param_type_invalid_raises(spec, match):
    with pytest.raises(ValueError, match=match):
        make_param_cells(spec, {"t": int, "x": int})


def test_param_type_without_typeinfo_ignored(caplog):
    lx = LexicalCellsInfo("M._mx_classes", "_c_S", "foo", ["t"])
    with caplog.at_level("WARNING", logger="modelx_cython.builder"):
        cells = CombinedCellsInfo(None, lx, None, {"param_type": {"t": "int"}})
    assert cells.get_argtype_expr("t", c_style=True) == "object"
    assert "is ignored because no type information was sampled" in caplog.text


# --------------------------------------------------------------------
# Arrays used only outside the model


def test_rettype_uncached_no_internal_uses_is_object():
    """An uncached cells returned the array itself before it was typed;
    with no compiled caller, it keeps doing so"""
    cells = make_cells(usage=UsageVerdict(FQ, 1, True, False))
    cells.has_formula_def = False
    assert cells.get_rettype_expr(c_style=True) == "object"
    # with element access inside the model, a memoryview as for a cached
    cells.usage = UsageVerdict(FQ, 1, True, True)
    assert cells.get_rettype_expr(c_style=True) == "const double[:]"
    # and as the spec says
    cells = make_cells(spec={"return_type": "memoryview"},
                       usage=UsageVerdict(FQ, 1, True, False))
    cells.has_formula_def = False
    assert cells.get_rettype_expr(c_style=True) == "const double[:]"


# --------------------------------------------------------------------
# Defaults and rebinding that do not fit the traced types

import libcst as cst

from modelx_cython.builder import default_fits


@pytest.mark.parametrize("default, ctype, fits", [
    ("0", "long long", True), ("-1", "long long", True),
    ("+0x10", "long long", True), ("(3)", "long long", True),
    ("0.5", "long long", False), ("None", "long long", False),
    ("True", "long long", False), ("'a'", "long long", False),
    ("-1", "double", True), ("1.5e3", "double", True),
    ("None", "double", False), ("1j", "double", False),
    ("b'a'", "double", False), ("math.pi", "double", False),
    ("True", "bint", True), ("0", "bint", False), ("None", "bint", False),
    ("None", "str", True), ("'a' 'b'", "str", True),
    ("b'a'", "str", False), ("f'{x}'", "str", False), ("0", "str", False),
    ("None", "object", True), ("[1]", "object", True),
])
def test_default_fits(default, ctype, fits):
    assert default_fits(cst.parse_expression(default), ctype) is fits


def make_dflt_cells(arg_types, defaults, spec=None, has_formula_def=True,
                    params=("t", "k")):
    lx = LexicalCellsInfo(
        "M._mx_classes", "_c_S", "foo", list(params), list(defaults),
        {p: cst.parse_expression(d) for p, d in defaults.items()})
    rt = types.SimpleNamespace(
        ret_type=ReturnTypeInfo(float), arg_types=arg_types,
        max_args={p: 3 for p, t in arg_types.items() if t is int})
    cells = CombinedCellsInfo(None, lx, rt, spec or {})
    cells.has_formula_def = has_formula_def
    return cells


def test_default_not_of_traced_type_is_object(caplog):
    with caplog.at_level("INFO", logger="modelx_cython.builder"):
        cells = make_dflt_cells({"t": int, "k": int}, {"k": "0.5"})
    for formula in (False, True):
        assert cells.get_argtype_expr("k", c_style=True, formula=formula) \
            == "object"
        assert cells.get_argtype_expr("t", c_style=True, formula=formula) \
            == "long long"
    assert not cells.is_arrayable()
    assert ("parameter 'k' of M._mx_classes._c_S.foo is typed object rather "
            "than long long: its default value 0.5 is not a long long "
            "literal") in caplog.text


def test_default_of_traced_type_is_kept():
    cells = make_dflt_cells({"t": int, "k": float}, {"k": "1"})
    assert cells.get_argtype_expr("k", c_style=True) == "double"
    cells = make_dflt_cells({"t": int, "k": int}, {"k": "-2"})
    assert cells.get_argtype_expr("k", c_style=True) == "long long"
    assert cells.is_arrayable()


def test_default_overrides_spec_param_type_with_warning(caplog):
    with caplog.at_level("WARNING", logger="modelx_cython.builder"):
        cells = make_dflt_cells({"t": int, "k": float}, {"k": "None"},
                                spec={"param_type": {"k": "float"}})
    assert cells.get_argtype_expr("k", c_style=True) == "object"
    assert "spec 'param_type' for parameter 'k'" in caplog.text


def test_rebound_param_of_cached_cells_is_object_in_formula_only():
    cells = make_dflt_cells({"t": int, "k": int}, {})
    assert cells.is_arrayable()
    cells.demote_rebound_param("t", "reason")
    assert cells.get_argtype_expr("t", c_style=True, formula=True) == "object"
    assert cells.get_argtype_expr("t", c_style=True) == "long long"
    assert cells.is_arrayable()     # the cache keeps its C array


def test_rebound_param_of_uncached_cells_is_object():
    cells = make_dflt_cells({"t": int, "k": int}, {}, has_formula_def=False)
    cells.demote_rebound_param("t", "reason")
    for formula in (False, True):
        assert cells.get_argtype_expr("t", c_style=True, formula=formula) \
            == "object"
