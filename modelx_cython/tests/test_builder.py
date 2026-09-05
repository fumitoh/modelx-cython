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
