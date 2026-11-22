"""cells_param_size at the model level and in enclosing Spaces: nearest declaration wins."""
from modelx_cython.config import TransSpec


def test_declared_sizes_inherit_by_space_path():
    spec = TransSpec({'cells_param_size': {'t': 100, 'x': 121, ('t', 'k'): (100, 3)},
                      'spaces': {'Outer': {'cells_param_size': {'t': 200},
                                           'spaces': {'Inner': {'cells_params': {'k': {'size': 5}}}}}}})
    assert spec.get_declared_sizes('Pkg._mx_classes._c_Other') == (
        {('t', 'k'): ((100, 3), 'model')}, {'t': (100, 'model'), 'x': (121, 'model')})
    assert spec.get_declared_sizes('Pkg._m_Outer._mx_classes._c_Inner') == (
        {('t', 'k'): ((100, 3), 'model')},
        {'t': (200, 'Outer'), 'x': (121, 'model'), 'k': (5, 'Outer.Inner')})


def test_space_level_only_is_upstream_form():
    spec = TransSpec({'spaces': {'Projection': {'cells_param_size': {'t': 241, ('i', 'j'): (3, 6)}}}})
    assert spec.get_declared_sizes('Pkg._mx_classes._c_Projection') == (
        {('i', 'j'): ((3, 6), 'Projection')}, {'t': (241, 'Projection')})
    assert spec.get_declared_sizes('Pkg._mx_classes._c_Other') == ({}, {})


def test_declared_lower_bound_pairs():
    """R-3-lower: a declaration is a size or a (lower, stop) pair; the stop keeps its place in the sizes."""
    import pytest
    spec = TransSpec({'cells_param_size': {'gen': (1940, 1981), 't': 10, ('x', 'y'): ((5, 20), 3)},
                      'spaces': {'Inner': {'cells_param_size': {'gen': 1981}}}})
    assert spec.get_declared_sizes('Pkg._mx_classes._c_Other') == (
        {('x', 'y'): ((20, 3), 'model')}, {'gen': (1981, 'model'), 't': (10, 'model')})
    assert spec.get_declared_lowers('Pkg._mx_classes._c_Other') == ({('x', 'y'): {'x': 5}}, {'gen': 1940})
    # a nearer plain size restores today's meaning (lower = min(0, sampled minimum))
    assert spec.get_declared_lowers('Pkg._mx_classes._c_Inner') == ({('x', 'y'): {'x': 5}}, {})
    for bad in ((5, 5), (1, 2, 3), (1.0, 2)):
        with pytest.raises(ValueError, match='lower, stop'):
            TransSpec({'cells_param_size': {'g': bad}}).get_declared_sizes('Pkg._mx_classes._c_A')
