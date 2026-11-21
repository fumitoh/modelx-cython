import pytest

from modelx_cython.config import TransSpec


testdata_nested = {"cells": {"sample": None}}
testdata_space = {
    "cells_params": {"t": {"size": 241}},
    "cells": {"disc_factors": {"return_type": "object"}},
    "spaces": {"Nested": testdata_nested},
}

testdata = {"spaces": {"Projection": testdata_space}}


class TestConf:
    @pytest.mark.parametrize(
        "obj_path",
        ["ModelName._mx_class_._c_Projection", "ModelName._mx_class_._m_Projection"],
    )
    def test_get_data(self, obj_path):
        data = TransSpec(testdata).get_spec(obj_path)
        assert data == testdata_space

    @pytest.mark.parametrize(
        "obj_path",
        [
            "ModelName._m_Projection._mx_classes._c_Nested",
            "ModelName._m_Projection._m_Nested",
        ],
    )
    def test_get_data2(self, obj_path):
        data = TransSpec(testdata).get_spec(obj_path)
        assert data == testdata_nested


@pytest.mark.parametrize('data',[
    {'automatic_optimizations':1}, {'native_math':'yes'}, {'parallel_workers':True},
    {'parallel_workers':0}, {'native_cache_hits':True},
    {'dense_miss_binding':False,'native_cache_hits':'both'},
    {'dense_miss_binding':False,'propagate_exact_receiver':True},
])
def test_invalid_dependent_policy_is_rejected_before_emission(data):
    with pytest.raises(ValueError):
        TransSpec(data)


def test_explicit_policy_overrides_keep_their_existing_dependencies():
    policy=TransSpec({'automatic_optimizations':False,'dense_miss_binding':True,
        'propagate_exact_receiver':True,'native_cache_hits':'both','parallel_workers':4}).policy
    assert policy.propagate_exact_receiver and policy.native_cache_hits=='both'
    assert policy.parallel_workers==4 and policy.native_math


def test_disabling_automatic_selection_disables_implicit_dense_entries():
    policy = TransSpec({'automatic_optimizations': False}).policy
    assert not policy.dense_miss_binding
    assert not policy.propagate_exact_receiver
    assert policy.native_cache_hits == 'none'


def test_guarded_rounding_can_be_disabled():
    assert TransSpec({}).policy.native_math
    assert TransSpec({'native_math': True}).policy.native_math
    assert not TransSpec({'native_math': False}).policy.native_math
