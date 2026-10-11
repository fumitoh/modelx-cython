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


class TestUseLibm:

    def test_default_off(self):
        assert TransSpec({}).get_use_libm() is False

    @pytest.mark.parametrize("value", [True, False])
    def test_valid(self, value):
        spec = TransSpec({"use_libm": value,
                          "spaces": {"Projection": testdata_space}})
        assert spec.get_use_libm() is value

    @pytest.mark.parametrize("value", [1, 0, "True", None])
    def test_invalid(self, value):
        with pytest.raises(ValueError, match="expected True or False"):
            TransSpec({"use_libm": value}).get_use_libm()
