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


class TestCompilerDirectives:

    def test_absent(self):
        assert TransSpec({}).get_compiler_directives() == {}

    def test_valid(self):
        directives = {"infer_types": True, "cdivision": False,
                      "language_level": 3, "c_string_type": "str"}
        spec = TransSpec({"compiler_directives": directives,
                          "spaces": {"Projection": testdata_space}})
        assert spec.get_compiler_directives() == directives

    def test_infer_types_none(self):
        """None is infer_types' default: safe inference only"""
        spec = TransSpec({"compiler_directives": {"infer_types": None}})
        assert spec.get_compiler_directives() == {"infer_types": None}

    @pytest.mark.parametrize("directives, match", [
        (["infer_types"], "expected a dict"),
        ({"infer_type": True}, "is not a Cython compiler directive"),
        ({"infer_types": 1}, "must be True or False or None"),
        ({"boundscheck": None}, "must be True or False, got None"),
        ({"boundscheck": "True"}, "must be True or False"),
        ({"c_string_type": "text"}, "must be a value Cython accepts"),
        ({"locals": {}}, "cannot be set for a whole module"),
        ({"nogil": True}, "cannot be set for a whole module"),
    ])
    def test_invalid(self, directives, match):
        spec = TransSpec({"compiler_directives": directives})
        with pytest.raises(ValueError, match=match):
            spec.get_compiler_directives()

    @pytest.mark.parametrize("directives", [
        {"cpow": True},
        {"cpow": 1},
        {"language_level": 2},
        {"language_level": "2"},
        {"annotation_typing": False},
        {"legacy_implicit_noexcept": True},
    ])
    def test_unsupported_values(self, directives):
        """Values that would silently change what the compiled model
        computes: 2 ** -3 is 0 with cpow=True, 3 / 12 is 0 under
        language_level=2, ..."""
        spec = TransSpec({"compiler_directives": directives})
        with pytest.raises(ValueError,
                           match="mx2cy does not support the Cython directive"):
            spec.get_compiler_directives()

    @pytest.mark.parametrize("directives", [
        {"cpow": False},
        {"cpow": None},
        {"language_level": "3str"},
        {"annotation_typing": True},
        {"legacy_implicit_noexcept": False},
    ])
    def test_supported_values_of_unsupported_directives(self, directives):
        spec = TransSpec({"compiler_directives": directives})
        assert spec.get_compiler_directives() == directives
