"""Generated model modules compile without floating-point contraction (W1.5)."""
import pathlib
from types import SimpleNamespace

from setuptools import Distribution
from setuptools.command.build_ext import build_ext

from modelx_cython.build import FP_CONTRACT_OFF, create_setup, fp_contract_off


def test_generated_setup_wraps_the_build_command(tmp_path):
    setup_file = tmp_path / "setup.py"
    create_setup("Model_nomx_cy", [pathlib.PurePath("Model_nomx_cy/_mx_classes.py"),
                                   pathlib.PurePath("Model_nomx_cy/_mx_helper_x.pyx")],
                 setup_file, providers=[pathlib.PurePath("Model_nomx_cy/_mx_helper_x.pyx")])
    code = setup_file.read_text(encoding="utf-8")
    assert "cmdclass=fp_contract_off(provider_build_cache(" in code
    assert "['Model_nomx_cy._mx_helper_x'])" in code


def _built(compiler_type, name, providers=(), args=None):
    seen = []

    class Recording(build_ext):
        def build_extension(self, extension):
            seen.append(list(extension.extra_compile_args or ()))

    command = fp_contract_off({"build_ext": Recording}, providers)["build_ext"](Distribution())
    command.compiler = SimpleNamespace(compiler_type=compiler_type)
    command.build_extension(SimpleNamespace(name=name, extra_compile_args=args))
    return seen[0]


def test_model_modules_get_the_flag_once_on_gcc_style_compilers():
    assert _built("unix", "m._mx_classes") == [FP_CONTRACT_OFF]
    assert _built("mingw32", "m._mx_classes", args=["-O2"]) == ["-O2", FP_CONTRACT_OFF]
    assert _built("unix", "m._mx_classes", args=[FP_CONTRACT_OFF]) == [FP_CONTRACT_OFF]


def test_providers_and_msvc_are_left_alone():
    assert _built("unix", "m._mx_helper_x", providers=["m._mx_helper_x"]) == []
    assert _built("msvc", "m._mx_classes") == []
