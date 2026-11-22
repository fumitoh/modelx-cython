import pathlib
from types import SimpleNamespace

import Cython.Build
import setuptools

from modelx_cython.cli import iter_module_files, create_setup


def test_create_setup_declares_free_threading(tmp_path):
    """Importing a compiled model must not re-enable the GIL on 3.13t+."""
    setup_file = tmp_path / "setup.py"
    create_setup("Model_nomx_cy",
                 [pathlib.PurePath("Model_nomx_cy/_mx_sys.py"),
                  pathlib.PurePath("Model_nomx_cy/_mx_classes.py")],
                 setup_file)
    code = setup_file.read_text(encoding="utf-8")
    assert 'compiler_directives={"freethreading_compatible": True}' in code
    assert '"Model_nomx_cy/_mx_classes.py"' in code


def test_create_setup_uses_cython_output_as_generated(tmp_path, monkeypatch):
    setup_file = tmp_path / "setup.py"
    module = pathlib.PurePath("Model_nomx_cy/_mx_classes.py")
    create_setup("Model_nomx_cy", [module], setup_file)
    source = tmp_path / "_mx_classes.c"
    dense = "static CYTHON_INLINE double __pyx_f_13Model_nomx_cy_11_mx_classes__mx_cache_entry_0(void)"
    original = f"{dense};\n{dense} {{ return 0.0; }}\n"
    source.write_text(original, encoding="utf-8")
    extensions = [SimpleNamespace(name="Model_nomx_cy._mx_classes", sources=[str(source)])]
    received = {}

    def cythonize(paths, **kwargs):
        assert paths == [module.as_posix()]
        return extensions

    monkeypatch.setattr(Cython.Build, "cythonize", cythonize)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: received.update(kwargs))
    exec(compile(setup_file.read_text(encoding="utf-8"), str(setup_file), "exec"), {})
    assert received["ext_modules"] is extensions
    assert source.read_text(encoding="utf-8") == original


def test_iter_module_files(tmp_path):
    root = tmp_path / "Model_nomx_cy"
    (root / "_m_Sub" / "_m_Nested").mkdir(parents=True)
    (root / "_mx_model.py").write_text("")
    (root / "_mx_classes.py").write_text("")
    (root / "_mx_sys.py").write_text("")    # not a model/space module
    (root / "_m_Sub" / "_mx_classes.py").write_text("")
    (root / "_m_Sub" / "_m_Nested" / "_mx_classes.py").write_text("")

    found = dict(iter_module_files(root))

    assert set(found) == {
        "Model_nomx_cy._mx_model",
        "Model_nomx_cy._mx_classes",
        "Model_nomx_cy._m_Sub._mx_classes",
        "Model_nomx_cy._m_Sub._m_Nested._mx_classes",
    }
    for fqname, path in found.items():
        assert path.is_file()
