import pathlib

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
