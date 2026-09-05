import logging
import sys
import os
import inspect
import subprocess
import pathlib
import shutil
import pytest


@pytest.fixture
def sample_dir(tmp_path_factory, request):
    sample = request.param
    dst = tmp_path_factory.mktemp("temp") / "samples" / sample
    shutil.copytree(pathlib.Path(__file__).parent / "samples" / sample, dst)
    return dst


@pytest.mark.parametrize("sample_dir, model", [["basicterm_s", "BasicTerm_S"],
                                               ["basicterm_sc", "BasicTerm_SC"]], indirect=["sample_dir"])
@pytest.mark.parametrize("target", ["mx2cy", pytest.param("main", marks=pytest.mark.skip(reason="Skipping 'main' target"))])
def test_mx2cy_with_lifelib(sample_dir, target, model):
    import lifelib
    import modelx as mx

    work_dir = sample_dir # / "basicterm_s"
    lifelib.create('basiclife', work_dir / 'basiclife')
    mx.read_model(work_dir / 'basiclife' / model).export(work_dir / (model + '_nomx'))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
         "--spec", str(work_dir / "spec.py"),
         "--sample", str(work_dir / "sample.py")]

    if target == "mx2cy":
        assert subprocess.run(argv, env=env).returncode == 0
    elif target == "main":
        from modelx_cython.cli import main
        assert main(argv[1:], sys.stdout, sys.stderr) == 0

    if model == "BasicTerm_S":
        # the usage classifier detects disc_factors' whole-array consumers
        # without a spec override
        pxd = (work_dir / (model + "_nomx_cy") / "_mx_classes.pxd").read_text()
        assert "cdef object _v_disc_factors\n" in pxd

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0

    result = subprocess.run(
        [sys.executable, str(work_dir / "benchmark.py")],
        env=env,
        capture_output=True,
        text=True
    )
    print(result.stdout.strip())
    assert result.returncode == 0


def generate_nomx(work_dir: pathlib.Path, model: str):
    import modelx as mx
    mx.read_model(work_dir / model).export(work_dir / (model + '_nomx'))
    del mx.get_models()[model]


def export_has_use_slots():
    """Whether modelx is new enough to export ``__slots__`` (v0.33.0)"""
    import modelx as mx
    return "use_slots" in inspect.signature(mx.core.model.Model.export).parameters


def export_has_locked_spaces():
    """Whether modelx is new enough to export locked Spaces (v0.33.0)"""
    import modelx as mx
    return "locked_spaces" in inspect.signature(
        mx.core.model.Model.export).parameters


def get_env(work_dir: pathlib.Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_dir) + os.pathsep + env.get("PYTHONPATH", "")
    return env


@pytest.mark.parametrize("sample_dir, model", [["ref_space", "RefSpace"],
                                               ["nested_params", "NestedParams"],
                                               ["duplicated_params", "DuplicatedParams"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("target", ["mx2cy", pytest.param("main", marks=pytest.mark.skip(reason="Skipping 'main' target"))])
def test_mx2cy_with_ref_space(sample_dir, target, model):

    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--spec", str(work_dir / "spec.py"),
            "--sample", str(work_dir / "sample.py")]

    if target == "mx2cy":
        assert subprocess.run(argv, env=env).returncode == 0
    elif target == "main":
        from modelx_cython.cli import main
        assert main(argv[1:], sys.stdout, sys.stderr) == 0

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["ref_space", "RefSpace"]],
                         indirect=["sample_dir"])
def test_translate_is_same_for_both_export_styles(sample_dir, model):
    """Refs are traced off __slots__ as well as off __dict__

    modelx v0.33.0 declares __slots__ on the exported Space classes by
    default, and a slotted instance has no __dict__ to read the refs
    from.  Either export must translate to the same Cython source, and
    the __slots__ tuple must not survive into the cdef class, where it
    would name members that are no longer reachable by getattr.
    """
    import modelx as mx

    if not export_has_use_slots():
        pytest.skip("modelx export() has no use_slots parameter")

    work_dir = sample_dir
    env = get_env(work_dir)
    exported = work_dir / (model + "_nomx")
    translated = work_dir / (model + "_nomx_cy")
    sources = {}

    # Export twice from one model object rather than calling
    # generate_nomx twice: modelx writes the id() of its IOSpec objects
    # into the export, and a second read_model gives different ids.
    model_obj = mx.read_model(work_dir / model)
    try:
        for use_slots in (True, False):
            model_obj.export(exported, use_slots=use_slots)

            assert ("__slots__" in (exported / "_mx_classes.py").read_text(
                encoding="utf-8")) is use_slots

            assert subprocess.run(
                [sys.executable, "-m", "modelx_cython", str(exported),
                 "--spec", str(work_dir / "spec.py"),
                 "--sample", str(work_dir / "sample.py"),
                 "--translate-only"], env=env).returncode == 0

            sources[use_slots] = {
                path.relative_to(translated).as_posix():
                    path.read_text(encoding="utf-8")
                for path in sorted(translated.rglob("_mx_classes.py"))
            }
            shutil.rmtree(exported)
            shutil.rmtree(translated)
    finally:
        del mx.get_models()[model]

    assert not any("__slots__" in src for src in sources[True].values())
    assert sources[True] == sources[False]

    # the refs the exported __slots__ used to hide are declared instead
    top = sources[True]["_mx_classes.py"]
    assert "    bar: _c_Bar\n" in top
    assert "    i: _mx_cy.longlong\n" in top


@pytest.mark.parametrize("sample_dir", ["locked_spaces"], indirect=True)
def test_locked_spaces(sample_dir):
    """A model exported with ``locked_spaces`` compiles and runs from threads.

    The locked classes keep the double-checked locking of the export: the
    ``_has_`` flags go through the acquire/release accessors of
    ``_mx_sys.pxd``, dict caches are created in ``__init__``, and the
    unlocked export of the same model translates without any of it.
    """
    if not export_has_locked_spaces():
        pytest.skip("modelx export() has no locked_spaces parameter")

    work_dir = sample_dir
    env = get_env(work_dir)
    assert subprocess.run(
        [sys.executable, str(work_dir / "build_model.py")], env=env
    ).returncode == 0

    for pkg, sample in (("LockedSpaces_nomx", "sample.py"),
                        ("LockedSpacesU_nomx", "sample_unlocked.py")):
        assert subprocess.run(
            [sys.executable, "-m", "modelx_cython", str(work_dir / pkg),
             "--spec", str(work_dir / "spec.py"),
             "--sample", str(work_dir / sample),
             "--translate-only"], env=env).returncode == 0

    locked = work_dir / "LockedSpaces_nomx_cy"
    unlocked = work_dir / "LockedSpacesU_nomx_cy"

    top = (locked / "_mx_classes.py").read_text(encoding="utf-8")
    data = top.split("class _c_Data")[1].split("\nclass ")[0]
    proj = top.split("class _c_Projection")[1].split("\nclass ")[0]
    table = top.split("class _c_Table")[1].split("\nclass ")[0]
    child = (locked / "_m_Table" / "_mx_classes.py").read_text(encoding="utf-8")

    # locked classes: atomic flag accessors on the fast path, the model
    # lock on the miss path, and the value stored before the flag
    for src, name in [(data, "scale"), (data, "table_arr"),
                      (table, "key_weight"), (child, "child_val")]:
        body = src.split(f"def {name}(self)")[1].split("\n    def ")[0]
        assert f"_mx_sys._mx_load_flag(_mx_cy.address(self._has_{name}))" in body
        assert "with self._mx_lock:" in body
        assert body.index(f"self._v_{name} = val") < body.index(
            f"_mx_sys._mx_store_flag(_mx_cy.address(self._has_{name}), True)")
    for src, name in [(data, "rate"), (table, "rec")]:
        body = src.split(f"def {name}(self")[1].split("\n    def ")[0]
        assert f"_mx_load_flag(_mx_cy.address(self._has_{name}[" in body
        assert "with self._mx_lock:" in body
    # a dict-cached cells keeps the exported body and its dict in __init__
    assert "self._v_lookup = {}" in data.split("def _mx_assign_refs")[0]
    assert "if self._v_lookup is None" not in data
    assert "if name in self._v_lookup:" in data
    assert "self._mx_lock = self._model._mx_lock" in data
    # __call__ keeps the exported body
    assert "_mx_root = self._mx_itemspaces.get(_mx_key)" in table

    # the per-thread Space is untouched
    assert "_mx_lock" not in proj
    assert "if self._v_pv is None:" not in proj      # arrayable, no dict
    # the unlocked export carries no lock at all
    for path in unlocked.rglob("*.py*"):
        if path.name != "_mx_sys.pxd":
            assert "_mx_lock" not in path.read_text(encoding="utf-8"), path
    assert 'compiler_directives={"freethreading_compatible": True}' in (
        work_dir / "setup.py").read_text(encoding="utf-8")

    # compile the locked model and run it from eight threads
    assert subprocess.run(
        [sys.executable, "-m", "modelx_cython", str(work_dir / "LockedSpaces_nomx"),
         "--spec", str(work_dir / "spec.py"),
         "--sample", str(work_dir / "sample.py")], env=env).returncode == 0
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")], env=env
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["no_spec", "NoSpec"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("target", ["mx2cy", pytest.param("main", marks=pytest.mark.skip(reason="Skipping 'main' target"))])
@pytest.mark.parametrize("no_spec", [True, False])
def test_no_spec(sample_dir, target, model, no_spec):

    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py")]

    if no_spec:
        argv.append("--no-spec")

    if target == "mx2cy":
        assert subprocess.run(argv, env=env).returncode == int(not no_spec)
    elif target == "main":
        from modelx_cython.cli import main
        if no_spec:
            assert main(argv[1:], sys.stdout, sys.stderr) == int(not no_spec)
        else:
            with pytest.raises(FileNotFoundError):
                main(argv[1:], sys.stdout, sys.stderr)

    if no_spec:
        assert subprocess.run(
            [sys.executable, str(work_dir / "assert_cy.py")],
            env=env
        ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["varying_types_of_args", "VaryingArgTypes"]],
                         indirect=["sample_dir"])
def test_varying_arg_types(sample_dir, model, caplog):
    """int and float numbers are given to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--no-spec"]

    assert (result := subprocess.run(argv + ['--log-level', 'INFO'], env=env, capture_output=True, text=True)).returncode == 0
    assert "varying types given to argument 'i' in VaryingArgTypes_nomx._mx_classes._c_Space1._f_foo: int 1, float 2.0" in result.stderr
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["varying_integral_types_of_args", "VaryingIntegralArgTypes"]],
                         indirect=["sample_dir"])
def test_varying_integral_arg_types(sample_dir, model):
    """int and np.int64 numbers are passed to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--no-spec"]

    assert subprocess.run(argv, env=env).returncode == 0
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0

    assert "cdef long long[3] _v_foo" in (work_dir / (model + "_nomx_cy") / "_mx_classes.pxd").read_text()


@pytest.mark.parametrize("sample_dir, model, spec", [["deep_recursion", "DeepRecursion", ""],
                                               ["index_range", "IndexRange", "--no-spec"]],
                         indirect=["sample_dir"])
def test_deep_recursion_and_index_range(sample_dir, model, spec):
    """int and np.int64 numbers are passed to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py")]

    if spec:
        argv.append(spec)

    assert subprocess.run(argv, env=env, cwd=work_dir).returncode == 0
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env,
        capture_output=True,
        text=True
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["various_types", "VariousTypes"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("sample, assertion", [['sample.py', 'assert_cy.py'],
                                               ['no_sample.py', 'assert_cy_no_sample.py']])
def test_various_types(sample_dir, model, sample, assertion):
    """int and np.int64 numbers are passed to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / sample),
            "--no-spec"]

    assert subprocess.run(argv, env=env, cwd=work_dir).returncode == 0
    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env,
        capture_output=True,
        text=True
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["array_usage", "ArrayUsage"]],
                         indirect=["sample_dir"])
def test_array_usage(sample_dir, model):
    """Memoryview return types are kept only for element-access-only cells"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--no-spec"]

    assert subprocess.run(argv, env=env, cwd=work_dir).returncode == 0

    pxd = (work_dir / (model + "_nomx_cy") / "_mx_classes.pxd").read_text()

    # element-access-only consumers keep the memoryview
    assert "cdef const double[:] _v_arr_elem\n" in pxd
    assert "cdef const double[:, :] _v_arr2d\n" in pxd
    assert "cdef const double[:] _v_arr_cross_elem\n" in pxd
    assert "cpdef const double[:] arr_elem(_c_Data self)\n" in pxd

    # whole-array consumers fall back to object
    assert "cdef object _v_arr_whole\n" in pxd
    assert "cdef object _v_arr_sliced\n" in pxd
    assert "cdef object _v_arr_cross_whole\n" in pxd
    assert "cpdef object arr_whole(_c_Data self)\n" in pxd

    # no model-internal uses: the memoryview is kept (policy)
    assert "cdef const double[:] _v_arr_external\n" in pxd

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env
    ).returncode == 0

    # spec return_type "memoryview" overrides the classifier
    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--spec", str(work_dir / "spec_force_mv.py"),
            "--translate-only"]

    assert subprocess.run(argv, env=env, cwd=work_dir).returncode == 0
    pxd = (work_dir / (model + "_nomx_cy") / "_mx_classes.pxd").read_text()
    assert "cdef const double[:] _v_arr_whole\n" in pxd

    # "memoryview" on a cells not returning a real-valued array is an error
    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--spec", str(work_dir / "spec_bad_mv.py"),
            "--translate-only"]

    result = subprocess.run(argv, env=env, cwd=work_dir,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires a real-valued numpy array return" in result.stderr


@pytest.mark.parametrize("sample_dir, model", [["array_size", "ArraySize"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("spec", [["--no-spec"], ["--spec", "spec_large.py"], ["--spec", "spec_small.py"]])
def test_array_size(sample_dir, model, spec):
    """int and np.int64 numbers are passed to the same arg"""
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            # "--sample", str(work_dir / "sample.py"),
            "--log-level", "INFO"]
    argv += spec

    assert (result:= subprocess.run(argv, env=env, cwd=work_dir,  capture_output=True, text=True)).returncode == 0

    if spec[-1] == "spec_small.py":
        assert "INFO:modelx_cython.builder:Specified max size of 6 for cells parameter i in _c_Space1 is replaced by 11 from ArraySize_nomx_cy._mx_classes._c_Space1._f_foo" in result.stderr
        assert "INFO:modelx_cython.builder:Specified max size of 3 for cells parameter i in _c_Space1 is replaced by 6 from ArraySize_nomx_cy._mx_classes._c_Space1._f_bar" in result.stderr
        assert "INFO:modelx_cython.builder:Specified max size of 6 for cells parameter j in _c_Space1 is replaced by 11 from ArraySize_nomx_cy._mx_classes._c_Space1._f_bar" in result.stderr

    assert subprocess.run(
        [sys.executable, str(work_dir / "assert_cy.py")],
        env=env,
        capture_output=True,
        text=True
    ).returncode == 0


@pytest.mark.parametrize("sample_dir, model", [["size_spec_change", "SizeSpecChange"]],
                         indirect=["sample_dir"])
@pytest.mark.parametrize("spec", ["spec_old.py", "spec_new.py"])
def test_size_spec_change(sample_dir, model, spec):
    """Array sizes follow the parameter sizes given in the spec file

    ``spec_old.py`` uses the deprecated ``cells_params``/``size`` keys and
    ``spec_new.py`` the current ``cells_param_size`` key, and the two specify
    different sizes. Both forms are honoured, so only the larger sizes in
    ``spec_new.py`` admit the parameter values in ``assert_cy_new.py``.
    """
    generate_nomx(work_dir := sample_dir, model)
    env = get_env(work_dir)

    argv = [sys.executable, "-m", "modelx_cython", str(work_dir / (model + "_nomx")),
            "--sample", str(work_dir / "sample.py"),
            "--spec", str(work_dir / spec)]

    assert (result := subprocess.run(argv, env=env, capture_output=True, text=True)).returncode == 0
    assert subprocess.run([sys.executable, str(work_dir / "assert_cy_old.py")], env=env).returncode == 0
    assert subprocess.run([sys.executable, str(work_dir / "assert_cy_new.py")], env=env
                          ).returncode == (1 if spec == 'spec_old.py' else 0)