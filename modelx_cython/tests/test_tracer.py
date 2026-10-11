import logging
import numbers
import types

import numpy as np
import pytest

from modelx_cython.tracer import (
    MxCallTraceLogger,
    MxCodeFilter,
    ReturnTypeInfo,
    RuntimeCellsInfo,
    RuntimeValueInfo,
    instance_attrs,
)
from modelx_cython.monkeytype_tracing import CallTrace


def foo(self, i):
    pass


FOO = foo.__module__ + ".foo"


def make_trace(ret_val, i):
    return CallTrace(foo, {"self": None, "i": i}, ret_val=ret_val)


@pytest.mark.parametrize("ret_vals", [
    [np.array([1.0, 2.0]), 3],
    [3, np.array([1.0, 2.0])],
    [np.array([1.0, 2.0]), 3, np.array([1.0, 2.0])]
])
def test_ret_type_array_and_non_array(ret_vals, caplog):
    """Array and non-array values returned from the same cells collapse to object"""
    traces = [make_trace(val, i) for i, val in enumerate(ret_vals)]
    with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
        info = RuntimeCellsInfo(traces)

    assert info.ret_type == ReturnTypeInfo(object)
    msgs = [r.message for r in caplog.records
            if "varying array and non-array types returned from" in r.message]
    assert len(msgs) == 1
    assert "array of float64" in msgs[0]
    assert "int" in msgs[0]


def test_ret_type_varying_non_array_types(caplog):
    """Both conflicting types appear in the log for non-array return values"""
    traces = [make_trace(1, 0), make_trace("abc", 1)]
    with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
        info = RuntimeCellsInfo(traces)

    assert info.ret_type == ReturnTypeInfo(object)
    msgs = [r.message for r in caplog.records
            if "varying types returned from" in r.message]
    assert len(msgs) == 1
    assert "int for i=0" in msgs[0]
    assert "str for i=1" in msgs[0]


class Dicted:
    """Attributes in ``__dict__``, as modelx exports before v0.33.0"""

    def __init__(self):
        self.a = 1
        self.b = 2


class Slotted:
    """Attributes in slots, as modelx exports with ``use_slots=True``"""

    __slots__ = ("a", "b", "never_set")

    def __init__(self):
        self.a = 1
        self.b = 2


class SlottedChild(Slotted):

    __slots__ = ("c",)

    def __init__(self):
        super().__init__()
        self.c = 3


class SlottedOnDicted(Dicted):
    """A base without ``__slots__`` gives the subclass a ``__dict__`` too"""

    __slots__ = ("c",)

    def __init__(self):
        super().__init__()
        self.c = 3


class Mangled:

    __slots__ = ("__x",)

    def __init__(self):
        self.__x = 1


class SlottedUnsorted:

    __slots__ = ("zeta", "alpha")

    def __init__(self):
        self.zeta = 1
        self.alpha = 2


class ShadowingSlot(Slotted):

    __slots__ = ("a",)

    def __init__(self):
        super().__init__()
        self.a = 99


def test_instance_attrs_reads_dict():
    assert list(instance_attrs(Dicted())) == [("a", 1), ("b", 2)]


def test_instance_attrs_reads_slots():
    """Unassigned slots are skipped rather than raising"""
    assert list(instance_attrs(Slotted())) == [("a", 1), ("b", 2)]


def test_instance_attrs_walks_mro():
    assert list(instance_attrs(SlottedChild())) == [("c", 3), ("a", 1), ("b", 2)]


def test_instance_attrs_reads_dict_and_slots():
    assert sorted(instance_attrs(SlottedOnDicted())) == [
        ("a", 1), ("b", 2), ("c", 3)]


def test_instance_attrs_reads_mangled_slot_by_stored_name():
    """A private slot is stored under the name mangled with its class,
    which is what getattr resolves, not the ``__slots__`` entry"""
    assert Mangled.__slots__ == ("__x",)
    assert list(instance_attrs(Mangled())) == [("_Mangled__x", 1)]


def test_instance_attrs_yields_slots_in_descriptor_order():
    """CPython sorts a class's slot descriptors, so the order the names
    come out in is not the order ``__slots__`` declares them"""
    assert list(instance_attrs(SlottedUnsorted())) == [("alpha", 2), ("zeta", 1)]


def test_instance_attrs_yields_a_shadowed_slot_once():
    """A slot redeclared in a subclass shadows the base one, as getattr
    resolves it"""
    assert list(instance_attrs(ShadowingSlot())) == [("a", 99), ("b", 2)]


# --------------------------------------------------------------------
# RuntimeValueInfo.init_mxobj


MODEL_PKG = "Model_nomx"


def make_value(module, qualname="_c_Foo"):
    """An instance of a class that reports ``module`` as its module"""
    cls = type(qualname, (), {"__module__": module})
    return cls()


def test_mxobj_in_the_model_root_module_is_detected():
    info = RuntimeValueInfo.init_mxobj(
        make_value("Model_nomx._mx_classes"), MODEL_PKG)
    assert info.mx_class == "Model_nomx._mx_classes._c_Foo"


def test_mxobj_in_a_model_submodule_is_detected():
    info = RuntimeValueInfo.init_mxobj(
        make_value("Model_nomx._m_Bar._mx_classes"), MODEL_PKG)
    assert info.mx_class == "Model_nomx._m_Bar._mx_classes._c_Foo"


def test_mxobj_in_a_prefix_sharing_package_is_not_detected():
    """Only the top-level package name decides, not a raw prefix.

    A package whose name merely extends the model package's name is a
    different package.  Comparing the raw prefix records its class as a
    model space class, and the ref is then declared with, and its
    package cimported by, a .pxd that the model does not build.
    """
    info = RuntimeValueInfo.init_mxobj(
        make_value("Model_nomx_cy._mx_classes"), MODEL_PKG)
    assert info.mx_class == ""


def test_mxobj_outside_the_model_package_is_not_detected():
    info = RuntimeValueInfo.init_mxobj(1, MODEL_PKG)
    assert info.mx_class == ""
    assert info.type_ is int


# --------------------------------------------------------------------
# Argument type merge


def make_arg_traces(*vals):
    return [make_trace(None, v) for v in vals]


def test_arg_float_and_numpy_float_widen_to_real(caplog):
    """float and numpy.float64 arguments, both C doubles, are typed
    double"""
    traces = make_arg_traces(1.5, np.float64(2.5))
    with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
        info = RuntimeCellsInfo(traces)
    assert info.arg_types == {"i": numbers.Real}
    assert info.max_args == {}
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]


@pytest.mark.parametrize("other", [
    np.float32(0.1), np.longdouble(0.1), np.float16(0.5)])
def test_arg_float_and_other_real_types_fall_back_to_object(other, caplog):
    """A real type that is not float computes differently from a double:
    float32(0.1) * 3 is float32(0.3), not 0.30000000447034836"""
    traces = make_arg_traces(1.5, other)
    with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
        info = RuntimeCellsInfo(traces)
    assert info.arg_types == {"i": object}
    msgs = [r.message for r in caplog.records]
    assert len(msgs) == 1
    assert "(typed object: real types that are not all float)" in msgs[0]


def test_arg_int_and_float_fall_back_to_object_and_are_logged(caplog):
    traces = make_arg_traces(1, 2.0)
    with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
        info = RuntimeCellsInfo(traces)
    assert info.arg_types == {"i": object}
    msgs = [r.message for r in caplog.records]
    assert len(msgs) == 1
    assert (f"varying types given to argument 'i' in "
            f"{FOO}: int 1, float 2.0") in msgs[0]
    assert "integral and non-integral" in msgs[0]


@pytest.mark.parametrize("vals", [
    ("a", 1.0),
    (None, 1.0),
    ([1], (1,)),
    (True, np.bool_(False)),
])
def test_arg_other_mixtures_fall_back_to_object_and_are_logged(vals, caplog):
    """Every demotion of an argument to object is logged, not only the
    integral and non-integral mixture"""
    with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
        info = RuntimeCellsInfo(make_arg_traces(*vals))
    assert info.arg_types == {"i": object}
    msgs = [r.message for r in caplog.records
            if "varying types given to argument 'i'" in r.message]
    assert len(msgs) == 1
    assert "(typed object)" in msgs[0]


def test_arg_log_shortens_long_values(caplog):
    with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
        RuntimeCellsInfo(make_arg_traces(np.arange(1000), "x"))
    msg = caplog.records[0].message
    assert "\n" not in msg
    assert "..." in msg
    assert len(msg) < 300


@pytest.mark.parametrize("vals, expected", [
    ((1, np.int64(5), 3), numbers.Integral),
    (("a", np.str_("b")), str),
    ((1.0, 2.0), float),
])
def test_arg_existing_rules_are_kept(vals, expected):
    info = RuntimeCellsInfo(make_arg_traces(*vals))
    assert info.arg_types == {"i": expected}
    if expected is numbers.Integral:
        assert info.max_args == {"i": 5}


# --------------------------------------------------------------------
# MxCodeFilter and the uncached cells


EXPORTED_CLASSES = '''
class _c_Space1:

    def _f_cached(self, t):
        return t

    def uncached(self, x):
        def helper(self):
            return 1
        return x * helper(self) + sum(i for i in range(2)) + (lambda: 0)()

    def cached(self, t):
        if t in self._v_cached:
            return self._v_cached[t]
        else:
            val = self._f_cached(t)
            self._v_cached[t] = val
            return val

    def __call__(self, i):
        return self

    def __init__(self):
        pass

    def _mx_assign_refs(self, io_data, pickle_data):
        pass

    def _mx_copy_refs(self, base, base_root):
        pass


def module_func(self):
    return 1
'''


def compiled_codes(filename):
    """{qualified name: code} of every function in EXPORTED_CLASSES"""
    result = {}

    def walk(code, prefix):
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                qualname = prefix + const.co_name
                result[qualname] = const
                walk(const, qualname + ".")

    walk(compile(EXPORTED_CLASSES, filename, "exec"), "")
    return result


def test_code_filter_admits_uncached_cells_only():
    codes = compiled_codes("C:/model/Model_nomx/_mx_classes.py")
    filt = MxCodeFilter()
    admitted = {k for k, code in codes.items() if filt(code)}
    assert admitted == {
        "_c_Space1._f_cached",
        "_c_Space1.uncached",
        "_c_Space1.__call__",
        "_c_Space1._mx_assign_refs",
    }
    # decided once per code object, and the same again
    assert {k for k, code in codes.items() if filt(code)} == admitted


def test_code_filter_ignores_other_modules():
    codes = compiled_codes("C:/model/Model_nomx/_mx_sys.py")
    filt = MxCodeFilter()
    assert not any(filt(code) for code in codes.values())


# --------------------------------------------------------------------
# MxCallTraceLogger.log keeps only the traces that add something


def test_logger_drops_traces_that_add_nothing():
    logger = MxCallTraceLogger(module="M")
    for i in [0, 1, 1, 0, 3, 2, 3]:
        logger.log(make_trace(1.0, i))
    logger.log(make_trace(1, 2))        # another return type
    logger.log(make_trace(1.0, 2.5))    # another argument type
    logger.log(make_trace(1.0, 2.5))
    kept = logger._traces[FOO]
    assert [(t.arg_vals["i"], t.ret_val) for t in kept] == [
        (0, 1.0), (1, 1.0), (3, 1.0), (2, 1), (2.5, 1.0)]


def test_logger_dedup_gives_the_same_cells_info(caplog):
    """What RuntimeCellsInfo derives is the same with and without the
    dropped traces, including the values its messages quote"""
    rng = np.random.default_rng(0)
    vals = [int(v) for v in rng.integers(0, 50, 500)]
    rets = [[1.0, 2, "a", np.array([1.0])][v % 4] for v in vals]
    vals[100] = 7.5
    traces = [make_trace(r, v) for r, v in zip(rets, vals)]

    logger = MxCallTraceLogger(module="M")
    for tr in traces:
        logger.log(tr)
    kept = logger._traces[FOO]
    assert len(kept) < len(traces)

    def messages(trs):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
            info = RuntimeCellsInfo(trs)
        return info, [r.message for r in caplog.records]

    full, full_msgs = messages(traces)
    dedup, dedup_msgs = messages(kept)
    assert full.arg_types == dedup.arg_types
    assert full.max_args == dedup.max_args
    assert full.ret_type == dedup.ret_type
    assert full_msgs == dedup_msgs


@pytest.mark.parametrize("seq", [
    # (return value, argument)
    [(1.0, 2), (np.array([1.0]), 1), (1.0, 0)],
    # the third trace adds nothing to the arguments and repeats the
    # second's return type, but merges into the object the second made
    [(np.array([1.0]), 5), (1.0, 2), (1.0, 1)],
    [(np.array([1.0]), 5), (np.array([[1.0]]), 2), (np.array([1.0]), 1),
     (1.0, 0), ("a", 0)],
    [(1, 0), (1.0, 0), ("a", 0), (1, 0), (np.array([1]), 0),
     (np.array([1.0]), 0), (np.array(["a"]), 0)],
])
def test_logger_dedup_keeps_every_message(seq, caplog):
    """Whatever the order, the kept traces give what all traces give,
    messages and the values they quote included"""
    traces = [make_trace(r, i) for r, i in seq]
    logger = MxCallTraceLogger(module="M")
    for tr in traces:
        logger.log(tr)
    kept = logger._traces[FOO]

    def derive(trs):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="modelx_cython.tracer"):
            info = RuntimeCellsInfo(trs)
        return (info.arg_types, info.max_args, info.ret_type,
                [r.message for r in caplog.records])

    assert derive(kept) == derive(traces)
