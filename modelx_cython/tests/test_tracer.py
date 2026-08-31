import logging

import numpy as np
import pytest

from modelx_cython.tracer import (
    ReturnTypeInfo,
    RuntimeCellsInfo,
    instance_attrs,
)
from modelx_cython.monkeytype_tracing import CallTrace


def foo(self, i):
    pass


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
