import textwrap

from modelx_cython.parser import ModuleVisitor


SOURCE = textwrap.dedent('''\
    from . import _mx_sys


    class _c_Data(_mx_sys.BaseSpace):

        def __init__(self, parent):
            self._space = self
            self._parent = parent
            self._model = parent._model
            self._name = "Data"
            self.Child = _m_Data._mx_classes._c_Child(self)
            self._mx_spaces = {'Child': self.Child}
            self._mx_roots = []
            # Lock shared by the locked Spaces
            self._mx_lock = self._model._mx_lock
            # Cache variables
            self._v_rate = {}

        def _f_rate(self, t):
            return t

        def rate(self, t):
            if t in self._v_rate:
                return self._v_rate[t]
            with self._mx_lock:
                if t in self._v_rate:
                    return self._v_rate[t]
                val = self._f_rate(t)
                self._v_rate[t] = val
                return val


    class _c_Projection(_mx_sys.BaseSpace):

        def __init__(self, parent):
            self._space = self
            self._parent = parent
            self._model = parent._model
            self._name = "Projection"
            self._mx_roots = []
            self._v_pv = {}

        def _f_pv(self, t):
            return t

        def pv(self, t):
            if t in self._v_pv:
                return self._v_pv[t]
            else:
                val = self._f_pv(t)
                self._v_pv[t] = val
                return val
    ''')


def test_locked_classes():
    visitor = ModuleVisitor(module="M._mx_classes", source=SOURCE)

    assert visitor.locked_classes == {"_c_Data"}
    # the lock is not mistaken for a child space, and nothing else changes
    assert visitor.spaces == {"_c_Data": ["Child"]}
    assert set(visitor.cells_info["_c_Data"]) == {"rate"}
    assert set(visitor.cells_info["_c_Projection"]) == {"pv"}
    assert visitor.closure_funcs == {}
    assert visitor.kwarg_called_names == set()
