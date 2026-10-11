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


# --------------------------------------------------------------------
# Parameters a formula binds again

import libcst as cst
import pytest

from modelx_cython.parser import collect_param_rebinds


def rebinds(body, params=("t", "x")):
    src = f"def _f_f(self, {', '.join(params)}):\n" + "".join(
        "    " + line + "\n" for line in body.splitlines())
    result = collect_param_rebinds(cst.parse_statement(src), params)
    return {p: [cst.Module([]).code_for_node(v)
                if isinstance(v, cst.BaseExpression) else v for v in vals]
            for p, vals in result.items()}


@pytest.mark.parametrize("body, expected", [
    ("return t + x", {}),
    ("t = t - 1\nreturn t", {"t": ["t - 1"]}),
    ("t = x = 2\nreturn t", {"t": ["2"], "x": ["2"]}),
    ("t += 1\nt /= 2\nreturn t", {"t": ["t + 1", "t / 2"]}),
    ("x **= 0.5\nreturn x", {"x": ["x ** 0.5"]}),
    ("t <<= 1\nreturn t", {"t": ["an augmented assignment"]}),
    ("if (t := t * 2) > 3:\n    pass\nreturn t", {"t": ["t * 2"]}),
    ("t: float = 1.0\nreturn t", {"t": ["an annotated assignment"]}),
    ("t, y = 1, 2\nreturn t", {"t": ["unpacking"]}),
    ("[t, *x] = [1, 2]\nreturn t", {"t": ["unpacking"], "x": ["unpacking"]}),
    ("for t in range(3):\n    pass\nreturn t", {"t": ["a for loop"]}),
    ("with open(x) as t:\n    pass\nreturn t", {"t": ["a with statement"]}),
    ("try:\n    pass\nexcept Exception as t:\n    pass\nreturn 1",
     {"t": ["an except clause"]}),
    ("import t\nreturn t", {"t": ["an import"]}),
    ("import os as x\nreturn x", {"x": ["an import"]}),
    ("del t\nreturn 1", {"t": ["a del statement"]}),
    ("def t():\n    return 1\nreturn t()", {"t": ["a nested def"]}),
    # comprehensions bind their own names, conservatively unknown
    ("return [t for t in range(3)]", {"t": ["a comprehension"]}),
    ("return [(t := i) for i in range(3)]", {"t": ["an assignment expression"]}),
    # a nested scope's bindings are its own, unless nonlocal
    ("def g(t):\n    t = 2.5\n    return t\nreturn g(t)", {}),
    ("g = lambda t: t\nreturn g(1)", {}),
    ("def g():\n    nonlocal t\n    t = 2.5\ng()\nreturn t",
     {"t": ["a nonlocal statement"]}),
    # attributes and subscripts bind no name
    ("self.t = 1\nx[0] = 1\nreturn t", {}),
])
def test_collect_param_rebinds(body, expected):
    assert rebinds(body) == expected


def test_visitor_records_rebinds_of_formulas_and_uncached_cells():
    src = '''
class _c_Space1:

    def _f_cached(self, t):
        t = t / 2
        return t

    def cached(self, t):
        return self._f_cached(t)

    def uncached(self, rate, k=0.5):
        rate = rate / 100
        return rate * k
'''
    visitor = ModuleVisitor("Pkg_cy._mx_classes", src)
    got = visitor.param_rebinds["_c_Space1"]
    assert set(got) == {"_f_cached", "uncached"}
    assert list(got["_f_cached"]) == ["t"] and list(got["uncached"]) == ["rate"]
    info = visitor.cells_info["_c_Space1"]["uncached"]
    assert info.params_with_defaults == ["k"]
    assert cst.Module([]).code_for_node(info.param_defaults["k"]) == "0.5"
