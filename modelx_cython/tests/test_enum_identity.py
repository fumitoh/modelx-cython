"""D98: ``is`` / ``is not`` between an admitted member and a typed integer holding a value of that
enum lower as ``==`` / ``!=``; any other identity test is left to Python."""
import enum
from types import SimpleNamespace

import libcst as cst
import pytest

from modelx_cython.enums import Constants


class Kind(enum.IntEnum):
    A = 1
    B = 2


class Basis(enum.IntEnum):
    FIRST = 1
    LAST = 2


SYMBOLS = {('_c_P', 'Kind', 'A'): ('_mx_enum_const_0', 1), ('_c_P', 'Kind', 'B'): ('_mx_enum_const_1', 2),
           ('_c_P', 'Basis', 'FIRST'): ('_mx_enum_const_2', 1), ('_c_P', 'Basis', 'LAST'): ('_mx_enum_const_3', 2)}


def ref(cls):
    return SimpleNamespace(enum=(cls.__module__, cls.__qualname__, {m.name: int(m) for m in cls}), enum_safe=True)


def cell(ret, args=None, params=()):
    rt = SimpleNamespace(ret_type=SimpleNamespace(value_type=ret, is_array=False), arg_types=args or {})
    return SimpleNamespace(params=params, _rt=rt, _spec_ret_t=None, has_typeinfo=lambda: True,
                           called_with_kwargs=False, body_has_closure=False,
                           get_rettype_expr=lambda **kw: 'long long' if issubclass(ret, int) else 'double',
                           get_argtype_expr=lambda name, **kw: 'long long')


def transform(source):
    formula = cell(float, {'k': Kind, 'n': int}, ('k', 'n'))
    cls = SimpleNamespace(name='_c_P', refs={'Kind': ref(Kind), 'Basis': ref(Basis)},
                          cells={'kind': cell(Kind), 'count': cell(int), 'basis': cell(Basis)})
    return cst.parse_module(source).visit(Constants(cls, formula, SYMBOLS)).code


@pytest.mark.parametrize('source, expected', [
    ('if self.kind(t) is self.Kind.A: pass', 'self.kind(t) == _mx_enum_const_0'),
    ('if self.kind(t) is not self.Kind.B: pass', 'self.kind(t) != _mx_enum_const_1'),
    ('if self.Kind.A is self.kind(t): pass', '_mx_enum_const_0 == self.kind(t)'),
    ('if self.Kind.B is not self.kind(t): pass', '_mx_enum_const_1 != self.kind(t)'),
    ('if k is self.Kind.A: pass', 'k == _mx_enum_const_0'),                 # parameter traced as Kind
    ('if self.Kind.B is not k: pass', '_mx_enum_const_1 != k'),
    ('x = self.kind(t)\nif x is self.Kind.B: pass', 'x == _mx_enum_const_1'),     # local of a Kind cell
])
def test_identity_with_a_value_of_the_enum_is_equality(source, expected):
    code = transform('def f(self, k, n):\n    ' + source.replace('\n', '\n    ') + '\n')
    assert expected in code and ' is ' not in code, code


@pytest.mark.parametrize('source', [
    'if 1 is self.Kind.A: pass',                    # a literal: Python says False, == would say True
    'if n is self.Kind.A: pass',                    # a plain int parameter
    'if self.count(t) is self.Kind.A: pass',        # an int-valued Cell
    'if self.basis(t) is self.Kind.A: pass',        # a value of another enum
    'if (k + 0) is self.Kind.A: pass',              # arithmetic
    'x = self.kind(t)\nx = 3\nif x is self.Kind.A: pass',       # a local also bound to an int
    'if label is self.Kind.A: pass',                # an untyped name
])
def test_other_identity_tests_stay_python(source):
    code = transform('def f(self, k, n):\n    ' + source.replace('\n', '\n    ') + '\n')
    assert ' is ' in code and '_mx_enum_const_0' not in code.split(' is ')[1].split(':')[0], code
