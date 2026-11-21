"""D83 (diag-mt): a local bound only to members of one admitted enum, and read only in C-integer
positions, is a C integer bound to the enum's module constants; any other load keeps it an object."""
import enum
from types import SimpleNamespace

import libcst as cst
import pytest

from modelx_cython.enums import Constants


class Kind(enum.IntEnum):
    ANNUITANT = 1
    ALL = 2
    OTHER = 7


class Basis(enum.IntEnum):
    FIRST = 0
    LAST = 1


SYMBOLS = {('_c_P', 'Kind', 'ANNUITANT'): ('_mx_enum_const_0', 1), ('_c_P', 'Kind', 'ALL'): ('_mx_enum_const_1', 2),
           ('_c_P', 'Kind', 'OTHER'): ('_mx_enum_const_2', 7),
           ('_c_P', 'Basis', 'FIRST'): ('_mx_enum_const_3', 0), ('_c_P', 'Basis', 'LAST'): ('_mx_enum_const_4', 1)}
CONSTANTS = {symbol: value for symbol, value in SYMBOLS.values()}


def cell(result='double', params=()):
    return SimpleNamespace(params=params, has_formula_def=True, has_typeinfo=lambda: True,
                           called_with_kwargs=False, body_has_closure=False, formula_allows_exact_receiver=True,
                           get_rettype_expr=lambda **kw: result,
                           get_argtype_expr=lambda name, **kw: 'long long' if name != 'label' else 'object')


def transform(source, params=('k',), result='double'):
    cls = SimpleNamespace(name='_c_P', cells={'pay': cell('double', ('k', 'kind')), 'basis': cell('long long'),
                                              'note': cell('object', ('label',))})
    formula = cell(result, params)
    pass_ = Constants(cls, formula, SYMBOLS)
    return cst.parse_module(source).visit(pass_).code, pass_, formula


class Receiver:
    Kind, Basis = Kind, Basis

    def __init__(self, basis):
        self._basis = basis
        self.calls = []

    def basis(self):
        return self._basis

    def pay(self, k, kind):
        self.calls.append((k, int(kind)))
        return 10.0 * k + int(kind)

    def note(self, label):
        return repr(label)


def run(text, basis, k):
    scope = dict(CONSTANTS)
    exec(text, scope)
    receiver = Receiver(basis)
    return scope['formula'](receiver, k), receiver.calls


IFEXP = ('def formula(self, k):\n'
         '    """doc"""\n'
         '    kind = self.Kind.ANNUITANT if self.basis() == self.Basis.FIRST else self.Kind.ALL\n'
         '    if kind == self.Kind.ALL or kind in (self.Kind.OTHER,):\n'
         '        return self.pay(k, kind) * 2.0\n'
         '    return self.pay(k, kind)\n')


def test_member_local_in_integer_positions_is_a_c_integer():
    text, pass_, formula = transform(IFEXP)
    assert pass_.enum_locals == {'kind': 'Kind'} and formula.enum_locals == ['kind']
    lines = text.splitlines()
    assert lines[1].strip() == '"""doc"""' and lines[2].strip() == 'kind: _mx_cy.longlong', text
    assert ('kind = _mx_enum_const_0 if self.basis() == _mx_enum_const_3 else _mx_enum_const_1' in text), text
    assert 'kind == _mx_enum_const_1' in text and 'kind in (_mx_enum_const_2,)' in text
    assert 'self.Kind' not in text
    # both branches of the conditional expression give the original's values and calls
    for basis in (Basis.FIRST, Basis.LAST):
        for k in (0, 3):
            assert run(text, basis, k) == run(IFEXP, basis, k)


@pytest.mark.parametrize('use', [
    '    return kind\n',                                   # returned (an object result)
    '    return f"{kind}"\n',                              # formatted: the member's text
    '    return kind.name\n',                              # an attribute of the member
    '    return self.note(kind)\n',                        # an object parameter of a Cell
    '    return str(kind)\n',                              # passed to Python
    '    return [self.pay(0, kind) for _ in range(2)]\n',  # a nested scope
    '    return self.pay(0, kind) if kind is self.Kind.ALL else 0.0\n',   # identity
])
def test_member_local_with_an_object_load_stays_an_object(use):
    source = ('def formula(self, k):\n'
              '    kind = self.Kind.ANNUITANT if self.basis() == self.Basis.FIRST else self.Kind.ALL\n'
              + use)
    text, pass_, _ = transform(source, result='object')
    assert pass_.enum_locals == {}
    assert 'kind: _mx_cy.longlong' not in text and 'kind = self._mx_definition_Kind.ANNUITANT if' in text, text   # as before (W1.4 field)


@pytest.mark.parametrize('binding', [
    '    kind = self.Kind.ANNUITANT if self.basis() == 0 else self.Basis.LAST\n',   # two enums
    '    kind = self.Kind.ANNUITANT\n    kind = 2\n',                              # a non-member binding
    '    kind = self.Kind.ANNUITANT\n    for kind in (self.Kind.ALL,):\n        pass\n',  # rebound by a loop
    '    kind: int = self.Kind.ANNUITANT\n',                                       # annotated
])
def test_member_local_needs_member_bindings_of_one_enum(binding):
    source = 'def formula(self, k):\n' + binding + '    return self.pay(k, kind)\n'
    text, pass_, _ = transform(source)
    assert pass_.enum_locals == {} and 'kind: _mx_cy.longlong' not in text, text
