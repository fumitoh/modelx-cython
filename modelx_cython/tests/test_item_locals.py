"""D110 P1: which item locals are declared their compiled class (properties of item_locals)."""
import textwrap
from types import SimpleNamespace

import libcst as cst

from modelx_cython.item_locals import returns_own_items, type_item_locals

EXPORTED = '''\
class _c_Projection:
    def __call__(self, point_id):
        _mx_key = point_id
        if _mx_key in self._mx_itemspaces:
            return self._mx_itemspaces[_mx_key]
        else:
            _mx_base = self
            _mx_root = _mx_base.__class__(self)
            self._mx_itemspaces[_mx_key] = _mx_root
            return _mx_root

    def __getitem__(self, item):
        return self.__call__(item)
'''


def _class(source):
    return cst.parse_module(source).body[0]


def test_exported_itemspace_returns_own_items():
    assert returns_own_items(_class(EXPORTED))


def test_other_returns_are_not_proven():
    assert not returns_own_items(_class(EXPORTED.replace('return _mx_root', 'return None')))
    assert not returns_own_items(_class(EXPORTED.replace('return self.__call__(item)', 'return item')))
    assert not returns_own_items(_class(EXPORTED.replace(
        '            _mx_base = self\n', '            _mx_base = other\n')))
    stored_elsewhere = EXPORTED + '\n    def put(self, key, value):\n        self._mx_itemspaces[key] = value\n'
    assert not returns_own_items(_class(stored_elsewhere))


def _typed(formula, proven=True, relative=True):
    module = SimpleNamespace(_mx_item_classes={'_c_Projection': proven})
    ref = SimpleNamespace(is_relative=relative, decl_type_expr='_c_Projection')
    cls_info = SimpleNamespace(refs={'projection': ref, 'other': SimpleNamespace(is_relative=True, decl_type_expr='')})
    node = cst.parse_module(textwrap.dedent(formula)).body[0]
    node, typed = type_item_locals(node, cls_info, module)
    return typed, cst.Module(body=(node,)).code


def test_single_binding_is_typed_and_value_unchanged():
    typed, code = _typed('''\
        def f(self):
            total = 0.0
            for key in self.projection.policy_keys():
                item = self.projection[key]
                total += item.g(1)
            return total
        ''')
    assert typed == {'item': '_c_Projection'}
    assert 'self.projection[key]' in code


def test_call_form_is_typed():
    typed, _ = _typed('def f(self, k):\n    item = self.projection(k)\n    return item.g(1)\n')
    assert typed == {'item': '_c_Projection'}


def test_not_typed_when_the_local_can_hold_anything_else():
    forms = [
        'def f(self, k):\n    item = None\n    item = self.projection[k]\n    return item.g(1)\n',
        'def f(self, k):\n    item = self.projection[k]\n    del item\n    return 0\n',
        'def f(self, ks):\n    for item in ks:\n        pass\n    item = self.projection[ks[0]]\n    return item.g(1)\n',
        'def f(self, k):\n    item = self.projection[k]\n    item += 1\n    return item\n',
        'def f(self, item):\n    item = self.projection[item]\n    return item.g(1)\n',
        'def f(self, k):\n    item = self.other[k]\n    return item.g(1)\n',
        'def f(self, k):\n    item = self.projection[k, 1]\n    return item.g(1)\n',
        'def f(self, k):\n    item = self.projection(*k)\n    return item.g(1)\n',
        'def f(self, k):\n    item = self.projection[k]\n    return [item for item in k]\n',
    ]
    for form in forms:
        assert _typed(form)[0] == {}, form


def test_unproven_or_foreign_class_is_not_typed():
    form = 'def f(self, k):\n    item = self.projection[k]\n    return item.g(1)\n'
    assert _typed(form, proven=False)[0] == {}
    assert _typed(form, relative=False)[0] == {}
