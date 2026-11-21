"""Namespace metadata shares direct-Reference validation and preserves binding."""
from types import ModuleType, SimpleNamespace

import libcst as cst
import pytest

from modelx_cython.helper_members import helper_reference, module_helpers
from modelx_cython.helpers import HelperCalls, helper_capability, module_name
from modelx_cython.tracer import RuntimeValueInfo
from modelx_cython.builder import CombinedRefInfo


def operation(values):
    return all(values)


operation.__modelx_cython_helper__ = ('sample.provider', 'all')
operation.__modelx_cython_scalar_result__ = 'bool'


def test_aliases_share_capabilities_without_freezing_member_callables():
    module = ModuleType('sample.operations')
    module.every = operation
    module.alias = operation
    metadata = module_helpers(module)
    assert metadata['every'].helper == ('sample.provider', 'all')
    assert metadata['every'].scalar_result == 'bool'
    assert metadata['alias'] == metadata['every']
    ref = SimpleNamespace(helper_members=metadata)
    call = cst.parse_expression('self.operations.every')
    assert helper_reference(call, {'operations': ref}) is metadata['every']
    replacement = lambda values: 17
    module.every = replacement
    assert module.every is replacement
    # Emission gets the CURRENT member expression, not a captured callable.
    assert not hasattr(metadata['every'], 'function')
    assert module_helpers(module).keys() == {'alias'}


def test_capture_does_not_execute_module_or_object_getattr():
    seen = []
    module = ModuleType('sample.dynamic')
    module.__getattr__ = lambda name: seen.append(name)
    assert module_helpers(module) == {}

    class Namespace:
        def __getattr__(self, name):
            seen.append(name)
            return operation

    assert module_helpers(Namespace()) == {}
    assert seen == []


def test_invalid_member_capability_is_not_silently_accepted():
    def invalid(values):
        return values
    invalid.__modelx_cython_helper__ = ('sample.provider', 'invalid')
    invalid.__modelx_cython_scalar_result__ = 'invented'
    module = ModuleType('sample.invalid')
    module.invalid = invalid
    with pytest.raises(ValueError, match='invalid helper capability'):
        module_helpers(module)


def test_direct_references_and_foreign_receiver_boundaries():
    direct = SimpleNamespace(helper=('sample.provider', 'all'))
    refs = {'every': direct, 'operations': SimpleNamespace(helper_members={'every': direct})}
    assert helper_reference(cst.parse_expression('self.every'), refs) is direct
    assert helper_reference(cst.parse_expression('other.operations.every'), refs) is None
    assert helper_reference(cst.parse_expression('self.operations.nested.every'), refs) is None


def test_shared_capabilities_reach_namespaces():
    def lookup(*args):
        return args
    lookup.__modelx_cython_helper__ = ('sample.provider', 'lookup')
    lookup.__modelx_cython_scalar_result__ = 'object'
    lookup.__modelx_cython_call_layout__ = 'keywords'
    module = ModuleType('sample.owned')
    module.lookup = lookup
    traced = RuntimeValueInfo(module)
    merged = CombinedRefInfo('exported._mx_classes', '_c_Test', 'operations', traced)
    record = merged.helper_members['lookup']
    assert record.scalar_result == 'object'
    assert record.call_layout == 'keywords'


@pytest.mark.parametrize('result', ['float64', 'int64', 'bool', 'unicode', 'object'])
def test_scalar_result_is_physical_abi_metadata_for_direct_and_module_refs(result):
    def lookup(*args):
        raise AssertionError('metadata capture must not invoke a helper')
    lookup.__modelx_cython_helper__ = ('sample.provider', 'lookup')
    lookup.__modelx_cython_scalar_result__ = result
    direct = CombinedRefInfo('exported._mx_classes', '_c_Test', 'lookup', RuntimeValueInfo(lookup))
    module = ModuleType('sample.typed')
    module.lookup = lookup
    merged = CombinedRefInfo('exported._mx_classes', '_c_Test', 'tables', RuntimeValueInfo(module))
    assert direct.scalar_result == result
    assert merged.helper_members['lookup'].scalar_result == result
    assert not hasattr(merged.helper_members['lookup'], 'function')


@pytest.mark.parametrize('result', ['float', 'double', 'int', 'number', 'Float64', ('float64',)])
def test_scalar_result_rejects_undeclared_physical_types(result):
    def lookup():
        pass
    lookup.__modelx_cython_scalar_result__ = result
    with pytest.raises(ValueError, match='invalid helper capability'):
        helper_capability(lookup, 'scalar_result')


def test_rebound_receiver_and_starred_calls_retain_ordinary_binding():
    ref = SimpleNamespace(helper=('sample.provider', 'lookup'), call_layout='keywords')
    for source in ['def formula(self):\n    self = other\n    return self.lookup(1)\n',
                   'def formula(self):\n    return self.lookup(*items)\n']:
        binder = HelperCalls({'lookup': ref}, set())
        assert cst.Module(body=(binder.rewrite(cst.parse_module(source).body[0]),)).code == source
        assert binder.operations == set()


def test_first_class_cells_use_bound_callables_without_changing_direct_calls():
    source = '''def formula(self):
    return (self.term, self.term(1), lambda self: self.term)
'''
    binder = HelperCalls({}, set(), cells={'term'})
    result = cst.Module(body=(binder.rewrite(cst.parse_module(source).body[0]),)).code
    assert '_mx_cy.cast(object, self).term' in result
    assert 'self.term(1)' in result
    assert 'lambda self: self.term' in result
