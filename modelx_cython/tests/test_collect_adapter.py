"""The compiler uses the current exporter protocol without rewriting it."""
from types import SimpleNamespace

import libcst as cst
import pytest

from modelx_cython.collect_adapter import CollectRuntime
from modelx_cython.helpers import read_resource


def current():
    return read_resource('modelx.export', '_mx_sys.py').decode('utf-8')


def module(source, options=None):
    return SimpleNamespace(
        classes={'Projection': SimpleNamespace(fqname='Projection')},
        spec=SimpleNamespace(get_spec=lambda _: options or {}),
        visitor=SimpleNamespace(wrapper=SimpleNamespace(module=cst.parse_module(source))),
    )


def test_current_exporter_runtime_is_kept_verbatim():
    source = current()
    runtime = CollectRuntime(source)
    assert runtime.hook
    runtime.require_current((module('pass\n'),), workers=4)
    assert runtime.syntax.code == source


@pytest.mark.parametrize('marker', ['', '_mx_collect_adapter_version = 1', '_mx_collect_adapter_version = 17'])
def test_obsolete_export_requires_reexport(marker):
    # one check of the export's protocol, whatever the model uses (an ordinary collect method
    # used to pass with an old export)
    runtime = CollectRuntime(current().replace('_mx_collect_adapter_version = 2', marker))
    assert not runtime.hook
    for source in ('def collect(self): return 42\n',
                   'def collect(self): return _mx_sys.collect_from_space(self)\n'):
        with pytest.raises(ValueError, match='export protocol 2.*Re-export the model'):
            runtime.require_current((module(source),), workers=1)


@pytest.mark.parametrize('options,workers', [({'serial_collect': 'reuse'}, 1), ({}, 4), ({}, 1)])
def test_a_runtime_without_the_protocol_requires_a_current_export(options, workers):
    runtime = CollectRuntime('pass\n')
    with pytest.raises(ValueError, match='Re-export the model'):
        runtime.require_current((module('pass\n', options),), workers)


def test_duplicate_current_hook_does_not_enable_native_collection():
    runtime = CollectRuntime(current() + '\ndef _mx_collect_chunk(*args): pass\n')
    assert not runtime.hook


def test_current_version_still_requires_the_worker_hook():
    class WithoutWorkers(cst.CSTTransformer):
        def leave_FunctionDef(self, original, updated):
            if updated.name.value != '_mx_collect_parallel':
                return updated
            params = [param for param in updated.params.params if param.name.value != 'workers']
            return updated.with_changes(params=updated.params.with_changes(params=params))

    source = cst.parse_module(current()).visit(WithoutWorkers()).code
    assert source != current()
    assert not CollectRuntime(source).hook
