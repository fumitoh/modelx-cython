"""Verify the scalar provider and its worker-owned rounding memo directly."""

import importlib
from pathlib import Path
import pickle
import struct
import subprocess
import sys

import pytest

from modelx_cython import runtime
from modelx_cython.helpers import install, module_name
from modelx_cython.rounding import NATIVE_ROUNDING_PROVIDER


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp('native-rounding')
    package = root / 'rounding_package'
    package.mkdir()
    (package / '__init__.py').touch()
    modules = install({NATIVE_ROUNDING_PROVIDER}, package)
    alias = module_name(NATIVE_ROUNDING_PROVIDER)
    (root / 'round_probe.pyx').write_text(f'''\
from rounding_package cimport {alias} as scalar

def context(bint worker=False):
    cdef scalar.LookupContext result = scalar.LookupContext()
    if worker:
        result._mx_enable_private_lookup()
    return result

def memo_ready(scalar.LookupContext context):
    return bool(context.round_ready)

def repeated(scalar.LookupContext context, function, values):
    return [scalar.repeated_round_float(context, function, value, digits)
            for value, digits in values]
''', encoding='utf-8')
    include = Path(runtime.__file__).resolve().parents[2]
    sources = [*(path.as_posix() for path in modules), 'round_probe.pyx']
    (root / 'setup.py').write_text(f'''\
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize({sources!r}, include_path=[{str(include)!r}],
    compiler_directives={{'language_level': 3, 'freethreading_compatible': True}}))
''', encoding='utf-8')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                   cwd=root, check=True)
    sys.path.insert(0, str(root))
    try:
        yield importlib.import_module('round_probe')
    finally:
        sys.path.remove(str(root))
        sys.modules.pop('round_probe', None)
        sys.modules.pop('rounding_package', None)
        sys.modules.pop(f'rounding_package.{alias}', None)


def test_rounding_preserves_bits_and_only_memoizes_for_workers(native):
    values = [(2.675, 2), (2.675, 2), (2.675, 1), (-0.0, 2), (0.0, 2),
              (-0.0, 2), (125.0, -1), (125.0, -1), (2.5, 1 << 62)]
    for worker in (False, True):
        context = native.context(worker)
        for actual, (value, digits) in zip(native.repeated(context, round, values), values):
            assert struct.pack('d', actual) == struct.pack('d', round(value, digits))
        assert native.memo_ready(context) is worker
        restored = pickle.loads(pickle.dumps(context))
        assert not native.memo_ready(restored)
        native.repeated(restored, round, values)
        assert not native.memo_ready(restored)


def test_replacement_callable_does_not_reuse_the_canonical_memo(native):
    context = native.context(True)
    values = [(2.675, 2)] * 2
    assert native.repeated(context, round, values) == [round(2.675, 2)] * 2
    calls = []

    def replacement(value, digits):
        calls.append((value, digits))
        return 9.5

    assert native.repeated(context, replacement, values) == [9.5, 9.5]
    assert calls == values


def test_replacement_present_at_provider_import_remains_a_python_call(native):
    probe = """
import builtins

calls = []
def replacement(value, digits):
    calls.append((value, digits))
    return 9.5

builtins.round = replacement
import round_probe
context = round_probe.context(True)
values = [(2.675, 2)] * 2
assert round_probe.repeated(context, builtins.round, values) == [9.5, 9.5]
assert calls == values
assert not round_probe.memo_ready(context)
"""
    subprocess.run([sys.executable, '-c', probe],
                   cwd=Path(native.__file__).parent, check=True)
