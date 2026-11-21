"""Test native provider cache reuse and declaration invalidation."""

import os
from pathlib import Path
import subprocess
import sys

from modelx_cython.build import create_setup


def test_provider_cache_is_relocatable_and_tracks_pxd(tmp_path, monkeypatch):
    cache = tmp_path / 'providers'
    monkeypatch.setenv('MODELX_CYTHON_PROVIDER_CACHE', str(cache))
    path = Path('cached_model/_mx_rate.pyx')

    def build(name, rate):
        root = tmp_path / name
        package = root / 'cached_model'
        package.mkdir(parents=True)
        (package / '__init__.py').touch()
        (package / 'inputs.pxd').write_text(f'cdef enum:\n    RATE = {rate}\n', encoding='utf-8')
        (root / path).write_text('from cached_model.inputs cimport RATE\ndef rate():\n    return RATE\n',
                                 encoding='utf-8')
        create_setup('cached_model', [path], root / 'setup.py', providers=[path])
        result = subprocess.run([sys.executable, '-B', 'setup.py', 'build_ext', '--inplace'],
                                cwd=root, capture_output=True, text=True, check=True)
        runtime_env = dict(os.environ)
        runtime_env.pop('MODELX_CYTHON_PROVIDER_CACHE')
        subprocess.run([sys.executable, '-B', '-c',
                        f'from cached_model._mx_rate import rate; assert rate() == {rate}'],
                       cwd=root, check=True, env=runtime_env)
        return result.stdout

    assert 'Reused native provider' not in build('first', 4)
    assert 'Reused native provider cached_model._mx_rate' in build('second', 4)
    assert 'Reused native provider' not in build('third', 5)
    assert len(list(cache.glob('*.bin'))) == 2
