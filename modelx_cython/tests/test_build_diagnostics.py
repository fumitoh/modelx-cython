import subprocess
import sys

import Cython
import modelx as mx
import pytest


def test_narrow_sample_names_cell_and_return_type(tmp_path):
    model = mx.new_model('NarrowDiagnostics')
    try:
        space = model.new_space('Outer').new_space('Projection')
        space.new_cells('half', 'lambda t: t // 2 if t % 2 == 0 else t / 2')
        space.new_cells('total', 'lambda n: sum(half(t) for t in range(0, n, 2))')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\nmx_model.Outer.Projection.total(10)\n',
        encoding='utf-8',
    )
    command = [sys.executable, '-m', 'modelx_cython', 'exported']
    subprocess.run(command + ['--no-spec', '--translate-only'], cwd=tmp_path, check=True)
    result = subprocess.run(command + ['--compile-only'], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "Cython could not compile cell 'Outer.Projection.half'." in result.stderr
    assert "set a wider 'return_type'" in result.stderr
    assert "Cannot assign type" in result.stderr

    (tmp_path / 'spec.py').write_text(repr({'spaces': {'Outer': {'spaces': {'Projection': {
        'cells': {'half': {'return_type': 'object'}}
    }}}}}), encoding='utf-8')
    subprocess.run(command, cwd=tmp_path, check=True)
    subprocess.run([sys.executable, '-c', 'from exported_cy import mx_model; '
                    'assert mx_model.Outer.Projection.half(3) == 1.5'],
                   cwd=tmp_path, check=True)


@pytest.mark.skipif(sys.version_info < (3, 10) or not Cython.__version__.startswith('3.2.'),
                    reason='This failure requires Python match/case syntax and Cython 3.2')
def test_unsupported_match_names_cell_and_cython_version(tmp_path):
    model = mx.new_model('MatchDiagnostics')
    try:
        space = model.new_space('Projection')
        space.new_cells('classify', '''\
def classify(value):
    match value:
        case 1:
            return 10
        case _:
            return 20
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\nassert mx_model.Projection.classify(1) == 10\n',
        encoding='utf-8',
    )
    result = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--no-spec'],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Cython could not compile cell 'Projection.classify'." in result.stderr
    assert 'Using match/case requires Cython 3.3 or later.' in result.stderr
    assert 'return_type' not in result.stderr
