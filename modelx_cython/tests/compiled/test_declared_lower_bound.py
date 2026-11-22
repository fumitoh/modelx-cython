"""R-3-lower: a cells_param_size declaration may be a (lower, stop) pair. The declared range is the cell's
logical domain: dense storage covers [lower, stop) (an absolute-valued parameter such as a birth year needs
no storage below lower), an argument below lower raises IndexError like one at or above stop, and a plain
size keeps its meaning [min(0, sampled minimum), size)."""
import json
import subprocess
import sys
import textwrap

import modelx as mx


def _build(tmp_path, spec, output=None):
    model = mx.new_model('DeclaredLower')
    try:
        s = model.new_space('Policy', formula='lambda key: None')
        s.new_cells('q', 'lambda gen, x: gen * 0.001 + x * 0.01')
        s.new_cells('q_year', 'lambda gen: gen * 0.5')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\ns = mx_model.Policy[1]\n'
        'for g in range(1950, 1961):\n    s.q_year(g)\n    for x in range(10, 20):\n        s.q(g, x)\n')
    (tmp_path / 'spec.py').write_text(repr(spec))
    run = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported'], cwd=tmp_path, check=True,
                         capture_output=True, text=True)
    if output is not None:
        output.append(run.stdout + run.stderr)
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    return {k.rsplit('._f_', 1)[1]: v for k, v in admission.items() if '._c_Policy._f_' in k}


def test_lower_stop_pair_is_the_dense_domain(tmp_path):
    cells = _build(tmp_path, {'spaces': {'Policy': {'cells_param_size': {'gen': (1940, 1981), 'x': 155}}}})
    assert cells['q']['dense_logical_domain'] == [[1940, 1981], [0, 155]]
    assert cells['q_year']['dense_logical_domain'] == [[1940, 1981]]
    assert cells['q']['cache'].startswith('dense')
    (tmp_path / 'probe.py').write_text(textwrap.dedent('''\
        from exported_cy import mx_model
        s = mx_model.Policy[2]
        assert s.q(1955, 15) == 1955 * 0.001 + 15 * 0.01
        assert s.q(1940, 0) == 1940 * 0.001 and s.q(1980, 154) == 1980 * 0.001 + 154 * 0.01   # declared, not sampled
        assert s.q_year(1940) == 970.0
        for args in ((1939, 35), (1981, 35), (1955, 155)):
            try:
                s.q(*args)
            except IndexError as error:
                assert 'Policy.q' in str(error) and 'gen in [1940, 1981)' in str(error) and 'cells_param_size' in str(error)
            else:
                raise AssertionError(f'q{args} outside the declared range did not raise')
        try:
            s.q_year(1939)
        except IndexError as error:
            assert 'gen in [1940, 1981)' in str(error)
        else:
            raise AssertionError('q_year below the declared lower bound did not raise')
        '''))
    subprocess.run([sys.executable, '-B', 'probe.py'], cwd=tmp_path, check=True)


def test_plain_size_keeps_its_meaning(tmp_path):
    # x 20 keeps the product (1981 x 20) under the dense ceiling, so the logical domain is reported
    cells = _build(tmp_path, {'spaces': {'Policy': {'cells_param_size': {'gen': 1981, 'x': 20}}}})
    assert cells['q']['dense_logical_domain'] == [[0, 1981], [0, 20]]


def test_sampled_minimum_below_the_declared_lower_replaces_it(tmp_path):
    output = []
    cells = _build(tmp_path, {'spaces': {'Policy': {'cells_param_size': {'gen': (1955, 1981), 'x': 155}}}}, output)
    assert cells['q']['dense_logical_domain'] == [[1950, 1981], [0, 155]]
    # the build says so, one line per cell and parameter
    lines = [line for line in output[0].splitlines() if 'beyond the declared lower' in line]
    assert len([line for line in lines if 'Policy.q (gen, x): parameter gen declared range [1955, 1981), sampled 1950 '
                'beyond the declared lower, range used [1950, 1981)' in line]) == 1, output[0]
    assert len([line for line in lines if 'Policy.q_year (gen): parameter gen declared range [1955, 1981), sampled 1950'
                in line]) == 1, output[0]


def test_sampled_maximum_above_the_declared_stop_replaces_it(tmp_path):
    output = []
    cells = _build(tmp_path, {'spaces': {'Policy': {'cells_param_size': {'gen': (1940, 1981), 'x': 15}}}}, output)
    assert cells['q']['dense_logical_domain'] == [[1940, 1981], [0, 20]]
    lines = [line for line in output[0].splitlines() if 'beyond the declared stop' in line]
    assert len(lines) == 1 and ('Policy.q (gen, x): parameter x declared range [0, 15), sampled 19 beyond the declared stop, '
                                'range used [0, 20)' in lines[0]), output[0]
