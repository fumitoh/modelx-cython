"""cells_param_size inherited from the model and enclosing Spaces, declared physical size,
param_bound_checks at collect entry, enum domains."""
import json
import os
import subprocess
import sys
import textwrap
from http import HTTPStatus

import pytest
import modelx as mx


def _build(tmp_path, spec):
    model = mx.new_model('ParamBounds')
    try:
        keys = (1, 2, 3)
        p = model.new_space('Projection', formula='lambda policy_id: None',
                            refs={'keys': keys, 'Status': HTTPStatus},
                            collect_over={'policy_id': keys},
                            collect_outputs='lambda: {"sum": {"total": total()}}')
        p.new_cells('proj_len', 'lambda: 4 + 3 * policy_id if policy_id < 9 else 1 // 0')
        p.new_cells('cf', 'lambda t: 1.0 * t + policy_id')
        p.new_cells('pair', 'lambda t, k: cf(t) + k')
        p.new_cells('total', 'lambda: sum(cf(t) for t in range(proj_len())) + pair(0, 0)')
        p.new_cells('rate', textwrap.dedent('''\
            def rate(s):
                return 0.5 * int(s)
            '''))
        p.new_cells('status', 'lambda: Status.OK')
        p.new_cells('rated', 'lambda: rate(status())')
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text(textwrap.dedent('''\
        from exported import mx_model as m
        m.Projection[1].total()
        m.Projection[1].rated()
        '''))
    (tmp_path / 'spec.py').write_text(repr(spec))
    build = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported',
                            '--sample', 'sample.py', '--spec', 'spec.py'],
                           cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True)
    (tmp_path / 'build.log').write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    return json.loads((tmp_path / 'exported_cy' / '_mx_admission.json').read_text())


def _cell(report, name):
    return next(v for k, v in report.items() if k.endswith('._f_' + name))


def _domain(entry):
    return entry['dense_logical_domain'] or [entry['associative_logical_domain'][k]
                                             for k in entry['arguments']]


@pytest.mark.parametrize('layer', ['model', 'space'])
def test_param_bounds_inherit_and_collect_check(tmp_path, layer):
    spaces = {'Projection': {'serial_collect': 'reuse', 'param_bound_checks': {'t': 'proj_len'}}}
    if layer == 'space':
        # the Space overrides the model for t; an exact cells_param_size entry wins for its axes
        spaces['Projection'].update(cells_param_size={'t': 20, ('t', 'k'): (14, 1)})
    report = _build(tmp_path, {'parallel_workers': 1, 'cells_param_size': {'t': 11}, 'spaces': spaces})
    cf, pair, check = _cell(report, 'cf'), _cell(report, 'pair'), _cell(report, 'proj_len')
    stop = 11 if layer == 'model' else 20
    assert _domain(cf) == [[0, stop]]
    assert _domain(pair) == ([[0, 11], [0, 1]] if layer == 'model' else [[0, 14], [0, 1]])
    assert cf['declared_sizes'] == {'t': 'model' if layer == 'model' else 'Projection'}
    # a declared size is allocated as declared (no traced window with a spill path)
    assert cf['dense_window'] == [[0, stop]] and not cf['exact_spill']
    # traced: policy 1, proj_len 7: cf t < 7 (margin 0), pair t = 0 only (margin -6), so the
    # accepted proj_len is min(stop of cf, stop of pair + 6)
    limit = 11 if layer == 'model' else 20
    assert [c['limit'] for c in check['param_bound_checks']] == [limit]
    probe = textwrap.dedent('''\
        import sys
        from exported_cy import mx_model as m
        limit = int(sys.argv[1])
        p = m.Projection
        # a policy beyond the traced horizon but inside the declared bound
        assert p[2].cf(9) == 11.0 and p[2].total() == sum(t + 2.0 for t in range(10)) + 2.0
        try:
            p[3].cf(limit)
        except IndexError as error:
            assert 'Projection.cf' in str(error) and f't in [0, {limit})' in str(error), error
        else:
            raise AssertionError('index outside the bound was evaluated')
        workers, evaluate, _, _ = p._mx_collect_options_v2()
        expect = sum(sum(t + 1.0 * k for t in range(4 + 3 * k)) + k for k in (1, 2))
        assert evaluate((1, 2)).sum.total == expect
        if limit < 13:
            try:
                evaluate((1, 2, 3))
            except IndexError as error:
                text = str(error)
                assert 'policy 3' in text and 'proj_len() = 13' in text and 'cells_param_size' in text, text
                assert f'accepts proj_len() <= {limit}' in text, text
            else:
                raise AssertionError('the bound check did not fire')
        else:
            assert evaluate((1, 2, 3)).sum.total == expect + sum(t + 3.0 for t in range(13)) + 3
        try:
            evaluate((1, 9))
        except RuntimeError as error:
            assert 'proj_len() raised for policy 9' in str(error), error
        else:
            raise AssertionError('a raising check Cell was not reported')
        # the check leaves no state behind: a clean run after a failed one
        assert evaluate((1, 2)).sum.total == expect
        ''')
    result = subprocess.run([sys.executable, '-B', '-c', probe, str(limit)], cwd=tmp_path,
                            env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_enum_argument_domain_is_the_enum(tmp_path):
    report = _build(tmp_path, {'parallel_workers': 1,
                               'spaces': {'Projection': {'serial_collect': 'reuse'}}})
    rate = _cell(report, 'rate')
    top = max(int(member) for member in HTTPStatus)
    assert _domain(rate) == [[0, top + 1]], rate
    assert 'HTTPStatus' in rate['enum_domains']['s']
    probe = textwrap.dedent('''\
        from http import HTTPStatus
        from exported_cy import mx_model as m
        p = m.Projection[2]
        # the sample passed OK only; an unused member works as it does live
        for member in (HTTPStatus.OK, HTTPStatus.NOT_FOUND, max(HTTPStatus)):
            assert p.rate(member) == 0.5 * int(member)
        ''')
    result = subprocess.run([sys.executable, '-B', '-c', probe], cwd=tmp_path,
                            env=dict(os.environ), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
