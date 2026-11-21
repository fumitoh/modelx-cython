"""R-3b: storage by use. Above the default dense ceiling (65,536 entries) a multi-argument integer-keyed Cell stays
dense while its rectangle is at most DENSE_USE_FACTOR (16) times the entries one policy fills in the sample; otherwise
it goes to the hash store with a build warning naming the cell, its ranges, the rectangle, the use, the ratio and the
remedies. The build report (_mx_admission.json) states per multi-argument Cell the store chosen and why."""
import json
import subprocess
import sys
import textwrap

import modelx as mx


def _build(tmp_path, spec):
    model = mx.new_model('DenseByUse')
    try:
        s = model.new_space('Policy', formula='lambda key: None')
        s.new_cells('rate', 'lambda t: 0.001 * (t % 13)')
        # well used: a triangle t >= s of a declared 400 x 400 rectangle, every entry filled by each policy
        s.new_cells('used', textwrap.dedent('''\
            def used(t, s):
                if t <= s:
                    return 1.0
                return used(t - 1, s) * (1.0 - rate(t - 1))
            '''))
        # sparse: the same declared rectangle, a single diagonal of 40 entries filled per policy
        s.new_cells('sparse', 'lambda t, s: rate(t) + s')
        s.new_cells('total', textwrap.dedent('''\
            def total():
                a = sum(used(t, s) for t in range(400) for s in range(0, t + 1, 2))
                b = sum(sparse(10 * k, 10 * k) for k in range(40))
                return a + b
            '''))
        model.export(tmp_path / 'exported')
    finally:
        model.close()
    (tmp_path / 'sample.py').write_text('from exported import mx_model\nfor k in (1, 2):\n    mx_model.Policy[k].total()\n')
    (tmp_path / 'spec.py').write_text(repr(spec))
    run = subprocess.run([sys.executable, '-B', '-m', 'modelx_cython', 'exported', '--translate-only'], cwd=tmp_path,
                         check=True, capture_output=True, text=True)
    admission = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text())
    cells = {k.rsplit('._f_', 1)[1]: v for k, v in admission.items() if '._c_Policy._f_' in k}
    return cells, run.stdout + run.stderr


def test_well_used_rectangle_above_the_ceiling_stays_dense(tmp_path):
    cells, log = _build(tmp_path, {'cells_param_size': {'t': 400, 's': 400}})
    used = cells['used']
    assert used['cache'] == 'dense', used['cache_detail']
    detail = used['cache_detail']
    # 160,000 entries > 65,536; one policy fills the 40,200-entry triangle t >= s, s even (ratio 4.0)
    assert detail['entries'] == 160000 and detail['filled_per_policy'] >= 10000, detail
    assert detail['store_reason'].startswith('dense by use: t [0, 400) declared x s [0, 400) declared = 160,000'), detail
    assert 'hash store by use' not in log or 'Policy.used' not in log.split('hash store by use')[0][-200:], log[-3000:]


def test_sparse_rectangle_goes_to_the_hash_store_with_a_warning(tmp_path):
    cells, log = _build(tmp_path, {'cells_param_size': {'t': 400, 's': 400}})
    sparse = cells['sparse']
    assert sparse['cache'] == 'native-numeric-keys', sparse['cache_detail']
    detail = sparse['cache_detail']
    assert detail['filled_per_policy'] == 40 and detail['use_ratio'] == 4000.0, detail
    assert detail['store_reason'].startswith('native-numeric-keys by use:') and '(> 16 x)' in detail['store_reason']
    lines = [line for line in log.splitlines() if 'hash store by use' in line]
    assert len(lines) == 1, log[-3000:]
    line = lines[0]
    for part in ('Policy.sparse (t, s)', 't [0, 400) declared x s [0, 400) declared = 160,000 entries',
                 '4,000.0 x the 40 entries one policy fills in the sample', 'or up to 16 x',
                 're-parametrise the cell if its parameters are tied', 'mark it not cached if each value is read once',
                 'tighten or correct the declaration'):
        assert part in line, (part, line)

