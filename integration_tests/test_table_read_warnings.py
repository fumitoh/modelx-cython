"""A table read left on pandas in a per-step cell (a cell with arguments) is a compiler warning at build time, and
is in the read audit (``_mx_read_audit.json``) with its reason; a per-policy one is in the audit only."""
import json
import subprocess
import sys
import textwrap

MODEL = '''\
import modelx as mx
import pandas as pd

FRAME = pd.DataFrame({"sa": [100.0, 200.0, 300.0, 400.0], "rate": [0.01, 0.02, 0.03, 0.04]},
                     index=pd.Index([1, 2, 3, 4], name="policy_id"))


def build():
    model = mx.new_model("Warn")
    data = model.new_space("Data")
    data.raw = FRAME
    data.new_cells("points", "def points():\\n    return raw.copy()\\n")
    proj = model.new_space("Projection", refs={"data": data})
    proj.new_cells("masked", "def masked(t):\\n    T = data.points()\\n"
                             "    return float(T.loc[T['sa'] > 100.0 * t, 'rate'].sum())\\n")
    proj.new_cells("once", "def once():\\n    T = data.points()\\n"
                           "    return float(T.loc[T['sa'] > 150.0, 'rate'].sum())\\n")
    proj.new_cells("nested_key", "def nested_key(t):\\n    T = data.points()\\n"
                                 "    return T.loc[T.index.get_loc(t) + 1, 'rate']\\n")
    return model
'''


def test_mask_read_in_a_per_step_cell_warns(tmp_path):
    (tmp_path / 'warn_model.py').write_text(MODEL, encoding='utf-8')
    (tmp_path / 'build_model.py').write_text(textwrap.dedent('''\
        from warn_model import build
        m = build()
        m.export("warn_nomx", locked_spaces=["Data"])
        m.close()
        '''), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(
        'from warn_nomx import mx_model\nfor t in range(3):\n    mx_model.Projection.masked(t)\n'
        '    mx_model.Projection.nested_key(t + 1)\n'
        'mx_model.Projection.once()\n', encoding='utf-8')

    def run(*args):
        done = subprocess.run([sys.executable, *args], cwd=tmp_path, timeout=1200, capture_output=True, text=True)
        assert done.returncode == 0, f'{args}\n' + done.stdout[-4000:] + done.stderr[-8000:]
        return done
    run('build_model.py')
    done = run('-m', 'modelx_cython', 'warn_nomx', '--sample', 'sample.py', '--no-spec', '--translate-only')
    log = done.stdout + done.stderr
    warnings = [line for line in log.splitlines() if 'table read left on pandas in a per-step cell' in line]
    assert len(warnings) == 1 and 'masked' in warnings[0] and "T['sa'] > 100.0 * t" in warnings[0], log[-4000:]
    compiled = tmp_path / 'warn_nomx_cy'
    audit = json.loads((compiled / '_mx_read_audit.json').read_text(encoding='utf-8'))
    lowered = json.loads((compiled / '_mx_table_reads.json').read_text(encoding='utf-8'))
    entries = {cell.rsplit('.', 1)[-1]: es for cell, es in audit.items() if cell != '_summary'}
    for cell, per_step in [('masked', True), ('once', False)]:
        reads = [entry for entry in entries[cell] if not entry.get('pandas_operation')]
        assert [entry['per_step'] for entry in reads] == [per_step], audit
        operations = [entry for entry in entries[cell] if entry.get('pandas_operation')]
        assert [(entry['expression'], entry['per_step']) for entry in operations] == [("T['sa']", per_step)], audit
    assert audit['_summary']['left_on_pandas_per_step'] == 1, audit
    assert audit['_summary']['pandas_operations_per_step'] == 2, audit
    assert audit['_summary']['pandas_operations_per_policy'] == 1, audit
    assert [read['form'] for cell, reads in lowered.items() if cell.endswith('.nested_key') for read in reads] == ['L1']
    assert [(entry['expression'], entry['evaluations'], entry.get('pandas_operation'))
            for entry in entries['nested_key']] == [('T.index.get_loc(t)', 3, True)]
