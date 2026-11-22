"""The per-formula safety net of table-read lowering (T-5c), with an injected recogniser fault.

The fault raises inside ``FrameReads.lower`` for the formulas whose cell name starts with ``boom``. Without strict
lowering the build succeeds, those formulas are compiled as written (their reads on pandas, nothing held for
them), the build report records "declined after internal error", and every cell still equals live. With strict
lowering the same fault fails the build.
"""
import json
import os
import subprocess
import sys

import test_table_probe as probe

STMTS = {
    'boom_read': ('flat grid', "return float(T.loc[T.index[0], 'q']) + t"),
    'boom_walk': ('flat', "return [k for k in T.index if k > 20 + t]"),
    'kept_read': ('flat', "return T.loc[20 + t, 'q']"),
}
FAULT = '''\
import runpy, sys
import modelx_cython.frame_reads as frame_reads
plain = frame_reads.FrameReads.lower
def lower(self, node):
    if self.cells.name.startswith('boom'):
        raise RuntimeError('injected recogniser fault')
    return plain(self, node)
frame_reads.FrameReads.lower = lower
sys.argv = ['modelx_cython', 'probe_nomx', '--sample', 'sample.py', '--spec', 'spec.py']
runpy.run_module('modelx_cython', run_name='__main__')
'''


def _setup(tmp_path):
    (tmp_path / 'probe_model.py').write_text(
        probe.MODEL.format(ops={}, stmts=STMTS, data=probe.DATA, t5_items={}), encoding='utf-8')
    (tmp_path / 'sample.py').write_text(probe.SAMPLE, encoding='utf-8')
    (tmp_path / 'checks.py').write_text(probe.CHECKS, encoding='utf-8')
    (tmp_path / 'spec.py').write_text(repr({'parallel_workers': 1}), encoding='utf-8')
    (tmp_path / 'fault.py').write_text(FAULT, encoding='utf-8')
    probe._run(tmp_path, 'probe_model.py')


def _build(tmp_path, strict):
    env = dict(os.environ, MX_CY_STRICT_LOWERING='1' if strict else '0')
    return subprocess.run([sys.executable, 'fault.py'], cwd=tmp_path, env=env, timeout=2400,
                          capture_output=True, text=True)


def test_fault_leaves_formula_on_pandas(tmp_path):
    _setup(tmp_path)
    built = _build(tmp_path, strict=False)
    assert built.returncode == 0, built.stderr[-6000:]
    reads = json.loads((tmp_path / 'probe_nomx_cy' / '_mx_table_reads.json').read_text())
    declined = {cell: es for cell, es in reads.items() if isinstance(es, list)
                and any(isinstance(e, dict) and 'declined after internal error' in e.get('reason', '') for e in es)}
    assert sorted(c.split('.')[-1] for c in declined) == ['boom_read__flat', 'boom_read__grid', 'boom_walk__flat']
    assert any(e['form'] == 'L1' for c, es in reads.items() if c.endswith('kept_read__flat') for e in es
               if isinstance(e, dict))
    source = (tmp_path / 'probe_nomx_cy' / '_mx_classes.py').read_text()
    body = source[source.index('def _f_boom_walk__flat'):][:1500]
    assert '_mx_tp_' not in body and 'PreparedRef' not in body       # nothing of the failed lowering
    verdicts = json.loads(probe._run(tmp_path, 'checks.py').stdout)
    assert all(v == 'same' or v.startswith('boundary') for v in verdicts.values()), verdicts


def test_fault_fails_strict_build(tmp_path):
    _setup(tmp_path)
    built = _build(tmp_path, strict=True)
    assert built.returncode != 0 and 'injected recogniser fault' in built.stderr
