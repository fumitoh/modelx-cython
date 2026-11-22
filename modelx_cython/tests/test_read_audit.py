"""Source matching and model-wide classification for the table-read audit."""
from types import SimpleNamespace

import libcst as cst
import pandas as pd

from modelx_cython import read_notes
from modelx_cython.frame_reads import PROVIDER, FrameReads, audit_subtables, without_fallbacks
from modelx_cython.helpers import module_name
from modelx_cython.usage import CellsResolver


def _table_note(op, observation, count):
    return {'seen': {observation}, 'count': count, 'table_reads': {(op, observation)}, 'table_count': count}


def test_renamed_comprehension_target_matches_sample_notes():
    cell = SimpleNamespace(name='rates', has_args=lambda: True, vector_loop_renames={'_mx_v10_ci': 'i'})
    cls = SimpleNamespace(name='_c_Projection')
    expression = 's.loc[s.index[i], "rate"]'
    observation = ('loc:table:Data.rates', ('num', 'str'), 'float', 'rate')
    resolver = SimpleNamespace(read_notes={('_c_Projection', '_f_rates'): {
        expression: _table_note('sub', observation, 3),
    }})
    reads = FrameReads(resolver, cls, cell, set(), None, builtin_int=True)
    reads.lowered_nodes = [cst.parse_expression(source) for source in (
        's.loc[s.index[_mx_v10_ci], "rate"]', 'other.loc[8, "rate"]',
    )]
    assert reads.audit(cst.parse_statement('return 0.0\n')) == []
    assert reads.unsampled_lowered == 1


def test_subtable_audit_checks_readers_across_spaces():
    def cell(cls, name, **kwargs):
        return SimpleNamespace(fqname=f'm.{cls}._f_{name}', **kwargs)

    native = cell('_c_Projection', 'native', read_audit=[{'per_step': True}, {'per_step': True}])
    mixed = cell('_c_Projection', 'mixed', read_audit=[{'per_step': True}])
    aliased = cell('_c_Projection', 'aliased', read_audit=[{'per_step': True}])
    receiver = cell('_c_Projection', 'receiver', read_audit=[{'per_step': False}, {'per_step': False}])
    calc = cell('_c_Projection', 'calc', read_param_readers={'native', 'mixed', 'aliased'},
                frame_reads=[{'receiver': 'self.receiver()', 'form': 'L5'}])
    projection = {'native': native, 'mixed': mixed, 'aliased': aliased, 'receiver': receiver, 'calc': calc}
    cls = SimpleNamespace(fqname='m._c_Projection', cells=projection)
    cross = SimpleNamespace(fqname='m._c_Cross', cells={})
    modules = {'m': SimpleNamespace(classes={'_c_Projection': cls, '_c_Cross': cross})}
    resolver = CellsResolver({cls.fqname: {name: value.fqname for name, value in projection.items()},
                              cross.fqname: {}},
                             {cross.fqname: {'proj': cls.fqname}}, {})
    provider = module_name(PROVIDER)
    syntax = cst.parse_module(f"""\
class _c_Projection:
    def _f_calc(self, t):
        return 0.0 if {provider}.slot(table, 0) > 0 else self.native(t).loc[t]

class _c_Cross:
    def _f_unlowered(self, t):
        proj = self.proj
        return self.proj.mixed(t) + proj.aliased(t)
""")
    audit_subtables(modules, {'m': SimpleNamespace(syntax=syntax)}, resolver)
    assert all(not entry['per_step'] and 'parameterised sub-table' in entry['reason'] for entry in native.read_audit)
    assert mixed.read_audit == [{'per_step': True}]
    assert aliased.read_audit == [{'per_step': True}]
    assert all(not entry['per_step'] and 'sub-table' in entry['reason'] for entry in receiver.read_audit)


def test_audit_counts_only_pandas_operations_at_an_items_loop_source():
    class _c_Data:
        def series(self):
            return pd.Series([0.5, 0.25], index=[1, 2])

    cls = SimpleNamespace(name='_c_Projection')
    cell = SimpleNamespace(name='rates', has_args=lambda: True)
    site = (cls.name, '_f_rates', 'series.items()')
    with read_notes.recording('unused'):
        series = read_notes.r_call((cls.name, '_f_rates', 'self.data.series()', 'call'), _c_Data(), 'series')
        for _ in range(4):
            source = read_notes.r_call((*site, 'call'), series, 'items')
            assert list(read_notes.r_it((*site, 'it'), source)) == [(1, 0.5), (2, 0.25)]
        assert read_notes.COUNTS[(*site, 'call')] == read_notes.COUNTS[(*site, 'it')] == 4
        resolver = SimpleNamespace(read_notes=read_notes.by_text())
    reads = FrameReads(resolver, cls, cell, set(), None, builtin_int=True)
    native = cst.parse_statement('for key, rate in series.items():\n    pass\n')
    assert [(entry['expression'], entry['evaluations'], entry['pandas_operation'])
            for entry in reads.audit(native)] == [('series.items()', 4, True)]


def test_audit_keeps_the_pandas_operation_evaluated_as_a_native_key():
    cls = SimpleNamespace(name='_c_Projection')
    cell = SimpleNamespace(name='rates', has_args=lambda: True)
    key = 'source.index.get_loc(t)'
    eliminated = 'offsets.index.max()'
    expression = f'table.loc[{key} + {eliminated}, "rate"]'
    resolver = SimpleNamespace(read_notes={(cls.name, '_f_rates'): {
        key: _table_note('call', ('index:table:Data.source', 'get_loc', 'int'), 3),
        eliminated: _table_note('call', ('index:table:Data.offsets', 'max', 'int'), 3),
        expression: _table_note('sub', ('loc:table:Data.rates', ('num', 'str'), 'float', 'rate'), 3),
    }})
    reads = FrameReads(resolver, cls, cell, set(), None, builtin_int=True)
    reads.lowered_nodes = [cst.parse_expression(expression)]
    provider = module_name(PROVIDER)
    compiled = cst.parse_statement(f'return {provider}.f64_1(hold, 0, {key} + limit) '
                                   f'if {provider}.slot(hold, 0) > 0 else {expression}\n')
    assert [entry['expression'] for entry in reads.audit(without_fallbacks(compiled))] == [key]
