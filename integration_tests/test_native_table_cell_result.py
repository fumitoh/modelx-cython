"""Verify native table dispatch through an ordinary parameterized Cell."""

import json
import subprocess
import sys
import textwrap

import modelx as mx
import pandas as pd
from modelx_helpers import pdn, typed_inputs
from modelx_helpers.immutable_inputs import Record, Table


def test_parameterized_table_result(tmp_path):
    model = mx.new_model('ParameterizedTable')
    try:
        space = model.new_space('Policy', formula='lambda policy: None')
        space.tables = typed_inputs
        space.pdn = pdn
        space.schedules = Record({
            2: Table((0, 1), ((2.0,), (5.0,)), ('rate',)),
            7: Table((0, 1, 2), ((11.0,), (13.0,), (17.0,)), ('rate',)),
        })
        space.rows = pdn.table(pd.DataFrame({'rate': [2.0, 5.0]}, index=[4, 9]), rows=True)
        space.schedule_frame = pd.DataFrame({'rate': [2.0, 5.0]}, index=[0, 1])
        index = pd.MultiIndex.from_tuples([('a', 4), ('a', 9), ('b', 3)])
        space.compound_rows = pdn.table(pd.DataFrame({'rate': [2.0, 5.0, 1.0]}, index=index))
        space.new_cells('row_table', '''def row_table() -> "modelx_helpers.immutable_inputs.Table":
    return rows
''')
        space.new_cells('row_rate', 'lambda key: row_table().slice(key)["rate"]')
        space.new_cells('compound_table', '''def compound_table() -> "modelx_helpers.immutable_inputs.Table":
    return compound_rows
''')
        space.new_cells('has_row', 'lambda group, key: (group, key) in compound_table().index')
        space.new_cells('schedule', textwrap.dedent('''\
            def schedule(account) -> "modelx_helpers.immutable_inputs.Table":
                return schedules[account]
            '''))
        space.new_cells('amount', textwrap.dedent('''\
            def amount(account, month):
                """Read the requested account's rate and repeat its last month."""
                table = schedule(account)
                key = min(month, table.index.max_int())
                return table.lookup_float(key, "rate")
            '''))
        space.new_cells('prepared_schedule', 'lambda: pdn.table(schedule_frame)')
        space.new_cells('fixed_schedule', '''def fixed_schedule() -> "modelx_helpers.immutable_inputs.Table":
    return schedules[7]
''')
        space.new_cells('previous_method', 'lambda month: fixed_schedule().previous(month, "rate")')
        space.new_cells('previous_column', 'lambda month, column: fixed_schedule().previous(month, column)')
        space.new_cells('bounded_method', '''def bounded_method(month):
    table = fixed_schedule()
    key = month_key(month)
    lo = int(min(table.index))
    hi = table.index.max_int()
    if key < lo:
        return 2.5
    return table.lookup_float(min(key, hi), "rate")
''')
        space.new_cells('month_key', 'lambda month: month')
        space.new_cells('interpolated_amount', 'lambda account, month: schedule(account).interpolate_float(month, "rate")')
        space.new_cells('policy_row', 'lambda: row_table().slice(4)')
        space.new_cells('float_field', 'lambda t: float(policy_row()["rate"])')
        space.new_cells('raw_field', '''def raw_field(t):
    field = policy_row()["rate"]
    return field
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'from modelx_helpers.immutable_inputs import Record, Table\n'
        's = mx_model.Policy[0]\n'
        's.amount(2, 0); s.amount(7, 1); s.row_rate(4); s.row_rate(9)\n'
        's.has_row("a", 4); s.prepared_schedule()\n'
        '[s.amount(a, m) for a in (2, 7) for m in range(8)]  # the months the probe reads\n'
        's.interpolated_amount(2, 0.5); s.float_field(1); s.raw_field(1)\n'
        's.previous_method(8); s.previous_column(8, -1); s.previous_column(8, 0)\n'
        's.bounded_method(-1); s.bounded_method(8)\n'
        '# Sample keys that the original scenario will reject at the table boundary.\n'
        'alternate = type(mx_model)().Policy[0]\n'
        'past = Table((-1,), ((23.0,),), ("rate",))\n'
        'alternate.schedules = Record({2: past, 7: past, 12: Table((0,), ((1.0,),), ("rate",))})\n'
        'alternate.previous_method(-1); alternate.amount(2, -1); alternate.amount(12, 0)\n',
        encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                   cwd=tmp_path, check=True)
    generated = (tmp_path / 'exported_cy/_mx_classes.py').read_text(encoding='utf-8')
    assert '.bind_native_table(self._f_schedule(account),' in generated
    report = json.loads((tmp_path / 'exported_cy/_mx_admission.json').read_text(encoding='utf-8'))
    helper_reads = report['exported_cy._mx_classes._c_Policy._f_bounded_method']['table_reads_untyped']
    assert helper_reads and all('helper Table' in read['reason'] for read in helper_reads)
    probe = textwrap.dedent('''\
        from exported_cy import mx_model
        s = mx_model.Policy[0]
        assert type(s.schedule(2)).__name__ == 'NativeTable'
        assert list(s.compound_table().index) == [('a', 4), ('a', 9), ('b', 3)]
        # D130: a zero-argument cell whose formula builds a helper Table gives Python that Table, as live;
        # the native table behind it keeps the helper's index API, and the two hold the same rows
        from modelx_helpers import pdn
        from modelx_helpers.immutable_inputs import Table as HelperTable
        table = s.row_table()
        assert isinstance(table, HelperTable) and table is s.row_table()
        native = s._mx_table_cell_value('row_table')
        assert pdn.to_pandas(native).equals(pdn.to_pandas(table))
        assert native.index.max_int() == 9
        assert native.index.max_float() == 9.0
        assert s.schedule(2) is s.schedule(2)
        assert s.schedule(2) is not s.schedule(7)
        from modelx_helpers.immutable_inputs import Table
        from modelx_helpers import pdn
        import numpy as np
        import pandas as pd
        def check(frame, event, arg):
            if event == 'call':
                assert frame.f_code not in (Table.row.__code__, Table.contains.__code__, Table.previous.__code__)
        import sys
        sys.setprofile(check)
        try:
            assert s.row_rate(4) == 2.0 and s.row_rate(9) == 5.0
            assert s.has_row('a', 4) is True and s.has_row('a', 3) is False
            assert s.has_row('b', 3) is True and s.has_row('c', 3) is False
            assert s.bounded_method(-1) == 2.5 and s.bounded_method(8) == 17.0
            assert s.interpolated_amount(2, 0.5) == 3.5
            assert s.interpolated_amount(7, 8.5) == 17.0
            assert s.float_field(1) == s.raw_field(1) == 2.0
        finally:
            sys.setprofile(None)
        owner = s.schedule(2).__modelx_table_owner__()
        assert not owner._native_cache
        assert s.previous_method(8) == 17.0
        assert s.previous_column(8, -1) == s.previous_column(8, 0) == 17.0
        assert not owner._native_cache
        try:
            s.previous_method(-1)
        except ValueError as error:
            assert str(error) == 'No key at or before -1'
        else:
            raise AssertionError('preceding row before the first key')
        try:
            s.row_rate(8)
        except KeyError as error:
            assert error.args == (8,)
        else:
            raise AssertionError('missing row key was accepted')
        for month in range(8):
            assert s.amount(2, month) == (2.0, 5.0)[min(month, 1)]
            assert s.amount(7, month) == (11.0, 13.0, 17.0)[min(month, 2)]
        for account, month in ((2, -1), (12, 0)):
            try:
                s.amount(account, month)
            except KeyError:
                pass
            else:
                raise AssertionError('missing input key was accepted')
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)


def test_native_table_provider_binds_a_pandas_result_outside_the_sample(tmp_path):
    model = mx.new_model('NativeTableFrameResult')
    try:
        data = model.new_space('Data', refs={
            'helper_table': Table((0,), ((2.0,),), ('rate',)),
            'valid_frame': pd.DataFrame({'rate': [7.0]}, index=[0]),
            'invalid_frame': pd.DataFrame({'rate': [3.0, 4.0]}, index=[0, 0]),
        })
        data.new_cells('table', '''def table(key) -> "modelx_helpers.immutable_inputs.Table":
    if key == 1:
        return valid_frame
    if key == 2:
        return invalid_frame
    return helper_table
''')
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text('from exported import mx_model\n'
                                       'mx_model.Data.table(0); mx_model.Data.table(3)\n', encoding='utf-8')
    build = subprocess.run([sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--no-spec'],
                           cwd=tmp_path, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    probe = textwrap.dedent('''\
        from exported_cy import mx_model
        from modelx_cython.runtime.table_admission import TableAdmissionError

        table = mx_model.Data.table
        assert table(0).lookup_float(0, 'rate') == 2.0
        assert table(1).lookup_float(0, 'rate') == 7.0
        try:
            table(2)
        except TableAdmissionError as error:
            assert any('duplicate key' in reason for reason in error.reasons)
        else:
            raise AssertionError('duplicate index accepted')
        ''')
    run = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
