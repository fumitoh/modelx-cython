"""Native unit-step range counters retain Python's bounds and binding semantics."""

import subprocess
import sys
import textwrap

import modelx as mx


def test_checked_native_range(tmp_path):
    model = mx.new_model('CheckedNativeRange')
    try:
        values = model.new_space('Values')
        # Uncached Cells admit signed arguments across the full native range.
        values.new_cells('term', 'lambda index: float(index)', is_cached=False)
        values.new_cells('step', 'lambda: 3')
        values.new_cells('add', textwrap.dedent('''\
            def add(origin, stop):
                total: float = 0.0
                for index in range(origin + 1, stop):
                    total += term(index) * ((index - origin) / 12.0)
                return total
            '''), is_cached=False)
        values.new_cells('add_from_one', textwrap.dedent('''\
            def add_from_one(stop):
                total: float = 0.0
                for slot in range(1, stop):
                    total += term(slot)
                return total
            '''), is_cached=False)
        values.new_cells('weighted', textwrap.dedent('''\
            def weighted(origin, stop, limit):
                step_size = step()
                total: float = 0.0
                for number in range(1, stop):
                    month = origin + (number - 1) * step_size
                    if number * step_size <= limit:
                        total += term(month) * ((month - origin) / 12.0)
                return total
            '''), is_cached=False)
        values.new_cells('square', textwrap.dedent('''\
            def square(origin, stop):
                total = 0.0
                for squared_index in range(origin + 1, stop):
                    total += float(squared_index * squared_index)
                return total
            '''), is_cached=False)
        values.new_cells('shadowed', textwrap.dedent('''\
            def shadowed(origin, stop):
                range = lambda a, b: (a,)
                total = 0.0
                for shadowed_index in range(origin + 1, stop):
                    total += term(shadowed_index)
                return total
            '''), is_cached=False)
        model.export(tmp_path / 'exported')
    finally:
        model.close()

    (tmp_path / 'sample.py').write_text(
        'from exported import mx_model\n'
        'v = mx_model.Values\n'
        'v.add(0, 5); v.add_from_one(5); v.weighted(0, 5, 5.5); v.square(0, 5); v.shadowed(0, 5)\n',
        encoding='utf-8')
    # the checked mode (owner's rule): the small sample would resolve the unchecked mode. The typed
    # range counters of the removed range proofs are no longer asserted; the values below are.
    (tmp_path / 'spec.py').write_text(repr({'integer_checks': 'checked'}), encoding='utf-8')
    subprocess.run(
        [sys.executable, '-m', 'modelx_cython', 'exported', '--sample', 'sample.py', '--spec', 'spec.py'],
        cwd=tmp_path, check=True)
    probe = textwrap.dedent('''\
        from exported_cy import mx_model
        v = mx_model.Values
        bound = (1 << 63) - 1
        for origin, stop in ((-7, 8), (4, 4), (-bound - 1, -bound + 2)):
            expected = 0.0
            for index in range(origin + 1, stop):
                expected += float(index) * ((index - origin) / 12.0)
            assert v.add(origin, stop) == expected
        # an intermediate beyond the signed range (origin + 1, index * index): the exact fallback of
        # the removed range proofs is gone, it raises naming the Cell
        for name, call in (('add', lambda: v.add(bound, bound)), ('square', lambda: v.square(bound - 2, bound))):
            try:
                call()
            except OverflowError as error:
                assert str(error).startswith('Values.' + name), error
            else:
                raise AssertionError(name + ': an intermediate beyond the signed range did not raise')
        assert v.add_from_one(-bound) == 0.0 and v.add_from_one(5) == 10.0
        for origin in (0, -(1 << 63), (1 << 63) - 16):
            expected = sum(float(origin + (k - 1) * 3) * ((k - 1) * 3 / 12.0)
                           for k in range(1, 5) if k * 3 <= 11.5)
            assert v.weighted(origin, 5, 11.5) == expected
        assert v.shadowed(5, 5) == 6.0
        ''')
    subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, check=True)
