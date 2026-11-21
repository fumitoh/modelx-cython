"""Square cases contributed by W1.2: {(form, operation, key): case}. Owned by W1.2.

A case is ``case(mode, tmp_path_factory)`` that runs the square in ``mode`` ('live',
'exported' or 'compiled') and asserts the pandas result (oracle) and, in compiled, that no
Python fallback ran. One session build serves every square (table_matrix_w1_2_support.Build:
a live model, its export and one compiled model with the handle driver).

Every applicable square of the runtime half is a case: the reads of a square run on every
storage variant of its form (W1.H's 13 forms), with the key type of the square on the int
parts of the key, hits and misses; compiled reads call the handles as W1.3 emits them.
"""
from table_matrix_squares import applicable
import table_matrix_cases_w1_3
import table_matrix_w1_2_support as support
from table_matrix_w1_2_support import VARIANTS, int_index, keys, levels

_BUILD = []


def build(tmp_path_factory):
    if not _BUILD:
        _BUILD.append(support.Build(tmp_path_factory.mktemp('w12matrix')))
    return _BUILD[0]


def _reads(form, operation, kt):
    """[(read, variant, key type, args)] of one square."""
    reads = []
    for variant in VARIANTS[form]:
        hits, misses = keys(variant, kt)
        c = 'q'
        single_int = int_index(variant)
        frame = support.frame_of(variant)
        n = levels(variant)
        add = lambda read, *args: reads.append((read, variant, kt, args))  # noqa: E731
        if operation == 1:
            for key in hits + misses:
                add('loc', key, c)
        elif operation == 2:
            for key in hits[:2]:
                add('at', key, c)
        elif operation == 3:
            for column in ('n', 'flag', 'e', 'q'):
                if column in frame.columns:
                    for key in hits[:2]:
                        add('int', key, column)
                        add('obj', key, column)
        elif operation == 4:
            for key in hits + misses:
                add('has', key)
                for depth in range(1, n):
                    add('has', key[:depth])
        elif operation == 5:
            add('minmax', 0)
            if single_int:
                lo, hi = int(frame.index.min()), int(frame.index.max())
                for x in (lo - 5, lo, (lo + hi) // 2, hi, hi + 5):
                    add('clamp', x)
                    add('clamp_read', x, c)
        elif operation == 6:
            for key in hits + misses:
                for depth in range(1, n):
                    add('sub_loc', key[:depth], key[depth:], c)
                    add('sub_has', key[:depth], key[depth:], c)
                    add('sub_obj', key[:depth], key[depth:], c)
        elif operation == 7:
            if len(hits) >= 2:
                add('rows', hits[0], hits[-1], c)
                add('rows', hits[-1], hits[0], c)
            if hits:
                add('rows', hits[0], hits[0], c)
        elif operation == 8 and single_int:
            lo, hi = int(frame.index.min()), int(frame.index.max())
            for x in (lo - 1, lo, lo + 1, (lo + hi) // 2, hi, hi + 3):
                add('asof', x)
                add('asof_read', x, c)
                add('asof_arith', x)
        elif operation == 9 and single_int:
            lo, hi = int(frame.index.min()), int(frame.index.max())
            for x in (lo - 1, lo, (lo + hi) // 2, hi - 1):
                add('ss', x)
                add('succ', x)
            size = len(frame.index)
            for i in (0, size - 1, -1, -size, size, -size - 1):
                add('idx', i)
            add('ss', hi + 2)
        elif operation == 10:
            for key in misses:
                add('loc', key, c)
                add('at', key, c)
            if hits:
                add('loc', hits[0], 'zz')
                add('col', 'zz', hits[0])
            if single_int and kt == 'int':
                lo = int(frame.index.min())
                add('asof', lo - 1)
                add('asof_read', lo - 1, c)
                for big in (2 ** 63 - 1, -2 ** 63, 2 ** 63, -2 ** 63 - 1, 2 ** 64, lo - 1, lo - 2 ** 40):
                    add('loc', (big,), c)
                    add('has', (big,))
        elif operation == 11:
            for name in ('q', 'n'):
                add('rec_f', name)
            for name in ('n', 'flag', 'e'):
                add('rec_i', name)
            for name in ('s', 'q', 'e', 'zz'):
                add('rec_o', name)
            add('rec_sum', 'q', 'n')
        elif operation == 12:
            for key in hits + misses[:1]:
                add('col', c, key)
    return reads


def _case(form, operation, kt):
    reads = _reads(form, operation, kt)

    def case(mode, tmp_path_factory):
        build(tmp_path_factory).check(mode, reads)
    case.reads = reads
    return case


# One case per square (test_every_square_is_declared): a square W1.3 contributes a case for
# (its lowering, end to end) is not repeated here.
CASES = {square: _case(*square) for square in applicable()
         if square not in table_matrix_cases_w1_3.CASES}
