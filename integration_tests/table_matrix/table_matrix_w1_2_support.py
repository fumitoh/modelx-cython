"""W1.2 runtime squares of the M1 table matrix: forms, reads, oracle and the three modes.

One session build serves every square: a live modelx model (the read vocabulary in formulas,
each read caught in the formula so errors compare as values), its export, and a compiled
model (mx2cy) whose Data cells build the tables with `pdn.table` (an ordinary Python call) and
bind them to NativeTable, as production binds a table cell. Compiled reads run in a driver
module (`table_matrix_w1_2_driver.pyx`) built into the compiled package: it calls the handles
in the shape W1.3's lowering emits. The oracle is pandas on the frame the table was prepared from, with the documented
deliberate differences (D25a asof below the first key raises KeyError(label); D27b a missing
key raises KeyError(key) where pandas raises another error).

Forms: W1.1's seven, each with the storage variants W1.H's matrix reconciles to (13 forms):
C (contiguous), R (rectangular: R2 int x int, R int x str, R3 three levels), F1, F2 (and F2B
with a bool level), G (and GS, a str index), S (sub-tables: S_R3 rectangular child, S_R3C
contiguous child, S_RF float-row child of R, S_F2, S_F2F, S_R2, S_F2B) and O (O_C, O_R, O_G).
"""
import enum
import importlib
import math
from pathlib import Path
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd

Lbl = enum.IntEnum('Lbl', {('M%d' % -v if v < 0 else 'V%d' % v): v for v in range(-5, 41)})


def _frames():
    pmi = pd.MultiIndex
    return {
        'C': pd.DataFrame({'q': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]}, index=pd.Index(range(-3, 4))),
        'R2': pd.DataFrame({'q': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
                           index=pmi.from_product([[2, 1], [10, 20, 30]])),
        'R': pd.DataFrame({'q': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
                          index=pmi.from_product([[2, 1], ['b', 'a', 'c']])),
        'R3': pd.DataFrame({'q': np.arange(12.0) / 8},
                           index=pmi.from_product([[1, 0], ['y', 'x'], [5, 6, 7]])),
        'F1': pd.DataFrame({'q': [0.1, 0.3, 0.7, 2.0], 'b': [1.0, 3.0, 7.0, 20.0]},
                           index=pd.Index([1, 3, 7, 20])),
        'F2': pd.DataFrame({'q': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'b': [0.5, 0.25, 0.125, 1.5, 2.5, 3.5]},
                           index=pmi.from_tuples([(0, 'x', 1), (0, 'x', 2), (0, 'y', 1), (1, 'x', 3),
                                                  (1, 'y', 1), (1, 'y', 2)])),
        'F2B': pd.DataFrame({'q': [1.0, 2.0, 3.0, 4.0]},
                            index=pmi.from_tuples([(True, 'x', 1), (True, 'y', 2), (False, 'x', 1),
                                                   (False, 'x', 3)])),
        'G': pd.DataFrame({'q': [0.25, 0.5, 0.75], 'n': [3, 2, 1], 'flag': [True, False, True],
                           's': ['MALE', 'FEMALE', 'MALE'], 'e': [Lbl.V0, Lbl.V1, Lbl.V0]},
                          index=pd.Index([10, 20, 30])),
        'GS': pd.DataFrame({'q': [1.25, 2.5, 3.75], 'n': [7, 8, 9]}, index=pd.Index(['a', 'b', 'c'])),
        'O_C': pd.DataFrame({'q': [0.125]}, index=pd.Index([5])),
        'O_R': pd.DataFrame({'q': [9.5]}, index=pmi.from_product([[7], ['x']])),
        'O_G': pd.DataFrame({'q': [1.25], 'n': [4], 's': ['A']}, index=pd.Index([2])),
    }


FRAMES = _frames()
# variant -> (root frame, prefix or None)
LAYOUT = {name: (name, None) for name in FRAMES}
LAYOUT.update({'S_R3': ('R3', (1,)), 'S_R3C': ('R3', (0, 'x')), 'S_RF': ('R', (2,)),
               'S_F2': ('F2', (1,)), 'S_F2F': ('F2', (0, 'x')), 'S_R2': ('R2', (1,)),
               'S_F2B': ('F2B', (True,))})
VARIANTS = {'C': ['C'], 'R': ['R2', 'R', 'R3'], 'F1': ['F1'], 'F2': ['F2', 'F2B'], 'G': ['G', 'GS'],
            'S': ['S_R3', 'S_R3C', 'S_RF', 'S_F2', 'S_F2F', 'S_R2', 'S_F2B'],
            'O': ['O_C', 'O_R', 'O_G']}


def frame_of(variant):
    root, prefix = LAYOUT[variant]
    frame = FRAMES[root]
    if prefix is None:
        return frame
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return frame.loc[prefix if len(prefix) > 1 else prefix[0]]


def levels(variant):
    return frame_of(variant).index.nlevels


def int_index(variant):
    index = frame_of(variant).index
    return index.nlevels == 1 and index.dtype.kind == 'i'


# ---------------------------------------------------------------- keys by key type

def _cast(part, kt):
    """An int label part as key type kt (None: not representable); other parts unchanged."""
    if type(part) is not int:
        return part
    if kt == 'enum':
        return Lbl(part)
    if kt == 'np.int64':
        return np.int64(part)
    if kt == 'np.uint64':
        return np.uint64(part) if part >= 0 else None
    if kt == 'np.float32':
        return np.float32(part)
    if kt == 'np.float64':
        return np.float64(part)
    if kt == 'float':
        return float(part)
    return part


def _natural(variant):
    """(hit keys, miss keys) as tuples of label parts."""
    frame = frame_of(variant)
    labels = [key if type(key) is tuple else (key,) for key in frame.index]
    labels = [tuple(part.item() if isinstance(part, np.generic) else part for part in key)
              for key in labels]
    hits = list(dict.fromkeys([labels[0], labels[len(labels) // 2], labels[-1]]))
    misses = []
    for j in range(len(labels[0])):
        parts = [key[j] for key in labels]
        if all(type(part) is int for part in parts):
            for value in (max(parts) + 1, min(parts) - 1):
                misses.append(labels[0][:j] + (value,) + labels[0][j + 1:])
        elif all(type(part) is str for part in parts):
            misses.append(labels[0][:j] + ('zz',) + labels[0][j + 1:])
        elif all(type(part) is bool for part in parts) and len(set(parts)) == 1:
            misses.append(labels[0][:j] + (not parts[0],) + labels[0][j + 1:])
    return hits, [key for key in misses if key not in labels]


def keys(variant, kt):
    """(hits, misses) for key type kt: each a tuple of key parts as passed."""
    hits, misses = _natural(variant)
    if kt == 'str':
        # the str parts decide: hits are natural keys, misses change a str part (or are str
        # keys on a level that holds no str)
        str_levels = [j for j, part in enumerate(hits[0]) if type(part) is str]
        if str_levels:
            j = str_levels[0]
            return hits, [hits[0][:j] + ('zz',) + hits[0][j + 1:]]
        return [], [('zz',) + hits[0][1:], ('1',) + hits[0][1:]]
    if kt == 'bool':
        bool_levels = [j for j, part in enumerate(hits[0]) if type(part) is bool]
        if bool_levels:
            return hits, misses
        return [], [(True,) + hits[0][1:], (False,) + hits[0][1:]]
    if kt == 'tuple':
        return hits, misses
    cast = lambda key: tuple(_cast(part, kt) for part in key)  # noqa: E731
    hits = [cast(key) for key in hits]
    misses = [cast(key) for key in misses]
    if kt in ('float', 'np.float32', 'np.float64'):
        number = [j for j, part in enumerate(hits[0]) if isinstance(part, (float, np.floating))]
        if number:
            j = number[0]
            misses.append(hits[0][:j] + (_cast_float(kt, 1.5),) + hits[0][j + 1:])
    return ([key for key in hits if None not in key], [key for key in misses if None not in key])


def _cast_float(kt, value):
    return {'float': float, 'np.float32': np.float32, 'np.float64': np.float64}[kt](value)


def cint(kt, key):
    """Compiled code passes int and enum parts as C long long when every part is an int that
    fits in 64 bits (a larger int can only be an object in compiled code)."""
    return kt in ('int', 'enum') and all(isinstance(part, int) and type(part) is not bool
                                         and -2 ** 63 <= part < 2 ** 63 for part in key)


def as_c(key):
    return tuple(int(part) for part in key)


def item(key):
    return key[0] if len(key) == 1 else key


# ---------------------------------------------------------------- outcomes and oracle

def norm(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, tuple):
        return tuple(norm(part) for part in value)
    if isinstance(value, float) and value != value:
        return 'nan'
    if isinstance(value, enum.Enum):
        return value.value
    return value


def outcome(read):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return ('ok', norm(read()))
    except Exception as exc:     # noqa: BLE001 - errors are compared as values
        return (type(exc).__name__, tuple(norm(arg) for arg in exc.args))


def _bool_on_number_level(frame, key):
    """A bool part on a level holding no bool (the helper's boolean label modes, W1.H): never a
    label. pandas matches True to 1 in a partial key (``(True, 'x') in index``, ``loc[(True,
    'x')]``) but not in a complete key; the helper, and so the handles, never match it."""
    index = frame.index
    for j, part in enumerate(key[:index.nlevels]):
        if type(part) is bool:
            values = index.levels[j] if index.nlevels > 1 else index
            if values.dtype.kind != 'b':
                return True
    return False


def oracle(read, frame, args):
    """pandas' outcome of read on frame, with the deliberate differences of the helper."""
    kind = read
    if kind == 'has' and _bool_on_number_level(frame, args[0]):
        return ('ok', False)
    if kind == 'sub_obj':
        kind = 'sub_loc'
    if kind in ('sub_loc', 'sub_has') and _bool_on_number_level(frame, args[0]):
        return ('KeyError', (norm(item(args[0])),))
    if kind in ('loc', 'at', 'int', 'obj'):
        key, column = args
        if column not in frame.columns:
            return ('KeyError', (column,))
        try:
            got = outcome(lambda: frame.loc[item(key), column])
        except Exception:     # noqa: BLE001
            got = ('error', ())
        if got[0] != 'ok' or isinstance(got[1], (pd.Series, pd.DataFrame)):
            return ('KeyError', (norm(item(key)),))
        if kind == 'int':
            return outcome(lambda: int(frame.loc[item(key), column]))
        if kind == 'loc' or kind == 'at':
            return ('ok', norm(frame.loc[item(key), column]))
        return got
    if kind == 'has':
        key, = args
        return outcome(lambda: item(key) in frame.index)
    if kind == 'minmax':
        return outcome(lambda: (frame.index.min(), frame.index.max()))
    if kind == 'clamp':
        x, = args
        return outcome(lambda: min(max(x, frame.index.min()), frame.index.max()))
    if kind == 'clamp_read':
        x, column = args
        return outcome(lambda: frame.loc[min(max(x, frame.index.min()), frame.index.max()), column])
    if kind in ('sub_loc', 'sub_has'):
        prefix, key, column = args
        try:
            sub = frame.loc[item(prefix)]
            if not isinstance(sub, pd.DataFrame):
                raise KeyError(item(prefix))
        except Exception:     # noqa: BLE001
            return ('KeyError', (norm(item(prefix)),))
        if kind == 'sub_has':
            return outcome(lambda: item(key) in sub.index)
        return oracle('loc', sub, (key, column))
    if kind == 'rows':
        first, second, column = args
        a = oracle('loc', frame, (first, column))
        if a[0] != 'ok':
            return a
        b = oracle('loc', frame, (second, column))
        if b[0] != 'ok':
            return b
        return ('ok', a[1] + b[1])
    if kind in ('asof', 'asof_read', 'asof_arith'):
        x = args[0]
        found = frame.index.asof(x)
        if isinstance(found, float) and found != found:
            return ('KeyError', (x,))
        if kind == 'asof':
            return ('ok', norm(found))
        if kind == 'asof_arith':
            return ('ok', norm(found) * 2 + 1)
        return ('ok', norm(frame.loc[found, args[1]]))
    if kind == 'ss':
        x, = args
        return outcome(lambda: int(frame.index.searchsorted(x, side='right')))
    if kind == 'idx':
        i, = args
        return outcome(lambda: frame.index[i])
    if kind == 'succ':
        x, = args
        return outcome(lambda: frame.index[int(frame.index.searchsorted(x, side='right'))])
    if kind == 'col':
        column, key = args
        if column not in frame.columns:
            return ('KeyError', (column,))
        try:
            return ('ok', norm(frame[column].loc[item(key)]))
        except Exception:     # noqa: BLE001
            return ('KeyError', (norm(item(key)),))
    if kind in ('rec_f', 'rec_i', 'rec_o'):
        name, = args
        row = frame.loc[30]
        if name not in row.index:
            return ('KeyError', (name,))
        value = row[name]
        return ('ok', norm(float(value) if kind == 'rec_f' else int(value) if kind == 'rec_i' else value))
    if kind == 'rec_sum':
        row = frame.loc[30]
        return ('ok', float(row[args[0]]) * 2 + float(row[args[1]]))
    raise ValueError(kind)


# ---------------------------------------------------------------- live and exported formulas

_TRY = '''\
def {name}(form, kt, {params}):
    try:
{body}
    except Exception as exc:
        return (type(exc).__name__, exc.args)
'''
READS = {
    # read kind: (parameters, body using t = table(form)) -- the target syntax, live and exported
    'loc': ('k, c', "t = table(form)\nreturn ('ok', t.loc[k, c])"),
    'at': ('k, c', "t = table(form)\nreturn ('ok', t.at[k, c])"),
    'int': ('k, c', "t = table(form)\nreturn ('ok', int(t.loc[k, c]))"),
    'obj': ('k, c', "t = table(form)\nreturn ('ok', t.loc[k, c])"),
    'has': ('k', "t = table(form)\nreturn ('ok', k in t.index)"),
    'minmax': ('x', "t = table(form)\nreturn ('ok', (t.index.min(), t.index.max()))"),
    'clamp': ('x', "t = table(form)\nreturn ('ok', min(max(x, t.index.min()), t.index.max()))"),
    'clamp_read': ('x, c', "t = table(form)\nreturn ('ok', t.loc[min(max(x, t.index.min()), t.index.max()), c])"),
    'sub_loc': ('p, k, c', "sub = table(form).loc[p]\nreturn ('ok', sub.loc[k, c])"),
    'sub_has': ('p, k, c', "sub = table(form).loc[p]\nreturn ('ok', k in sub.index)"),
    'sub_obj': ('p, k, c', "sub = table(form).loc[p]\nreturn ('ok', sub.loc[k, c])"),
    'rows': ('k1, k2, c', "row = table(form).loc[k1]\nvalue = row[c]\nrow = table(form).loc[k2]\n"
                          "return ('ok', value + row[c])"),
    'asof': ('x', "t = table(form)\nreturn ('ok', t.index.asof(x))"),
    'asof_read': ('x, c', "t = table(form)\nreturn ('ok', t.loc[t.index.asof(x), c])"),
    'asof_arith': ('x', "t = table(form)\nreturn ('ok', t.index.asof(x) * 2 + 1)"),
    'ss': ('x', "t = table(form)\nreturn ('ok', t.index.searchsorted(x, side='right'))"),
    'idx': ('i', "t = table(form)\nreturn ('ok', t.index[i])"),
    'succ': ('x', "t = table(form)\nreturn ('ok', t.index[t.index.searchsorted(x, side='right')])"),
    'col': ('c, k', "t = table(form)\nreturn ('ok', t[c][k])"),
    'rec_f': ('name', "return ('ok', float(point()[name]))"),
    'rec_i': ('name', "return ('ok', int(point()[name]))"),
    'rec_o': ('name', "return ('ok', point()[name])"),
    'rec_sum': ('a, b', "return ('ok', point()[a] * 2 + point()[b])"),
}


def _formula(kind):
    params, body = READS[kind]
    return _TRY.format(name='r_' + kind, params=params,
                       body=textwrap.indent(body, ' ' * 8))


_DATA = {
    'base': '''\
def base(root) -> "modelx_helpers.immutable_inputs.Table":
    return pdn.table(frames[root])
''',
    'point': '''\
def point():
    return base("G").loc[30]
''',
}
_TABLE = '''\
def table(form):
    root, prefix = layout[form]
    t = base(root)
    if prefix is None:
        return t
    return t.loc[prefix if len(prefix) > 1 else prefix[0]]
'''


def _model(name, with_reads):
    import modelx as mx
    from modelx_helpers import pdn
    model = mx.new_model(name)
    space = model.new_space('M', refs={'frames': FRAMES, 'layout': LAYOUT, 'pdn': pdn})
    for source in _DATA.values():
        space.new_cells(formula=source)
    if with_reads:
        space.new_cells(formula=_TABLE)
        for kind in READS:
            space.new_cells(formula=_formula(kind))
    return model


# ---------------------------------------------------------------- the session build

class Build:
    """The three modes of the W1.2 squares (one per test session)."""

    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.live_model = _model('W12Live', True)
        self.live = self.live_model.spaces['M']
        self.live_model.export(self.tmp / 'w12x')
        compiled = _model('W12Comp', False)
        compiled.export(self.tmp / 'w12c')
        compiled.close()
        sys.path.insert(0, str(self.tmp))
        self.exported = importlib.import_module('w12x').mx_model.M
        self._compile()
        package = importlib.import_module('w12c_cy')
        self.compiled = package.mx_model.M
        self.driver = importlib.import_module('w12c_cy.table_matrix_w1_2_driver')
        self.fallbacks = importlib.import_module(self.provider).handle_fallbacks

    def _compile(self):
        (self.tmp / 'sample_w12.py').write_text(
            'from w12c import mx_model\nm = mx_model.M\n'
            f'for root in {sorted(FRAMES)!r}:\n    m.base(root)\nm.point()\n', encoding='utf-8')
        done = subprocess.run([sys.executable, '-m', 'modelx_cython', 'w12c', '--sample', 'sample_w12.py',
                               '--no-spec'], cwd=self.tmp, capture_output=True, text=True)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        package = self.tmp / 'w12c_cy'
        providers = [path for path in package.glob('_mx_helper_*.pxd')
                     if 'cdef class TableHandle' in path.read_text(encoding='utf-8')]
        assert len(providers) == 1, providers
        self.provider = 'w12c_cy.' + providers[0].stem
        template = (Path(__file__).with_name('table_matrix_w1_2_driver.pyx')).read_text(encoding='utf-8')
        (package / 'table_matrix_w1_2_driver.pyx').write_text(
            template.replace('{PROVIDER}', self.provider), encoding='utf-8')
        import modelx_cython
        root = Path(modelx_cython.__file__).resolve().parents[1]
        (self.tmp / 'setup_w12_driver.py').write_text(textwrap.dedent(f'''\
            from setuptools import setup, Extension
            from Cython.Build import cythonize
            setup(script_args=['build_ext', '--inplace'], ext_modules=cythonize(
                [Extension('w12c_cy.table_matrix_w1_2_driver', ['w12c_cy/table_matrix_w1_2_driver.pyx'])],
                include_path=[{str(self.tmp)!r}, {str(root)!r}],
                compiler_directives={{'language_level': 3}}, quiet=True))
            '''), encoding='utf-8')
        done = subprocess.run([sys.executable, 'setup_w12_driver.py'], cwd=self.tmp,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]

    # -- one read in one mode -------------------------------------------------------------

    def table(self, variant):
        """The compiled table of a variant: the bound root, or its eager child (sub_n)."""
        root, prefix = LAYOUT[variant]
        native = self.compiled.base(root)
        assert type(native).__name__ == 'NativeTable', native
        if prefix is None:
            return native
        return self.driver.sub(native, tuple(prefix), False)

    def run(self, mode, read, variant, kt, args):
        if mode in ('live', 'exported'):
            space = self.live if mode == 'live' else self.exported
            result = getattr(space, 'r_' + read)(variant, kt, *_live_args(read, args))
            return (result[0], norm(result[1]) if result[0] == 'ok' else
                    tuple(norm(arg) for arg in result[1]))
        return outcome(lambda: self._compiled(read, variant, kt, args))

    def _compiled(self, read, variant, kt, args):
        d = self.driver
        if read.startswith('rec'):
            record = self.compiled.point()
            if read == 'rec_f':
                return d.field(record, args[0])
            if read == 'rec_i':
                return d.field_int(record, args[0])
            if read == 'rec_o':
                return d.field_obj(record, args[0])
            return d.field_sum(record, *args)
        table = self.table(variant)

        def key(parts):
            return (as_c(parts), True) if cint(kt, parts) else (tuple(parts), False)
        if read in ('loc', 'at', 'int', 'obj'):
            parts, c_int = key(args[0])
            return {'loc': d.loc, 'at': d.loc, 'int': d.loc_int, 'obj': d.loc_obj}[read](
                table, parts, args[1], c_int)
        if read == 'has':
            return d.contains(table, *key(args[0]))
        if read == 'minmax':
            if int_index(variant):
                return d.first_last(table)
            return d.first_last_key(table)
        if read == 'clamp':
            return d.clamp(table, args[0])
        if read == 'clamp_read':
            return d.clamp_read(table, args[0], args[1])
        if read == 'sub_obj':
            # a table-valued cell returns the shared child's bound table (D45); the caller reads it
            prefix, c_prefix = key(args[0])
            table_object = d.sub_object(table, prefix, c_prefix)
            assert table_object is d.sub_object(table, prefix, c_prefix)
            assert type(table_object).__name__ == 'NativeTable'
            assert table_object.handle is d.sub(table, prefix, c_prefix)
            parts, c_int = key(args[1])
            return d.loc(table_object, parts, args[2], c_int)
        if read in ('sub_loc', 'sub_has'):
            prefix, c_prefix = key(args[0])
            sub = d.sub(table, prefix, c_prefix)
            if read == 'sub_has':
                return d.contains(sub, *key(args[1]))
            parts, c_int = key(args[1])
            return d.loc(sub, parts, args[2], c_int)
        if read == 'rows':
            first, c_first = key(args[0])
            second, _ = key(args[1])
            return d.row_reads(table, first, second, args[2], c_first and cint(kt, args[1]))
        if read == 'asof':
            return d.asof(table, args[0])
        if read == 'asof_read':
            return d.asof_read(table, args[0], args[1])
        if read == 'asof_arith':
            return d.asof_arith(table, args[0])
        if read == 'ss':
            return d.searchsorted(table, args[0])
        if read == 'idx':
            return d.key_at(table, args[0])
        if read == 'succ':
            return d.successor(table, args[0])
        if read == 'col':
            parts, c_int = key(args[1])
            return d.column(table, args[0], parts, c_int)
        raise ValueError(read)

    def check(self, mode, reads):
        """Run reads [(read, variant, kt, args)] in mode against the oracle."""
        assert reads, 'a square needs at least one read'
        failures = []
        for read, variant, kt, args in reads:
            frame = frame_of(variant) if not read.startswith('rec') else FRAMES['G']
            expected = oracle(read, frame, args)
            got = self.run(mode, read, variant, kt, args)
            if not _same(got, expected):
                failures.append((read, variant, kt, args, 'got', got, 'expected', expected))
        assert not failures, failures


def _live_args(read, args):
    if read in ('loc', 'at', 'int', 'obj', 'has'):
        return (item(args[0]),) + tuple(args[1:])
    if read in ('sub_loc', 'sub_has', 'sub_obj'):
        return (item(args[0]), item(args[1])) + tuple(args[2:])
    if read == 'rows':
        return (item(args[0]), item(args[1]), args[2])
    if read == 'col':
        return (args[0], item(args[1]))
    return tuple(args)


def _same(got, expected):
    if got == expected:
        return True
    if got[0] == expected[0] == 'ok' and isinstance(got[1], float) and isinstance(expected[1], float):
        return got[1] == expected[1] or (math.isnan(got[1]) and math.isnan(expected[1]))
    return False
