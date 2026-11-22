"""M1 table-read matrix (design section 4): storage form x operation x key type x mode.

Declared once here (W1.1). Applicability changes need a reason in the row and a note in the
pull request; implementation status lives in the done files of each package:
    table_matrix_done_w1_2.py   (W1.2 runtime handles: live, exported and compiled squares)
    table_matrix_done_w1_3.py   (W1.3 lowering: squares whose cases it contributes)
and the cases themselves in table_matrix_cases_w1_2.py / table_matrix_cases_w1_3.py.
A square is done in a mode when it is in the union of the done sets. Every applicable square
that is not done is xfail(strict=True): an unmarked square that passes fails the suite.
"""

FORMS = {
    'C': 'contiguous: one float64 column, int64 range index',
    'R': 'rectangular: one float64 column, dense MultiIndex product',
    'F1': 'float rows, sparse int key',
    'F2': 'float rows, multi-level key with enum and str parts',
    'G': 'general rows: float, int, bool, str and enum-label columns',
    'S': 'sub-table (T.loc[prefix]) of R and of F2',
    'O': 'one-row table',
}

OPERATIONS = {
    1: 'loc[k, "c"] float',
    2: 'at[k, "c"]',
    3: 'int / enum / bool column (int(...))',
    4: 'k in T.index',
    5: 'index.min() / index.max(), incl. min(max(x, T.index.min()), T.index.max())',
    6: 'loc[prefix] then reads',
    7: 'row = T.loc[k]; row["c"], incl. reassignment',
    8: 'index.asof(x) as row key and as a value in arithmetic',
    9: 'index.searchsorted(x, side="right") / index[i] (D25b)',
    10: 'errors: missing key, missing column, asof below the first key (D25a), DENSE at +-2**63 '
        'and below a negative first key',
    11: 'record field reads model_point()["x"] in any position',
    12: 'column selection T["c"][k] (D13)',
}

# Type of the key parts passed (part by part for multi-level keys).
KEYS = ('int', 'enum', 'str', 'tuple', 'np.int64', 'np.uint64', 'np.float32', 'np.float64',
        'bool', 'float')
KEY_AXIS_OPERATIONS = (1, 4, 6, 10)     # the key-type axis runs on these operations only
MODES = ('live', 'exported', 'compiled')

# One row per (form, operation): one code per key type in KEYS order ('y' applicable,
# '-' not applicable), then the reason for the '-' codes.
DECLARED = '''
C  1  yy--yyyy-y  no str/tuple/bool labels on an int range
C  2  y---------  key axis on 1, 4, 6, 10 only
C  3  ----------  float column only (form G carries int/enum/bool columns)
C  4  yy--yyyy-y  no str/tuple/bool labels on an int range
C  5  y---------  key axis on 1, 4, 6, 10 only
C  6  ----------  single level: no prefix
C  7  y---------  key axis on 1, 4, 6, 10 only
C  8  y---------  key axis on 1, 4, 6, 10 only
C  9  y---------  key axis on 1, 4, 6, 10 only
C  10 yy--yyyy-y  no str/tuple/bool labels on an int range
C  11 ----------  record reads are form-independent: declared once on G
C  12 y---------  key axis on 1, 4, 6, 10 only
R  1  yy--yyyy-y  int levels: no str/tuple-of-mixed/bool parts
R  2  y---------  key axis on 1, 4, 6, 10 only
R  3  ----------  float column only
R  4  yy--yyyy-y  int levels: no str/tuple-of-mixed/bool parts
R  5  y---------  key axis on 1, 4, 6, 10 only
R  6  yy--yyyy-y  int levels: no str/tuple-of-mixed/bool parts
R  7  y---------  key axis on 1, 4, 6, 10 only
R  8  ----------  multi-level: asof on a sub-table (S)
R  9  ----------  multi-level: searchsorted on a sub-table (S)
R  10 yy--yyyy-y  int levels: no str/tuple-of-mixed/bool parts
R  11 ----------  record reads are form-independent: declared once on G
R  12 y---------  key axis on 1, 4, 6, 10 only
F1 1  yy--yyyy-y  sparse int key: no str/tuple/bool labels
F1 2  y---------  key axis on 1, 4, 6, 10 only
F1 3  ----------  float rows only
F1 4  yy--yyyy-y  sparse int key: no str/tuple/bool labels
F1 5  y---------  key axis on 1, 4, 6, 10 only
F1 6  ----------  single level: no prefix
F1 7  y---------  key axis on 1, 4, 6, 10 only
F1 8  y---------  key axis on 1, 4, 6, 10 only
F1 9  y---------  key axis on 1, 4, 6, 10 only
F1 10 yy--yyyy-y  sparse int key: no str/tuple/bool labels
F1 11 ----------  record reads are form-independent: declared once on G
F1 12 y---------  key axis on 1, 4, 6, 10 only
F2 1  yyyyyyyyyy  -
F2 2  ---y------  key axis on 1, 4, 6, 10 only (natural key: mixed tuple)
F2 3  ----------  float rows only
F2 4  yyyyyyyyyy  -
F2 5  ---y------  key axis on 1, 4, 6, 10 only; min/max are first_key/last_key tuples
F2 6  yyyyyyyyyy  -
F2 7  ---y------  key axis on 1, 4, 6, 10 only
F2 8  ----------  multi-level: asof on a sub-table (S)
F2 9  ----------  multi-level: searchsorted on a sub-table (S)
F2 10 yyyyyyyyyy  -
F2 11 ----------  record reads are form-independent: declared once on G
F2 12 ---y------  key axis on 1, 4, 6, 10 only
G  1  yyy-yyyy-y  single-level labels: no tuple/bool keys
G  2  y---------  key axis on 1, 4, 6, 10 only
G  3  y---------  key axis on 1, 4, 6, 10 only
G  4  yyy-yyyy-y  single-level labels: no tuple/bool keys
G  5  y---------  key axis on 1, 4, 6, 10 only
G  6  ----------  single level: no prefix
G  7  y---------  key axis on 1, 4, 6, 10 only
G  8  y---------  key axis on 1, 4, 6, 10 only
G  9  y---------  key axis on 1, 4, 6, 10 only
G  10 yyy-yyyy-y  single-level labels: no tuple/bool keys
G  11 --y-------  field names are str; one square for all record reads
G  12 y---------  key axis on 1, 4, 6, 10 only
S  1  yyyyyyyyyy  -
S  2  y---------  key axis on 1, 4, 6, 10 only
S  3  ----------  float rows only
S  4  yyyyyyyyyy  -
S  5  y---------  key axis on 1, 4, 6, 10 only
S  6  yyyyyyyyyy  - (sub-table of a three-level F2 table)
S  7  y---------  key axis on 1, 4, 6, 10 only
S  8  y---------  key axis on 1, 4, 6, 10 only
S  9  y---------  key axis on 1, 4, 6, 10 only
S  10 yyyyyyyyyy  -
S  11 ----------  record reads are form-independent: declared once on G
S  12 y---------  key axis on 1, 4, 6, 10 only
O  1  y-y-yyyy-y  one label: no enum/tuple/bool keys
O  2  y---------  key axis on 1, 4, 6, 10 only
O  3  ----------  float column only
O  4  y-y-yyyy-y  one label: no enum/tuple/bool keys
O  5  y---------  key axis on 1, 4, 6, 10 only
O  6  ----------  single level: no prefix
O  7  y---------  key axis on 1, 4, 6, 10 only
O  8  y---------  key axis on 1, 4, 6, 10 only
O  9  y---------  key axis on 1, 4, 6, 10 only
O  10 y-y-yyyy-y  one label: no enum/tuple/bool keys
O  11 ----------  record reads are form-independent: declared once on G
O  12 y---------  key axis on 1, 4, 6, 10 only
'''


def declared():
    """{(form, operation): (codes, reason)} parsed from DECLARED."""
    rows = {}
    for line in DECLARED.strip().splitlines():
        form, operation, codes, reason = line.split(None, 3)
        key = (form, int(operation))
        if key in rows:
            raise ValueError(f'duplicate row {key}')
        rows[key] = (codes, reason)
    return rows


def applicable():
    """Applicable squares (form, operation, key type), in declaration order."""
    return [(form, operation, key) for (form, operation), (codes, _) in declared().items()
            for key, code in zip(KEYS, codes) if code == 'y']
