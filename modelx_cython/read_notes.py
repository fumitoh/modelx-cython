"""Notes of what arrives at each table read during the sample run (decisions from tracing, not from text form).

While the sample runs, the modules of the exported model are imported through a loader that wraps, in the formula
text only: every subscript load, every method call, every ``in`` and every loop source. The wrapper performs the
original operation unchanged and records, for that place in the text, what kind of value the receiver was, what kind
of key it took and what kind of value came back.  A pandas object is followed by identity: the frame a zero-argument
Cell returned, a row or a sub-table read from it, its index.

``NOTES``: (class, function, text of the node, operation) -> set of observations.
The transformer reads them by position in the same text; a place with no note was never reached by the sample and a
place with more than one observation saw more than one kind of value: both stay as written.
"""
import enum
import importlib.abc
import importlib.machinery
import sys
from contextlib import contextmanager

import libcst as cst

NOTES = {}
COUNTS = {}        # the same key -> evaluations in the sample
_TRACED_COUNTS = {}  # evaluations whose observed receiver came from a table
_TABLES, _SUBS, _ROWS, _COLS, _INDEXES, _KEEP = {}, {}, {}, {}, {}, []
_TRACED = ('table:', 'sub:', 'row:', 'col:', 'index:', 'loc:')
_PLAIN_SUBS, _PLAIN_CALLS = {}, {}
_BUILTIN_TYPES = frozenset((bool, int, float, str, bytes, complex, list, tuple, dict, set, range, slice, type(None)))


def _pandas():
    return sys.modules.get('pandas')


def kind(value):
    pd = _pandas()
    if isinstance(value, bool) or type(value).__name__ == 'bool_':
        return 'bool'
    if isinstance(value, enum.Enum):
        return 'enum'
    if isinstance(value, int) or type(value).__module__ == 'numpy' and 'int' in type(value).__name__:
        return 'int'
    if isinstance(value, float) or type(value).__module__ == 'numpy' and 'float' in type(value).__name__:
        return 'float'
    if isinstance(value, str):
        return 'str'
    if pd is not None:
        if isinstance(value, pd.DataFrame):
            if id(value) in _TABLES:
                return 'table:' + _TABLES[id(value)]
            if id(value) in _SUBS:
                return 'sub:' + _SUBS[id(value)]
            return 'frame'
        if isinstance(value, pd.Series):
            if id(value) in _ROWS:
                return 'row:' + _ROWS[id(value)]
            return 'col:' + _COLS[id(value)] if id(value) in _COLS else 'series'
        if isinstance(value, (pd.core.indexing._LocIndexer, pd.core.indexing._AtIndexer)):
            inner = kind(value.obj)
            return 'loc:' + inner if inner not in ('frame', 'series') else 'loc'
        if isinstance(value, pd.Index):
            return 'index:' + _INDEXES[id(value)] if id(value) in _INDEXES else 'Index'
    return type(value).__name__


def _shape(key):
    if isinstance(key, tuple):
        return tuple(_shape(part) for part in key)
    found = kind(key)
    return 'num' if found in ('int', 'float', 'enum') else found


def _note(nid, *what):
    NOTES.setdefault(nid, set()).add(what)
    COUNTS[nid] = COUNTS.get(nid, 0) + 1
    if what[0].startswith(_TRACED) and nid[3] != 'v':
        _TRACED_COUNTS[nid] = _TRACED_COUNTS.get(nid, 0) + 1


def _follow(value, origin):
    """Remember where a pandas object came from; its index belongs to it."""
    pd = _pandas()
    if pd is None:
        return
    origin_kind, name = origin
    if isinstance(value, pd.DataFrame):
        if id(value) not in _TABLES and id(value) not in _SUBS:
            (_SUBS if origin_kind == 'sub' else _TABLES)[id(value)] = name
            _KEEP.append(value)
        index = value.index
        index_origin = ('sub:' if id(value) in _SUBS else 'table:') + name
    elif isinstance(value, pd.Series):
        if id(value) not in _ROWS and id(value) not in _COLS:
            (_ROWS if origin_kind == 'row' else _COLS)[id(value)] = name
            _KEEP.append(value)
        index = value.index
        index_origin = ('row:' if id(value) in _ROWS else 'col:') + name
    elif isinstance(value, pd.Index):
        index, index_origin = value, name
    else:
        return
    if id(index) not in _INDEXES:
        _INDEXES[id(index)] = index_origin
        _KEEP.append(index)


RAISES = 'raises'      # the result kind of an evaluation that raised


def r_sub(nid, receiver, key, /):
    receiver_type = type(receiver)
    plain_sequence = (receiver_type is list or receiver_type is tuple) and type(key) is int
    found = receiver_type.__name__ if plain_sequence else kind(receiver)
    try:
        out = receiver[key]
    except Exception:
        column = key[-1] if isinstance(key, tuple) and key and isinstance(key[-1], str) else key if isinstance(key, str) else None
        _note(nid, found, _shape(key), RAISES, column)
        raise
    if plain_sequence:
        # These observations depend only on the exact result type. Count repeated ones without rebuilding
        # observations or calling the pandas provenance and key-shape machinery under the sample tracer.
        result_type = type(out)
        if nid in _PLAIN_SUBS:
            last_receiver, last_result = _PLAIN_SUBS[nid]
            if receiver_type is last_receiver and result_type is last_result:
                COUNTS[nid] += 1
                return out
        _note(nid, found, 'num', kind(out), None)
        if type(result_type) is type and result_type in _BUILTIN_TYPES:
            _PLAIN_SUBS[nid] = receiver_type, result_type
        return out
    pd = _pandas()
    if pd is not None and found.startswith('loc:') and not isinstance(key, (pd.Series, list, slice)) \
            and type(key).__name__ != 'ndarray':
        base = found.split(':')[-1]
        if isinstance(out, pd.Series) and (not isinstance(key, tuple) or all(not isinstance(p, str) for p in key)):
            _follow(out, ('row', base))
        elif isinstance(out, pd.DataFrame) and found.startswith('loc:table:'):
            _follow(out, ('sub', base))
    elif pd is not None and found.startswith(('table:', 'sub:')) and isinstance(out, pd.Series):
        _follow(out, ('col', found.split(':')[-1]))
    column = key[-1] if isinstance(key, tuple) and key and isinstance(key[-1], str) else key if isinstance(key, str) else None
    _note(nid, found, _shape(key), kind(out), column)
    return out


def r_call(nid, receiver, name, /, *args, **kwargs):
    try:
        out = getattr(receiver, name)(*args, **kwargs)
    except Exception:
        _note(nid, kind(receiver), name, RAISES)
        raise
    receiver_type, result_type = type(receiver), type(out)
    if nid in _PLAIN_CALLS:
        last_receiver, last_name, last_result = _PLAIN_CALLS[nid]
        if receiver_type is last_receiver and name is last_name and result_type is last_result:
            COUNTS[nid] += 1
            return out
    pd = _pandas()
    found = kind(receiver)
    if pd is not None and isinstance(out, pd.DataFrame) and not args and not kwargs \
            and id(out) not in _TABLES and id(out) not in _SUBS:
        owner = type(receiver).__name__
        _follow(out, ('table', (owner[3:] if owner.startswith('_c_') else owner) + '.' + name))     # a Cell's table
    elif pd is not None and isinstance(out, (pd.Index, pd.Series)):
        if found.startswith(_TRACED):
            base = found.split(':')[-1]
            _follow(out, ('col', 'derived:' + base if isinstance(out, pd.Index) else base))
        elif not args and not kwargs and isinstance(out, pd.Series):
            owner = type(receiver).__name__
            if owner.startswith('_c_'):
                _follow(out, ('col', owner[3:] + '.' + name))
    _note(nid, found, name, kind(out))
    if (found.startswith('_c_') and found == receiver_type.__name__
            and type(result_type) is type and result_type in _BUILTIN_TYPES):
        _PLAIN_CALLS[nid] = receiver_type, name, result_type
    return out


def r_iter(nid, func, source, /):
    """Observe the pandas input when the formula constructs an iterator explicitly."""
    if func is not iter:
        return func(source)
    found = kind(source)
    try:
        out = func(source)
    except Exception:
        _note(nid, found, '__iter__', RAISES)
        raise
    _note(nid, found, '__iter__', kind(out))
    return out


def r_in(nid, left, right):
    try:
        out = left in right
    except Exception:
        _note(nid, kind(right), _shape(left), RAISES)
        raise
    _note(nid, kind(right), _shape(left))
    return out


def r_it(nid, source):
    _note(nid, kind(source))
    return source


def r_v(nid, value):
    _note(nid, kind(value))
    return value


class _Wrap(cst.CSTTransformer):
    """Wraps the reads of the formulas (methods ``_f_<cell>`` of a Space class); everything else is left as it is."""

    def __init__(self):
        super().__init__()
        self.cls = self.fn = None
        self.depth = 0
        self.targets = set()        # ids of nodes that are assigned to or deleted, not read

    @staticmethod
    def code(node):
        return cst.Module(body=()).code_for_node(node)

    def nid(self, node, op):
        return cst.parse_expression(repr((self.cls, self.fn, normal(self.code(node)), op)))

    @staticmethod
    def call(name, original, *args, keywords=()):
        return cst.Call(func=cst.Name(name), args=[cst.Arg(a) for a in args] + list(keywords),
                        lpar=getattr(original, 'lpar', ()), rpar=getattr(original, 'rpar', ()))

    @staticmethod
    def bare(node):
        return node.with_changes(lpar=(), rpar=()) if getattr(node, 'lpar', None) else node

    def _mark(self, target):
        for node in _walk(target):
            self.targets.add(id(node))

    def visit_ClassDef(self, node):
        if self.cls is not None or self.fn is not None:
            return False
        self.cls = node.name.value
        return True

    def leave_ClassDef(self, original, updated):
        if self.cls == original.name.value and self.fn is None:
            self.cls = None
        return updated

    def visit_FunctionDef(self, node):
        if self.cls is None or self.fn is not None or not node.name.value.startswith('_f_'):
            return False
        self.fn = node.name.value
        return True

    def leave_FunctionDef(self, original, updated):
        if self.fn == original.name.value:
            self.fn = None
        return updated

    def visit_Lambda(self, node):
        return False

    def visit_Assign(self, node):
        for target in node.targets:
            self._mark(target.target)

    def visit_AugAssign(self, node):
        self._mark(node.target)

    def visit_AnnAssign(self, node):
        self._mark(node.target)

    def visit_Del(self, node):
        self._mark(node.target)

    def visit_For(self, node):
        self._mark(node.target)

    def visit_CompFor(self, node):
        self._mark(node.target)

    def visit_WithItem(self, node):
        if node.asname is not None:
            self._mark(node.asname.name)

    def leave_Subscript(self, original, updated):
        if self.fn is None or id(original) in self.targets:
            return updated
        parts = []
        for element in updated.slice:
            if not isinstance(element.slice, cst.Index) or element.slice.star is not None:
                return updated
            parts.append(element.slice.value)
        key = parts[0] if len(parts) == 1 else cst.Tuple([cst.Element(p) for p in parts])
        return self.call('_mx_note_sub', updated, self.nid(original, 'sub'), updated.value, key)

    def leave_Call(self, original, updated):
        if self.fn is None:
            return updated
        if any(arg.star for arg in updated.args):
            return updated
        func = updated.func
        if isinstance(func, cst.Name) and func.value in ('int', 'float', 'bool') and len(updated.args) == 1 \
                and updated.args[0].keyword is None:
            inner = self.call('_mx_note_v', None, self.nid(original.args[0].value, 'v'), updated.args[0].value)
            return updated.with_changes(args=[updated.args[0].with_changes(value=inner)])
        if (isinstance(func, cst.Name) and func.value == 'iter' and len(updated.args) == 1
                and updated.args[0].keyword is None):
            return self.call('_mx_note_iter', updated, self.nid(original, 'call'), func, updated.args[0].value)
        if not isinstance(func, cst.Attribute):
            return updated
        return self.call('_mx_note_call', updated, self.nid(original, 'call'), func.value,
                         cst.SimpleString(repr(func.attr.value)), keywords=list(updated.args))

    def leave_Comparison(self, original, updated):
        if self.fn is None or len(updated.comparisons) != 1:
            return updated
        target = updated.comparisons[0]
        if not isinstance(target.operator, (cst.In, cst.NotIn)):
            return updated
        found = self.call('_mx_note_in', None, self.nid(original, 'in'), updated.left, target.comparator)
        if isinstance(target.operator, cst.NotIn):
            found = cst.UnaryOperation(operator=cst.Not(), expression=found.with_changes(
                lpar=[cst.LeftParen()], rpar=[cst.RightParen()]))
        return found.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])

    def _source(self, original, updated):
        if self.fn is None:
            return updated
        return updated.with_changes(iter=self.call('_mx_note_it', None, self.nid(original.iter, 'it'), updated.iter))

    def leave_For(self, original, updated):
        return self._source(original, updated)

    def leave_CompFor(self, original, updated):
        return self._source(original, updated)


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def instrument(source, filename):
    module = cst.parse_module(source).visit(_Wrap())
    return compile(module.code, filename, 'exec', dont_inherit=True)


class _Loader(importlib.machinery.SourceFileLoader):
    def source_to_code(self, data, path, *, _optimize=-1):
        source = importlib.util.decode_source(data) if isinstance(data, bytes) else data
        return instrument(source, path)

    def get_code(self, fullname):       # never a cached byte-code file of the plain text
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, package):
        self.package = package

    def find_spec(self, fullname, path, target=None):
        if fullname != self.package and not fullname.startswith(self.package + '.'):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader) \
                and not fullname.endswith('_mx_sys'):
            spec.loader = _Loader(spec.loader.name, spec.loader.path)
        return spec


@contextmanager
def recording(package):
    """Record the reads of the model package ``package`` while it is imported and run inside the block."""
    import builtins
    import importlib.util       # noqa: F401  (decode_source)
    for store in (NOTES, COUNTS, _TRACED_COUNTS, _TABLES, _SUBS, _ROWS, _COLS, _INDEXES, _PLAIN_SUBS, _PLAIN_CALLS):
        store.clear()
    del _KEEP[:]
    hooks = {'_mx_note_sub': r_sub, '_mx_note_call': r_call, '_mx_note_in': r_in, '_mx_note_it': r_it,
             '_mx_note_iter': r_iter, '_mx_note_v': r_v}
    finder = _Finder(package)
    sys.meta_path.insert(0, finder)
    for name, hook in hooks.items():
        setattr(builtins, name, hook)
    try:
        yield NOTES
    finally:
        sys.meta_path.remove(finder)
        for name in hooks:
            delattr(builtins, name)
        del _KEEP[:]


def as_json():
    """The notes as a JSON value (the build writes them next to the per-read report)."""
    return [{'class': key[0], 'function': key[1], 'text': key[2], 'op': key[3], 'evaluations': COUNTS.get(key, 0),
             'seen': sorted(repr(n) for n in NOTES[key])}
            for key in sorted(NOTES, key=lambda k: tuple(str(x) for x in k))]


def normal(text):
    """Source text of a node, as both parsers give it: without surrounding spaces and outer parentheses."""
    text = ' '.join((text or '').split())
    while text.startswith('(') and text.endswith(')') and _balanced(text[1:-1]):
        text = text[1:-1].strip()
    return text


def _balanced(text):
    depth = 0
    for ch in text:
        depth += ch in '([{'
        depth -= ch in ')]}'
        if depth < 0:
            return False
    return depth == 0


def by_text():
    """Group notes by formula and expression text, keeping counts for operations on pandas receivers."""
    out = {}
    for key, seen in NOTES.items():
        entry = out.setdefault(key[:2], {}).setdefault(key[2], {
            'seen': set(), 'count': 0, 'table_reads': set(), 'table_count': 0,
        })
        count = COUNTS.get(key, 0)
        entry['seen'] |= seen
        entry['count'] += count
        # A method call and the loop over its result can share source text but have different receivers.
        op = key[3]
        reads = {(op, observation) for observation in seen if table_read(op, observation)}
        if reads:
            entry['table_reads'].update(reads)
            entry['table_count'] += _TRACED_COUNTS[key]
    return out


SELECTORS = ('series', 'list', 'ndarray', 'slice', 'frame', 'Index', 'set', 'dict', 'DataFrame', 'Series')


def selects_rows(shape):
    """A recorded key shape with a part that selects several rows (a mask, a list, a slice, a pandas object)."""
    if isinstance(shape, tuple):
        return any(selects_rows(part) for part in shape)
    return shape in SELECTORS or str(shape).startswith((*_TRACED, 'loc'))


def only_raised(note):
    """Whether every evaluation the sample made at a place raised (no evidence of what a read there returns)."""
    seen = note['seen'] if note else ()
    return bool(seen) and all(len(s) >= 3 and s[2] == RAISES for s in seen)


def table_read(op, seen):
    """Whether an observation uses a traced pandas receiver."""
    lengths = {'sub': (4,), 'call': (3,), 'in': (2, 3), 'it': (1,)}
    return len(seen) in lengths.get(op, ()) and seen[0].startswith(_TRACED)


def scalar_table_read(op, seen):
    """The table reads tracked separately from the general set of pandas operations in the audit."""
    receiver = seen[0]
    if op == 'sub':
        return receiver.startswith(('loc:table:', 'loc:sub:', 'loc:row:', 'row:'))
    index = receiver.startswith(('index:table:', 'index:sub:'))
    if op == 'call':
        return index and seen[1] in ('max', 'min', 'asof', 'searchsorted')
    return op in ('in', 'it') and index
