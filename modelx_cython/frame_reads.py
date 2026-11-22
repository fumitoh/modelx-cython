"""Reads of a traced pandas DataFrame or Series lowered onto a PreparedTable (T-3; toolkit p2/tables/DESIGN.md).

The formula keeps the user's frame: the sample runs on it (no substitution) and the compiled model reads the same
object. A read whose form the compiler recognises is answered from a PreparedTable bound to that frame object
(runtime ``_tables``); every other operation runs on the frame itself.

Recognised here (the receiver ``R`` a self-rooted zero-argument call of a cached Cell whose traced value is a
pandas DataFrame or Series with a unique index, or a local assigned once from such a call):

- L1 ``R.loc[key, "c"]`` / ``R.at[key, "c"]`` of a float64 or int64 column, the key complete (one part per
  level); a Series ``R.loc[k]``, ``R.at[k]``, ``R[k]``;
- L7 ``key in R.index`` / ``not in`` with a complete key.

Binding, by the kind of the reading class:

- in a class whose instances are items (an ItemSpace), the receiver is evaluated, and its PreparedTable found by
  identity of the frame object, once per item: when the hold is first needed. The item's fields hold the table
  (``_mx_tp_<n>``) and the column slot (``_mx_tq_<n>``; 0 until resolved, -1 when the read stays pandas, else one
  plus the slot) and are
  reset with its per-policy caches. A read then uses the held table borrowed: no reference count of the frame,
  of the Data space's cached value or of the table is touched per read;
- in any other class, the receiver is evaluated and its PreparedTable looked up by identity once per call of the
  formula (the lookup takes references; the reads themselves do not).

Either way an assignment of a new frame is a new object: its PreparedTable is prepared afresh and no prepared copy
of an older frame is used for it. A read the table cannot answer natively (the frame not prepared: a non-unique
index, NaN labels; the column absent or of another kind) runs the original pandas expression.

Rule: a recogniser or emitter never raises on a shape it does not recognise (a loop target, a slice, a literal of
another spelling): it returns None, and the expression stays pandas (with a report line where the form records one).
Every unpacking or indexing of a syntax node is preceded by the check of its shape.
"""

import re

import libcst as cst

from .holds import hold_table
from .read_notes import RAISES, normal, only_raised

PROVIDER = 'modelx_cython.runtime._tables'
COLUMN_KIND = {'f64': 0, 'i64': 1}
READ = ('f64', 'i64')
CTYPE = {'f64': '_mx_cy.double', 'i64': '_mx_cy.longlong'}
MAX_PARTS = 6
# displays whose value as a key part (or an argument of asof/searchsorted) selects several rows, never one label
SEVERAL_ROWS = (cst.List, cst.ListComp, cst.Set, cst.Slice, cst.Tuple, cst.Lambda)
SEVERAL_ROWS_WHY = 'a key part selects several rows'
ONLY_RAISED_WHY = 'every evaluation of this place raised in the sample (no evidence of a single-cell read)'
# kinds of a call's result in the sample (read_notes.kind) that are a single number
ONE_NUMBER = frozenset(('int', 'float', 'bool'))
MUTATORS = frozenset(('__setitem__', '__delitem__', 'update', 'insert', 'pop'))


def _code(node):
    return cst.Module(body=()).code_for_node(node)


class _ReadSources(cst.CSTVisitor):
    """Source text of the expressions the sample recorder can observe as table reads."""

    def __init__(self):
        self.texts = set()

    def _expression(self, node):
        self.texts.add(normal(_code(node)))

    visit_Subscript = visit_Call = visit_Comparison = _expression

    def visit_For(self, node):
        self._expression(node.iter)

    visit_CompFor = visit_For


def _literal_str(node):
    if isinstance(node, cst.SimpleString):
        value = node.evaluated_value
        return value if type(value) is str else None
    return None


def _text(node):
    """A text literal (T-5c: never cast to a C number; pandas raises its own TypeError)."""
    return isinstance(node, (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString))


def _cast(type_expr, value):
    return cst.Call(func=cst.Attribute(cst.Name('_mx_cy'), cst.Name('cast')),
                    args=[cst.Arg(cst.parse_expression(type_expr)), cst.Arg(value)])


def _parts(key):
    """The parts of a key: the elements of a tuple display, else the key itself."""
    return [e.value for e in key.elements] if isinstance(key, cst.Tuple) else [key]


def _index_parts(subscript):
    parts = []
    for element in subscript.slice:
        if not isinstance(element.slice, cst.Index) or element.slice.star is not None:
            return None
        parts.append(element.slice.value)
    return parts


class _Hold:
    """One (receiver, column, kind) a formula reads natively; ``table`` (and ``state`` in items) are the fields of
    the receiver's binding, shared by every read of that receiver."""

    def __init__(self, table, state, receiver, label, kind):
        self.table, self.state, self.receiver, self.label, self.kind = table, state, receiver, label, kind


class FrameReads:
    """``lower(node)`` returns the formula with recognised reads lowered; ``lowered`` lists what was lowered
    (one dict per read) and ``declined`` why a read of a frame receiver was not."""

    def __init__(self, resolver, cls_info, cells, providers, policy, row_names=None, prefix_cells=None,
                 param_cells=None, *, builtin_int, builtin_zip_walk=False):
        self.param_cells = param_cells or {}    # T-5c: cell with parameters -> (names, its `T.loc[key]` expression)
        self.row_names = row_names or {}       # code of an inlined row expression -> its row cell (assignments)
        self.prefix_cells = prefix_cells or {}  # zero-argument cell of the class -> its `T.loc[key]` expression
        self.resolver, self.cls_info, self.cells = resolver, cls_info, cells
        self.providers, self.policy = providers, policy
        self.builtin_int = builtin_int
        self.builtin_zip_walk = builtin_zip_walk
        self.lowered, self.declined = [], []
        self.lowered_nodes = []     # original nodes, including their nested operations
        # what the sample run saw at each table read of this formula, by the text of the node (read_notes.py)
        self.notes = (getattr(resolver, 'read_notes', None) or {}).get((cls_info.name, '_f_' + cells.name), {})
        self.lowered_text = set()   # texts of the nodes this formula's rewrite replaced
        self.param_readers = set()  # names of parameterised prefix Cells replaced by a native read
        self.key_count = 0
        self.held_string_selector = None

    def note_of(self, node):
        return self.notes.get(normal(_code(node)))

    def audit(self, native_node):
        """The table reads the sample saw in this formula that were not lowered: (expression, reason, ...)."""
        from .read_notes import scalar_table_read
        out = []
        declined = {normal(entry['expression']): entry.get('reason') for entry in self.declined}
        lowered = [normal(_code(node)) for node in self.lowered_nodes]

        inlined = {normal(code): (name if isinstance(name, str) else getattr(name, 'name', str(name)))
                   for code, name in self.row_names.items()}
        inlined.update({normal(_code(expr)): name for name, expr in self.prefix_cells.items()})
        renames = getattr(self.cells, 'vector_loop_renames', {})
        targets = re.compile(r'\b(?:' + '|'.join(map(re.escape, renames)) + r')\b') if renames else None

        def as_written(text):
            if targets is not None:
                text = targets.sub(lambda match: renames[match[0]], text)
            # A row cell inlined before this pass was a `self.<cell>()` call in the sample.
            for code, name in inlined.items():
                text = text.replace(code, f'self.{name}()')
            return text
        exact = set(lowered) | {as_written(text) for text in lowered}
        replaced = _ReadSources()
        for node in self.lowered_nodes:
            node.visit(replaced)
        candidates = replaced.texts | self.lowered_text
        candidates |= {as_written(text) for text in candidates}
        native_sources = None
        declined.update({as_written(e): why for e, why in declined.items()})
        for text, note in sorted(self.notes.items()):
            reads = note['table_reads']
            if not reads or text in exact:
                continue
            if text in candidates:
                # A nested read can survive as a key or as part of a lowered loop body.
                if native_sources is None:
                    source = _ReadSources()
                    native_node.visit(source)
                    native_sources = source.texts | {as_written(text) for text in source.texts}
                if text not in native_sources:
                    continue
            reason = declined.get(normal(text[:160]))
            operation = reason is None and not any(scalar_table_read(op, o) for op, o in reads)
            if reason is None and self.cells.name in self.prefix_cells:
                reason = ('a row or sub-table cell: its readers take the row directly; the cell itself runs on pandas '
                          'only when its own value is asked for')
            entry = {'expression': text[:200], 'seen': sorted({repr(o) for _, o in reads}),
                     'evaluations': note['table_count'], 'per_step': bool(self.cells.has_args()),
                     'reason': reason or 'no rule recognises this form'}
            if operation:
                entry['pandas_operation'] = True
            out.append(entry)
        # a read of a table the text shows but the sample never reached: declined from the text alone
        for entry in self.declined:
            text = normal(entry['expression'])
            if entry.get('form') == 'pandas' and text not in self.notes and as_written(text) not in self.notes:
                out.append({'expression': text[:200], 'seen': [], 'evaluations': 0, 'per_step': bool(self.cells.has_args()),
                            'reason': 'not reached by the sample; ' + (entry.get('reason') or 'left on pandas')})
        # every other read this pass declined (left on pandas with a reason) that the sample reached: also reported,
        # so no compile-time choice of pandas is silent (a note that is not a table read by its observations, or a
        # text inside a lowered expression)
        reported = {normal(e['expression']) for e in out}
        for entry in self.declined:
            text = normal(entry['expression'])
            if entry.get('form') != 'pandas' or text[:200] in reported or text in exact:
                continue
            note = self.notes.get(text) or self.notes.get(as_written(text))
            if note is None:
                continue
            reported.add(text[:200])
            out.append({'expression': text[:200], 'seen': sorted(repr(o) for o in note['seen']),
                        'evaluations': note['count'], 'per_step': bool(self.cells.has_args()),
                        'reason': entry.get('reason') or 'left on pandas'})
        self.unsampled_lowered = sum(1 for text in lowered if text not in self.notes
                                     and as_written(text) not in self.notes)
        return out

    # ------------------------------------------------------------------ facts
    def _facts(self, node):
        """The key-typing facts (is_int, exact_long_long, enum constants) for this formula."""
        from .key_facts import KeyFacts, _Formula
        facts = KeyFacts(self.resolver, self.cls_info, self.cells, self.policy, builtin_int=self.builtin_int)
        facts.formula = _Formula(node)
        node.visit(facts.formula)
        facts._params = set(getattr(self.cells, 'params', ()) or ())
        plain = facts._int_index_read

        def int_index_read(expr):       # T-5c: also on a traced pandas frame (`top = T.index.max()[0]`)
            return plain(expr) or (isinstance(expr, cst.Subscript) and self._extreme_part_int(expr))
        facts._int_index_read = int_index_read
        plain_int = facts.is_int

        def is_int(expr, depth=0):      # T-5c: `max(a for a, s in T.index)` of an int level
            return plain_int(expr, depth) or (
                'extreme of an integer level' if isinstance(expr, cst.Call) and self._extreme_part_int(expr) else None)
        facts.is_int = is_int
        return facts

    def _extreme_part_int(self, expr):
        """``R.index.max()[k]`` / ``.min()[k]``, R a unique frame of several levels whose level k is int; or
        ``max(a for a, s in R.index)`` with ``a`` naming an int level (``level_walk_extreme``)."""
        if (isinstance(expr, cst.Call) and isinstance(expr.func, cst.Name) and expr.func.value in ('max', 'min')
                and len(expr.args) == 1 and isinstance(expr.args[0].value, cst.GeneratorExp)):
            gen = expr.args[0].value
            target, it = gen.for_in.target, gen.for_in.iter
            if (gen.for_in.ifs or gen.for_in.inner_for_in is not None or not isinstance(gen.elt, cst.Name)
                    or not isinstance(target, cst.Tuple) or not isinstance(it, cst.Attribute) or it.attr.value != 'index'
                    or not all(isinstance(e.value, cst.Name) for e in target.elements)):
                return False
            names = [e.value.value for e in target.elements]
            found = self._receiver(it.value)
            if names.count(gen.elt.value) != 1 or found is None or not found[1].get('unique'):
                return False
            levels = found[1]['levels']
            return len(levels) == len(names) and levels[names.index(gen.elt.value)] == 'int'
        if not (isinstance(expr, cst.Subscript) and len(expr.slice) == 1 and isinstance(expr.slice[0].slice, cst.Index)
                and isinstance(expr.slice[0].slice.value, cst.Integer) and isinstance(expr.value, cst.Call)
                and not expr.value.args and isinstance(expr.value.func, cst.Attribute)
                and expr.value.func.attr.value in ('max', 'min') and isinstance(expr.value.func.value, cst.Attribute)
                and expr.value.func.value.attr.value == 'index'):
            return False
        found = self._receiver(expr.value.func.value.value)
        if found is None or not found[1].get('unique'):
            return False
        levels, level = found[1]['levels'], int(expr.slice[0].slice.value.value, 0)
        return 1 < len(levels) <= 8 and 0 <= level < len(levels) and levels[level] == 'int'

    def _once(self, name):
        """``name`` is bound once by an assignment of the formula (not a parameter, a loop target or another binding)."""
        f = self.facts.formula
        return f.binds.get(name, 0) == 1 and name not in f.other and name not in self.facts._params

    def _value_of(self, name):
        """The value of the local ``name``: the one value of a local bound once, or (T-5c) the call of a local whose
        every assignment is the same zero-argument call (``tbl = data.T()`` in several branches), else None."""
        values = self.facts.formula.values.get(name) or ()
        if len(values) == 1 and self._once(name):
            return values[0]
        f = self.facts.formula
        if (len(values) > 1 and f.binds.get(name, 0) == len(values) and name not in f.other
                and name not in self.facts._params and all(isinstance(v, cst.Call) and not v.args for v in values)
                and len({_code(v) for v in values}) == 1):
            return values[0]
        return None

    @staticmethod
    def _simple_assign(statement):
        """(name, value) of a statement ``name = value``, else None."""
        if (isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1
                and isinstance(statement.body[0], cst.Assign) and len(statement.body[0].targets) == 1
                and isinstance(statement.body[0].targets[0].target, cst.Name)):
            return statement.body[0].targets[0].target.value, statement.body[0].value
        return None

    def _schema_of_call(self, expr):
        if not (isinstance(expr, cst.Call) and not expr.args):
            return None
        from .usage import RESOLVED, _extract_self_chain
        chain = _extract_self_chain(expr.func)
        if not isinstance(chain, list) or not chain:
            return None
        status, name = self.resolver.resolve_chain(self.cls_info.fqname, chain)
        if status != RESOLVED or name not in getattr(self.resolver, 'cached_cells', ()):
            return None
        return getattr(self.resolver, 'frame_schemas', {}).get(name)

    def _receiver(self, expr):
        """(cell call, schema) when ``expr`` is a frame receiver, else None. In an item class a frame or row the item
        computes itself is prepared per item: it is one only when the reading formula is evaluated at least
        ``PREPARE_MIN_EVALUATIONS`` times per item in the sample (T-5c), so that the preparation pays for itself;
        a formula evaluated fewer times reads it on pandas."""
        schema = self._schema_of_call(expr)
        found = (expr, schema) if schema is not None else None
        if found is None:
            value = self._value_of(expr.value) if isinstance(expr, cst.Name) else None
            schema = self._schema_of_call(value) if value is not None else None
            found = (value, schema) if schema is not None else None
        if found is not None and getattr(self, 'per_item', False) and self.published(found[0]) is None:
            evaluations = getattr(getattr(self.cells, '_rt', None), 'max_evaluations_per_receiver', 0) or 0
            if evaluations < PREPARE_MIN_EVALUATIONS:
                return None
        return found

    def param_prefix(self, expr):
        """T-5c (shape G): ``self.P(a, ...)`` with P a cell of parameters whose formula is ``return T.loc[key]`` (T a
        table the reads may read natively): (T call, T schema, the key's parts with the arguments in place of the
        parameters) when they are fewer than T's levels, else None. Compiled cells with parameters cannot be
        assigned, so P's value is always its formula's."""
        if not (isinstance(expr, cst.Call) and isinstance(expr.func, cst.Attribute)
                and isinstance(expr.func.value, cst.Name) and expr.func.value.value == 'self'):
            return None
        entry = self.param_cells.get(expr.func.attr.value)
        if entry is None or any(a.keyword is not None or a.star for a in expr.args) or len(expr.args) != len(entry[0]):
            return None
        params, row = entry
        found = self._receiver(row.value.value)
        if found is None or found[1]['kind'] != 'frame' or not found[1].get('unique') or _code(found[0]) in self.mutated:
            return None
        arguments = {name: arg.value if isinstance(arg.value, (cst.Name, cst.Integer, cst.SimpleString))
                     else arg.value.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
                     for name, arg in zip(params, expr.args)}        # an argument keeps its precedence

        class Put(cst.CSTTransformer):
            def leave_Name(self, original, updated):
                return arguments.get(original.value, updated)

            def leave_Attribute(self, original, updated):       # x.p is not the parameter p
                return updated.with_changes(attr=original.attr)
        parts = _parts(row.slice[0].slice.value.visit(Put()))
        if len(parts) >= len(found[1]['levels']) or any(
                isinstance(p, (cst.Slice, cst.List, cst.ListComp, cst.Tuple, cst.StarredElement)) for p in parts):
            return None
        return found[0], found[1], parts

    def prefix_receiver(self, expr):
        """T-4 (L3 through a cell): ``self.P()`` (or a local assigned once from it) where the formula of P is
        ``return T.loc[prefix]``, T a frame receiver and every prefix part fixed per item: (P call, T call, T's
        schema, prefix parts, P). The keyed reads of P's value read T with the prefix held as codes."""
        call = self._value_of(expr.value) if isinstance(expr, cst.Name) else expr
        if not (isinstance(call, cst.Call) and not call.args and isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name) and call.func.value.value == 'self'):
            return None
        name = call.func.attr.value
        row = self.prefix_cells.get(name)
        if row is None or not self.cached_call(call):
            return None
        found = self._receiver(row.value.value)
        if found is None or found[1]['kind'] != 'frame':
            return None
        key = row.slice[0].slice.value
        parts = _parts(key)
        from .row_cells import item_constant
        if len(parts) >= len(found[1]['levels']) or not all(self.fixed_part(p) for p in parts):
            return None
        # by traced type: every traced value of P is a frame of the parent's remaining levels and of its columns
        mine = self._schema_of_call(call)
        parent = found[1]
        if not (mine and mine['kind'] == 'frame' and mine['levels'] == parent['levels'][len(parts):]
                and dict(mine['columns']).items() <= dict(parent['columns']).items()):
            self.declined.append({'expression': _code(call), 'receiver': _code(call), 'form': 'pandas',
                                  'reason': 'a prefix cell whose traced value is not the sub-frame of its table'})
            return None
        return call, found[0], found[1], parts, name

    # ------------------------------------------------------------------ driver
    def lower(self, node):
        params = node.params.params
        if not params or params[0].name.value != 'self' or not getattr(self.resolver, 'frame_schemas', None):
            return node
        self.facts = self._facts(node)
        plain = self.facts._int_index_read

        def int_index_read(expr):       # T-4 (L5): extremes of a one-level int frame index are C integers
            return plain(expr) or self.extreme(expr) is not None or self.int_label(expr)
        self.facts._int_index_read = int_index_read
        from .helpers import module_name
        self.module = module_name(PROVIDER)
        self.per_item = (self.cls_info.name in self.cls_info.visitor.itemspace_classes
                         and not getattr(self.cls_info, 'collect_shared_item', False))
        self.holds = {}         # (receiver code, label, kind) -> _Hold
        self.bindings = {}      # receiver code -> its binding (_Hold of the table field)
        self.code_fields, self.row_fields = {}, {}
        self.tuple_guards = {}  # T-5c: code of a part ``k[i]`` of a key tuple held at run time -> (code of k, count)
        self.mutated = self._mutated(node)
        self.asof_locals = self._asof_locals(node)
        self.views = {}         # T-5: view local -> _View
        self.ambiguous = set()  # T-5: codes of `T.loc[a, b]` prefixes (row prefix or row and column)
        self.columns = {}       # T-5: column local -> (view expression, column label)
        self.row_locals = self._row_locals(node)
        self.enum_key_locals = self._enum_key_locals()
        self.declarations = []  # T-5: C Py_ssize_t locals of the formula (loop positions)
        self.local_count = 0
        rewriter = _Rewrite(self)
        if self.builtin_zip_walk and not getattr(self.policy, 'verify_inputs', False):
            zipped = rewriter.zip_prefix(node.body.body)
            if zipped is not None:
                self.providers.add(PROVIDER)
                return node.with_changes(body=node.body.with_changes(body=zipped))
        body = []
        changed = False
        for statement in node.body.body:
            before = len(self.holds)
            rewriter.new_holds = self.new_items = []
            asof = self._asof_assignment(statement)
            view = self.view_assignment(statement) if asof is None else None
            if asof is not None or view is not None:    # the whole assignment is rewritten: its read is lowered
                assigned = self._simple_assign(statement)
                if assigned is not None:
                    self.lowered_text.add(normal(_code(assigned[1])))
            if asof is not None:
                new = rewriter.asof_statement(statement, asof)
            elif view is not None:
                new = rewriter.view_statement(statement, view)
            else:
                new = statement.visit(rewriter)
                column = self.column_assignment(statement)
                if column is not None:      # T-5 shape B: a column of a view in a local
                    source = column[1]
                    if isinstance(source, tuple):       # `R.loc[prefix, "c"]`: a view local of R.loc[prefix] first
                        _, prefix, ((call, schema), parts) = source
                        vname = f'_mx_cv_{column[0]}'
                        synthetic = rewriter._assign(vname, prefix)
                        new = [*rewriter.view_statement(synthetic, (vname, call, schema, parts, None)), new]
                        source = cst.Name(vname)
                    self.columns[column[0]] = (source, column[2])
                    rewriter.swaps[column[0]] = statement.body[0].value
            if self.new_items:
                body.extend(self._hold_statements(item) for item in self.new_items)
            changed = changed or new is not statement or len(self.holds) != before
            body.append(new)
        if not changed:
            return node
        for view in self.views.values():
            if not view.used:       # no use needed the view's codes or rows: only the assignment stays
                view.setup[:] = view.setup[-1:]
            elif isinstance(value := view.setup[-1].body[0].value, cst.Call) and self.param_prefix(value):
                self.param_readers.add(value.func.attr.value)
        body = self._drop_unused_locals(body, rewriter.replaced_names | set(rewriter.key_locals) | set(rewriter.swaps))
        if self.declarations:
            from .key_facts import _is_docstring
            head = 1 if body and _is_docstring(body[0]) else 0
            body[head:head] = [cst.parse_statement(f'{name}: _mx_cy.Py_ssize_t') for name in self.declarations]
        self.providers.add(PROVIDER)
        lowered = self._flatten(body)
        from .key_facts import _is_docstring
        original = [s for s in node.body.body if not _is_docstring(s)]
        if (rewriter.held_string_read is not None and len(original) == 1
                and isinstance(original[0], cst.SimpleStatementLine) and len(original[0].body) == 1
                and isinstance(original[0].body[0], cst.Return)
                and isinstance(lowered[-1], cst.SimpleStatementLine) and isinstance(lowered[-1].body[0], cst.Return)):
            read, test, state = rewriter.held_string_read
            expr = lowered[-1].body[0].value
            if isinstance(expr, cst.IfExp) and expr.body.deep_equals(read) and expr.test.deep_equals(test):
                self.held_string_selector = f'self.{state} > 0 and ({_code(test)})'
        return node.with_changes(body=node.body.with_changes(body=lowered))

    def _asof_assignment(self, statement):
        assign = self._simple_assign(statement)
        return self.asof_locals.get(assign[0]) if assign else None

    def _asof_locals(self, node):
        """T-4: ``lo = P.index.asof(x)`` at the top level of an item formula, P the value of a prefix cell with one
        remaining int level, and every load of ``lo`` the key of a keyed read ``P.loc[lo, "c"]`` of that value:
        {name: (P name, x)}. ``lo`` is then the row of the label in the parent table, never a Python value."""
        if not self.per_item:
            return {}
        f, found = self.facts.formula, {}
        for statement in node.body.body:
            name, value = self._simple_assign(statement) or (None, None)
            if not (isinstance(value, cst.Call) and len(value.args) == 1 and not value.args[0].keyword
                    and isinstance(value.func, cst.Attribute) and value.func.attr.value == 'asof'
                    and isinstance(value.func.value, cst.Attribute) and value.func.value.attr.value == 'index'):
                continue
            if not self._once(name):
                continue
            prefix = self.prefix_receiver(value.func.value.value)
            if prefix is None or len(prefix[2]['levels']) - len(prefix[3]) != 1 or prefix[2]['levels'][-1] != 'int':
                continue
            uses = [use for use, nested in f.loads.get(name, ()) if not nested]
            if len(uses) != len(f.loads.get(name, ())) or not uses or not all(
                    self._keyed_use(use, prefix[4]) for use in uses):
                continue
            found[name] = (prefix, value.args[0].value, value)
        return found

    def _keyed_use(self, use, cell):
        """``use`` (a Name load) is the first key part of ``X.loc[use, "c"]`` with X the value of prefix cell ``cell``."""
        parents = self.facts.formula.parents
        index = parents.get(id(use))
        element = parents.get(id(index)) if isinstance(index, cst.Index) else None
        subscript = parents.get(id(element)) if isinstance(element, cst.SubscriptElement) else None
        parts = _index_parts(subscript) if isinstance(subscript, cst.Subscript) else None
        if not (parts is not None and len(parts) == 2 and subscript.slice[0] is element
                and _literal_str(parts[1]) is not None and isinstance(subscript.value, cst.Attribute) and subscript.value.attr.value == 'loc'):
            return False
        prefix = self.prefix_receiver(subscript.value.value)
        return prefix is not None and prefix[4] == cell

    def _root(self, expr):
        while isinstance(expr, (cst.Attribute, cst.Subscript)):
            expr = expr.value
        return expr

    def _mutated(self, node):
        """Codes of the receivers the formula itself edits in place (an assignment or ``del`` through them,
        ``__setitem__``/``__delitem__``/``update``/``insert``/``pop``, or any call with ``inplace``): their
        reads stay pandas, so an edit is read as it is made, as live."""
        owner, found = self, set()

        def note(expr):
            got = owner._receiver(owner._root(expr))
            if got is not None:
                found.add(_code(got[0]))

        class Scan(cst.CSTVisitor):
            def visit_Assign(self, n):
                for target in n.targets:
                    if isinstance(target.target, (cst.Attribute, cst.Subscript)):
                        note(target.target)

            def visit_AugAssign(self, n):
                if isinstance(n.target, (cst.Attribute, cst.Subscript)):
                    note(n.target)

            def visit_AnnAssign(self, n):
                if isinstance(n.target, (cst.Attribute, cst.Subscript)):
                    note(n.target)

            def visit_Del(self, n):
                note(n.target)

            def visit_Call(self, n):
                if isinstance(n.func, cst.Attribute) and (
                        n.func.attr.value in MUTATORS
                        or any(a.keyword is not None and a.keyword.value == 'inplace' for a in n.args)):
                    note(n.func.value)
        node.visit(Scan())
        return found

    @staticmethod
    def _flatten(body):
        out = []
        for item in body:
            out.extend(item if isinstance(item, list) else [item])
        return out

    def _drop_unused_locals(self, body, replaced):
        """A frame local whose every use was lowered is no longer evaluated (no Cell call, no reference). Repeated
        until nothing more is dropped: dropping a view's assignment can leave its frame local unused (T-5c)."""
        out = self._flatten(body)
        while True:
            loads = {}

            class Loads(cst.CSTVisitor):
                def visit_Name(self, name):
                    loads[name.value] = loads.get(name.value, 0) + 1
            cst.Module(body=out).visit(Loads())
            kept = []
            for statement in out:
                name, value = self._simple_assign(statement) or (None, None)
                if name in replaced and loads.get(name, 0) == 1:     # the target itself only
                    self.lowered_text.add(normal(_code(value)))
                else:
                    kept.append(statement)
            if len(kept) == len(out):
                return out
            out = kept

    def _hold_statements(self, hold):
        """The statements that resolve one hold, placed before the top-level statement of its first read."""
        m = self.module
        if isinstance(hold, tuple):         # T-4: a held code or row, after the hold of its table
            what, field, table, level_or_key, extra = hold
            if what == 'code':
                value = f'{m}.hold_code(self.{table}, {level_or_key}, {_code(extra)})'
                if _code(extra) in self.tuple_guards:      # T-5c: -1 (the reads run on pandas) when not a tuple
                    key, count = self.tuple_guards[_code(extra)]
                    value = f'({value} if type({key}) is tuple and len({key}) == {count} else -1)'
            elif what == 'range':       # T-5: the stored rows of a prefix held per item, packed, plus 2 (0: unresolved)
                value = f'{level_or_key} + 2'
            else:
                value = f'{m}.hold_row(self.{table}, {_code(level_or_key)}, False)'
            return [cst.parse_statement(f'if self.{field} == 0:\n    self.{field} = {value}\n')]
        # T-4.2: one binding per receiver and item (or per call outside items): the frame's PreparedTable, or a
        # table whose every slot is -1; each read then takes its slot from the table by its signature id
        function = 'bind_ref_checked' if getattr(self.policy, 'verify_inputs', False) else 'bind_ref'
        receiver = _code(hold.receiver)
        if self.per_item:
            held = self.published(hold.receiver)
            if held is None and function == 'bind_ref':     # T-5c: a frame the item computes: prepared for it alone
                function = 'bind_own'
            value = held if held is not None and function == 'bind_ref' else f'{m}.{function}({receiver})'
            return [cst.parse_statement(
                f'if self.{hold.state} == 0:\n'
                f'    self.{hold.table} = {value}\n'
                f'    self.{hold.state} = 1\n')]
        return [cst.parse_statement(f'{hold.table}: object = {m}.{function}({receiver})')]

    # ------------------------------------------------------------------ T-5: views of a prefix
    def view_assignment(self, statement):
        """T-5: ``name = R.loc[prefix]`` (or ``name = self.P()``, P a prefix cell) at the top level of the formula,
        ``name`` bound once and never in a nested function, R a frame receiver of more levels than the prefix:
        (name, R call, R schema, prefix parts, prefix cell name or None); else None."""
        name, value = self._simple_assign(statement) or (None, None)
        if name is None or not self._once(name) or any(nested for _, nested in self.facts.formula.loads.get(name, ())):
            return None
        prefix = self.prefix_receiver(value) if isinstance(value, cst.Call) else None
        if prefix is not None:
            return name, prefix[1], prefix[2], prefix[3], prefix[4]
        param = self.param_prefix(value)        # T-5c: ``name = self.P(a)``, P a prefix cell of parameters
        if param is not None:
            return name, param[0], param[1], param[2], None
        # T-5c: ``name = self.P().loc[more]``, a prefix of a prefix cell's value: one view of the parent table, the
        # cell's parts held per item and ``more`` coded per call
        inner = (value.value.value if isinstance(value, cst.Subscript) and isinstance(value.value, cst.Attribute)
                 and value.value.attr.value == 'loc' else None)
        prefix = self.prefix_receiver(inner) if isinstance(inner, cst.Call) else None
        extra = _index_parts(value) if prefix is not None else None
        if extra is not None and len(extra) == 1:
            more = _parts(extra[0])
            if len(prefix[3]) + len(more) < len(prefix[2]['levels']) and not any(
                    isinstance(p, (cst.Slice, cst.List, cst.ListComp, cst.Tuple, cst.StarredElement)) for p in more):
                return name, prefix[1], prefix[2], [*prefix[3], *more], prefix[4], len(prefix[3])
        got = self.prefix_of(value)
        if got is None:
            return None
        found, parts = got
        return name, found[0], found[1], parts, None

    def column_assignment(self, statement):
        """``name = X["c"]`` at the top level, X a view local, a frame receiver or ``R.loc[prefix]``, ``name`` bound
        once and only walked (``for``, a comprehension, ``.items()``) or read whole (``.tolist()``, list, tuple,
        len): (name, X, "c"), else None."""
        name, value = self._simple_assign(statement) or (None, None)
        if name is None:
            return None
        parts = _index_parts(value) if isinstance(value, cst.Subscript) else None
        label = _literal_str(parts[0]) if parts and len(parts) == 1 else None
        f = self.facts.formula
        source = value.value if isinstance(value, cst.Subscript) else None
        if (label is None and parts and len(parts) >= 2 and isinstance(source, cst.Attribute)
                and source.attr.value == 'loc' and _literal_str(parts[-1]) is not None):
            # `R.loc[prefix, "c"]`: the column of the view R.loc[prefix]
            label = _literal_str(parts[-1])
            prefix = value.with_changes(slice=value.slice[:-1])
            if len(prefix.slice) == 1:
                prefix = prefix.with_changes(slice=[prefix.slice[0].with_changes(comma=cst.MaybeSentinel.DEFAULT)])
            got = self.prefix_of(prefix)
            if got is None:
                return None
            source = ('view', prefix, got)
        if label is None or not self._once(name):
            return None
        if isinstance(source, tuple):
            pass
        elif not ((isinstance(source, cst.Name) and source.value in self.views) or self._receiver(source) is not None
                  or self.prefix_of(source) is not None):
            return None
        for use, nested in f.loads.get(name, ()):
            up = f.parents.get(id(use))
            if nested:
                return None
            if isinstance(up, (cst.CompFor, cst.For)) and up.iter is use:
                continue
            if isinstance(up, cst.Attribute) and up.attr.value in ('items', 'tolist'):
                call = f.parents.get(id(up))
                if isinstance(call, cst.Call) and not call.args:
                    continue
            if isinstance(up, cst.Attribute) and up.attr.value in ('iloc', 'loc', 'index'):
                continue        # read by position or label, or its index walked (else pandas on the column)
            if isinstance(up, cst.Arg):
                call = f.parents.get(id(up))
                if isinstance(call, cst.Call) and isinstance(call.func, cst.Name) and call.func.value in (
                        'list', 'tuple', 'len') and len(call.args) == 1:
                    continue
            return None
        return name, source, label

    def int_label(self, expr):
        """T-5: ``X.index[i]``, ``X.index.asof(x)``, ``.max()``, ``.min()`` or ``.searchsorted(x)`` of a view or frame
        whose one remaining level is an int level: an integer (key facts)."""
        if isinstance(expr, cst.Subscript):
            index = expr.value
        elif isinstance(expr, cst.Call) and isinstance(expr.func, cst.Attribute) and expr.func.attr.value in (
                'asof', 'max', 'min', 'searchsorted'):
            index = expr.func.value
        else:
            return False
        if not (isinstance(index, cst.Attribute) and index.attr.value == 'index'):
            return False
        receiver = index.value
        if isinstance(receiver, cst.Name) and receiver.value in getattr(self, 'views', {}):
            view = self.views[receiver.value]
            schema, depth = view.schema, view.depth
        else:
            found = self._receiver(receiver)
            prefix = self.prefix_of(receiver) if found is None else None
            if found is None and prefix is None:
                return False
            schema, depth = (found[1], 0) if found is not None else (prefix[0][1], len(prefix[1]))
        return len(schema['levels']) - depth == 1 and (schema['levels'][depth] == 'int' or (
            isinstance(expr, cst.Call) and expr.func.attr.value == 'searchsorted'))

    def prefix_of(self, value):
        """``R.loc[prefix]`` with R a frame receiver (unique index, not edited) of more levels than the prefix:
        ((R call, schema), prefix parts), else None."""
        if not (isinstance(value, cst.Subscript) and isinstance(value.value, cst.Attribute)
                and value.value.attr.value == 'loc'):
            return None
        parts = _index_parts(value)
        found = self._receiver(value.value.value)
        if parts is None or found is None or not found[1].get('unique') or _code(found[0]) in self.mutated:
            return None
        if len(parts) == 1 and isinstance(parts[0], cst.Tuple):
            if any(isinstance(e, cst.StarredElement) for e in parts[0].elements):
                return None
            parts = [e.value for e in parts[0].elements]
        elif len(parts) > 1 and _literal_str(parts[-1]) is not None:
            return None                     # a key and a column (L1)
        levels = found[1]['levels']
        if not 0 < len(parts) < len(levels) or any(isinstance(p, (cst.Slice, cst.List, cst.ListComp, cst.Tuple))
                                                   for p in parts):
            return None
        if len(_index_parts(value)) > 1:
            if len(parts) != 2:
                return None
            self.ambiguous.add(_code(value))    # `T.loc[a, b]`: a row prefix only by the table's verdict (kind -5)
        return found, parts

    def _row_locals(self, node):
        """T-5: ``row = R.loc[key]`` (at any depth, ``row`` bound once), R a frame receiver of a unique index and the
        key complete, every load of ``row`` a read of a literal column ``row["c"]``: {name: (R call, schema)}."""
        f, found = self.facts.formula, {}

        class Scan(cst.CSTVisitor):
            def visit_FunctionDef(inner, n):
                return n is node

            def visit_Lambda(inner, n):
                return False

            def visit_Assign(inner, n):
                if not (len(n.targets) == 1 and isinstance(n.targets[0].target, cst.Name)
                        and isinstance(n.value, cst.Subscript) and isinstance(n.value.value, cst.Attribute)
                        and n.value.value.attr.value == 'loc'):
                    return
                name = n.targets[0].target.value
                parts = _index_parts(n.value)
                got = self._receiver(n.value.value.value)
                if (got is None or parts is None or len(parts) != 1 or got[1]['kind'] != 'frame'
                        or not got[1].get('unique') or _code(got[0]) in self.mutated or not self._once(name)):
                    return
                levels = got[1]['levels']
                key = parts[0]
                if (len(levels) == 1) == isinstance(key, cst.Tuple) or (
                        isinstance(key, cst.Tuple) and len(key.elements) != len(levels)):
                    return
                if field_reads(name, got[1]):
                    found[name] = (got[0], got[1], n)

            def visit_For(inner, n):     # T-5 shape D: `for k, row in R.iterrows():`, R of one level
                it, target = n.iter, n.target
                if not (isinstance(it, cst.Call) and not it.args and isinstance(it.func, cst.Attribute)
                        and it.func.attr.value == 'iterrows' and isinstance(target, cst.Tuple)
                        and len(target.elements) == 2
                        and all(isinstance(e, cst.Element) and isinstance(e.value, cst.Name) for e in target.elements)):
                    return      # (`for k, *row in ...` is not this form)
                got = self._receiver(it.func.value)
                key, name = (e.value.value for e in target.elements)
                if (got is None or got[1]['kind'] != 'frame' or not got[1].get('unique') or len(got[1]['levels']) != 1
                        or _code(got[0]) in self.mutated or key == name or f.binds.get(name, 0) != 1
                        or name in self.facts._params):     # (a loop target is an `other` binding)
                    return
                if field_reads(name, got[1]):
                    found[name] = (got[0], got[1], n)

        def field_reads(name, schema):
            """Every load of ``name`` reads a literal column, or a computed one under float()."""
            columns = dict(schema['columns'])
            for use, nested in f.loads.get(name, ()):
                sub = f.parents.get(id(use))
                parts = _index_parts(sub) if isinstance(sub, cst.Subscript) else None
                if nested or parts is None or sub.value is not use or len(parts) != 1:
                    return False
                if _literal_str(parts[0]) in columns:
                    continue
                up = f.parents.get(id(sub))
                call = f.parents.get(id(up)) if isinstance(up, cst.Arg) else None
                if not (isinstance(call, cst.Call) and isinstance(call.func, cst.Name) and call.func.value == 'float'
                        and len(call.args) == 1 and _literal_str(parts[0]) is None):
                    return False
            return True
        node.visit(Scan())
        return found

    def position_name(self):
        name = f'_mx_p{self.local_count}'
        self.local_count += 1
        return name

    def published(self, receiver):
        """T-4.2: ``X._mx_prepared_<name>()`` when the receiver ``X.<name>()`` is a frame cell of a class without
        items in this module, which publishes its prepared table (one field read per item, as main's held handle);
        else None (the binding prepares or looks the frame up)."""
        provider = self.published_provider(receiver)
        if provider is not None:
            _, name = provider
            return f'{_code(receiver.func.value)}._mx_prepared_{name}()'
        return None

    def published_provider(self, receiver):
        """The same-module Data class and Cell that publish this frame."""
        if not (isinstance(receiver, cst.Call) and not receiver.args and isinstance(receiver.func, cst.Attribute)):
            return None
        from .usage import RESOLVED, _extract_self_chain
        from .table_cells import data_frame_cells
        chain = _extract_self_chain(receiver.func)
        if not isinstance(chain, list) or not chain:
            return None
        status, fqname = self.resolver.resolve_chain(self.cls_info.fqname, chain)
        if status != RESOLVED:
            return None
        for owner in self.cls_info.module.classes.values():
            for name, cells in owner.cells.items():
                if cells.fqname == fqname and name in data_frame_cells(owner):
                    return owner, name
        return None

    def read_signature(self, hold):
        """T-4 item 0 / T-4.2: the id of the kinds a read is compiled for (index levels; every column for a row read by
        field; the column read and its kind), whose slot each prepared table decides once."""
        schema = self._schema_of_call(hold.receiver) or {}
        levels = ''.join('i' if kind == 'int' else 'o' for kind in schema.get('levels', ()))
        kinds = {'f64': 0, 'i64': 1, 'obj': 2}
        columns = (tuple((name, kinds[kind]) for name, kind in schema.get('columns', ()))
                   if getattr(hold, 'row_columns', False) else None)
        label = hold.label if hold.label is not _NO_LABEL else None
        return self.signature(levels, columns, label, hold.kind)

    def extreme(self, expr):
        """``R.index.max()`` / ``R.index.min()`` on a frame receiver with one unique int level: (R call, 'max'|'min')."""
        if not (isinstance(expr, cst.Call) and not expr.args and isinstance(expr.func, cst.Attribute)
                and expr.func.attr.value in ('max', 'min') and isinstance(expr.func.value, cst.Attribute)
                and expr.func.value.attr.value == 'index'):
            return None
        found = self._receiver(expr.func.value.value)
        if found is None or found[1]['levels'] != ('int',) or not found[1].get('unique') or _code(found[0]) in self.mutated:
            return None
        return found[0], expr.func.attr.value

    def signature(self, levels, columns, label, kind):
        """The model-wide id of a read signature (written to ``_mx_table_signatures.py`` at the end of the build)."""
        table = self.resolver.__dict__.setdefault('table_signatures', {})
        return table.setdefault((levels, columns, label, kind), len(table))

    def cached_call(self, call):
        """A self-rooted zero-argument call of a cached Cell (as the transformer's ``_cached_call``)."""
        name = call.func.attr.value if isinstance(call.func, cst.Attribute) else None
        return name is not None and any(str(c).endswith('._f_' + name)
                                        for c in getattr(self.resolver, 'cached_cells', ()))

    def item_fixed(self, part, level_kind=None):
        """A key part whose value is fixed per item (held as a code), in a per-step cell of an item class."""
        from .row_cells import item_constant
        if isinstance(part, cst.Integer) or not (self.per_item and self.cells.has_args()):
            return False            # an integer literal is passed as it is (a C integer)
        # T-5c: a text literal, an enum member and a field of a cached row cell are fixed too, unless the read takes
        # the part as a C integer (key_part): their code is held per item instead of looked up at every read
        return item_constant(part, self.cached_call) or (
            self._fixed_value(part) and not (level_kind == 'int' and (
                self.facts.is_int(part) or self.facts.enum_constant(part) is not None)))

    def fixed_part(self, part):
        """A prefix part fixed per item: ``row_cells.item_constant``, or one of ``_fixed_value`` (T-5c)."""
        from .row_cells import item_constant
        return item_constant(part, self.cached_call) or self._fixed_value(part)

    def _fixed_value(self, part):
        """T-5c: a text literal, an enum member ``self.<Enum>.<member>``, or ``self.R()["f"]`` with R a cached
        zero-argument cell (a field of a row cell): values that do not change while the item's caches hold."""
        from .row_cells import item_constant
        if isinstance(part, cst.SimpleString) or self._enum_attribute(part):
            return True
        parts = _index_parts(part) if isinstance(part, cst.Subscript) else None
        return (parts is not None and len(parts) == 1
                and (_literal_str(parts[0]) is not None or _code(part) in self.tuple_guards)
                and isinstance(part.value, cst.Call) and item_constant(part.value, self.cached_call))

    def tuple_key(self, key, count):
        """T-5c: ``k[0]`` ... ``k[count - 1]`` for a key held in a local or returned by a cached zero-argument cell,
        whose value is a tuple of ``count`` parts at run time (checked by every read: ``guards``); a local assigned
        once from such a cell is that cell. None for another key."""
        from .row_cells import item_constant
        if count < 2 or count > MAX_PARTS:
            return None
        if isinstance(key, cst.Name):
            value = self._value_of(key.value)
            if (isinstance(value, cst.Tuple) and len(value.elements) == count
                    and not any(isinstance(e, cst.StarredElement) for e in value.elements)):
                return [e.value for e in value.elements]      # a local bound once to a tuple display: its parts
            if isinstance(value, cst.Call) and item_constant(value, self.cached_call):
                key = value
            elif key.value in self.facts._params or self.views.get(key.value) or key.value in self.columns:
                return None
        elif not (isinstance(key, cst.Call) and item_constant(key, self.cached_call)):
            return None
        parts = [cst.Subscript(key, [cst.SubscriptElement(cst.Index(cst.Integer(str(i))))]) for i in range(count)]
        for part in parts:
            self.tuple_guards[_code(part)] = (_code(key), count)
        return parts

    def guards(self, parts):
        """The run-time tests that the key tuples of ``parts`` are tuples of their count (one per tuple)."""
        tests = dict.fromkeys(self.tuple_guards[_code(p)] for p in parts if _code(p) in self.tuple_guards)
        return [cst.parse_expression(f'(type({k}) is tuple and len({k}) == {n})') for k, n in tests]

    def _enum_attribute(self, part):
        """``self.<Enum>.<member>`` (or ``self._mx_definition_<Enum>.<member>``) on an enum Reference."""
        if not (isinstance(part, cst.Attribute) and isinstance(part.value, cst.Attribute)
                and isinstance(part.value.value, cst.Name) and part.value.value.value == 'self'):
            return False
        ref = part.value.attr.value
        ref = ref[len('_mx_definition_'):] if ref.startswith('_mx_definition_') else ref
        return bool(getattr(getattr(self.cls_info, 'refs', {}).get(ref), 'enum', None))

    def code_field(self, hold, level, part, emit=None):
        """The item field holding the code of ``part`` (``emit``, the part with its own reads lowered, is what the hold
        evaluates; T-5c)."""
        key = (hold.table, level, _code(part))
        fields = self.code_fields
        if key not in fields:
            fields[key] = hold_table(self.cls_info).field('prepared_code', key)
            self.new_items.append(('code', fields[key], hold.table, level, part if emit is None else emit))
        return fields[key]


    def row_field(self, hold, key_expr, name, emit=None):
        key = (hold.table, _code(key_expr))
        fields = self.row_fields
        if key not in fields:
            fields[key] = hold_table(self.cls_info).field('prepared_row', key)
            self.new_items.append(('row', fields[key], hold.table, key_expr if emit is None else emit, name))
        return fields[key]

    def hold(self, receiver, label, kind):
        """The read (receiver, column, kind): its table is the receiver's binding, one per receiver."""
        code = _code(receiver)
        key = (code, label, kind)
        found = self.holds.get(key)
        if found is None:
            binding = self.bindings.get(code)
            if binding is None:
                if self.per_item:
                    table = hold_table(self.cls_info).field('prepared', code)
                    state = hold_table(self.cls_info).field('prepared_slot', code)
                else:
                    table, state = f'_mx_pt{self.local_count}', None
                    self.local_count += 1
                binding = self.bindings[code] = _Hold(table, state, receiver, None, None)
                self.new_items.append(binding)
            found = self.holds[key] = _Hold(binding.table, binding.state, receiver, label, kind)
        return found

    def ref(self, hold):
        where = f'self.{hold.table}' if self.per_item else hold.table
        return _cast(f'{self.module}.PreparedRef', cst.parse_expression(where))

    def slot(self, hold):
        """The slot of a read: one C load from the held table by the read's signature id."""
        return cst.Call(func=cst.Attribute(cst.Name(self.module), cst.Name('slot')),
                        args=[cst.Arg(self.ref(hold)), cst.Arg(cst.Integer(str(self.read_signature(hold))))])

    def _enum_constants(self, value):
        """{code of each enum member: its C constant} when ``value`` is an admitted IntEnum member (``self.E.m``) or a
        conditional between such values, else None."""
        if isinstance(value, cst.IfExp):
            body, orelse = self._enum_constants(value.body), self._enum_constants(value.orelse)
            return {**body, **orelse} if body is not None and orelse is not None else None
        constant = self.facts.enum_constant(value)
        return {_code(value): constant} if constant is not None else None

    def _enum_key_locals(self):
        """T-5c: {local: its enum constants} for each local bound once to an IntEnum member, or to a choice between
        members (``item = E.a if paid_up(k) else E.b``), whose every read is a key part of a ``.loc``/``.at`` read:
        it is bound to the members' C constants, and a key part it gives is a C integer (D79), not the member
        object (shared by every thread) looked up at every read."""
        found = {}
        f = self.facts.formula
        for name, values in f.values.items():
            value = self._value_of(name)
            constants = self._enum_constants(value) if value is not None else None
            loads = f.loads.get(name) or ()
            if constants and loads and all(not nested and self._key_part_position(load) for load, nested in loads):
                found[name] = constants
        return found

    def _key_part_position(self, node):
        """``node`` is the key, or a part of a tuple key, of a ``.loc[...]`` / ``.at[...]`` read."""
        parents = self.facts.formula.parents
        up = parents.get(id(node))
        if isinstance(up, cst.Element):
            up = parents.get(id(parents.get(id(up))))
        element = parents.get(id(up)) if isinstance(up, cst.Index) else None
        subscript = parents.get(id(element)) if isinstance(element, cst.SubscriptElement) else None
        return (isinstance(subscript, cst.Subscript) and isinstance(subscript.value, cst.Attribute)
                and subscript.value.attr.value in ('loc', 'at'))

    def key_part(self, part, level_kind, emit=None):
        """Emit a key part for typed dispatch on integer levels and object lookup elsewhere.

        The facts apply to the original ``part``. ``emit`` is the same part with its own reads lowered.
        """
        emit = part if emit is None else emit
        if level_kind == 'int' and isinstance(part, cst.Name) and part.value in self.enum_key_locals:
            return _cast('_mx_cy.longlong', emit), 'int'     # T-5c: bound to the members' C constants
        constant = self.facts.enum_constant(part) if level_kind == 'int' else None
        if constant is not None:        # an IntEnum member: its C constant (enums.py, D79; T-5c: not only under is_int)
            return cst.Name(constant), 'int'
        if level_kind == 'int':
            # Cython types int() as an object. Give it a C type only when the input fits:
            # casting int() of a C double directly to long long can skip Python's conversion checks.
            inner = self.facts.int_argument(part)
            if inner is not None and self.facts.fits_long_long(inner):
                return _cast('_mx_cy.longlong', emit), 'typed'
            return emit, 'typed'
        return _cast('object', emit), 'object'


_NO_LABEL = object()

# T-5c: in an item class, a frame the item computes itself is prepared (about 0.3-2M instructions, once per item)
# only for a formula evaluated at least this many times per item; a pandas read costs about 20-100k instructions
PREPARE_MIN_EVALUATIONS = 8


# T-5c: attributes of a pandas object whose subscript is a pandas selection (a bool key is passed as the Python bool)
PANDAS_INDEXERS = frozenset(('index', 'iloc', 'loc', 'at', 'iat', 'columns'))


def _starred_target(target):
    """T-5c: for a loop target holding a starred element (``for a, *b in ...``): a rewrite of the loop into ``for
    _mx_item in ...:`` and ``a, *b = _mx_item`` first in its body (Cython types a starred loop target from the
    items; the assignment makes ``b`` the list Python makes it), else None."""
    found = []

    class Find(cst.CSTVisitor):
        def visit_StarredElement(self, node):
            found.append(node)
    target.visit(Find())
    if not found:
        return None

    def rewrite(updated, owner):
        if not isinstance(updated.body, cst.IndentedBlock):
            return updated          # a one-line body: kept as written
        item = cst.Name(f'_mx_item{owner.local_count}')
        owner.local_count += 1
        unpack = cst.SimpleStatementLine([cst.Assign([cst.AssignTarget(updated.target)], item)])
        body = updated.body.with_changes(body=[unpack, *updated.body.body])
        return updated.with_changes(target=item, body=body)
    return rewrite


# T-5c: builtins of one iterable whose generator argument is hoisted out of the read's conditional
WALK_BUILTINS = frozenset(('sorted', 'list', 'tuple', 'set', 'frozenset', 'sum', 'min', 'max', 'any', 'all'))

# T-5c: the run-time conditions that send a lowered read to pandas, by the signature kind of a hold it takes (the
# report's ``pandas_when``; the kinds of levels and columns apply to every read)
RUN_TIME = {
    -2: 'the table or view is empty',
    -3: "the stored order of the index within the prefix differs from pandas' order",
    -5: '`T.loc[a, b]` is not a row prefix of the table at run time (pandas reads it as row and column)',
    -6: 'the level has no longer that name, is not an int level, or the table is empty',
}


class _View:
    """T-5: the rows of a frame receiver ``call`` a read walks: the whole table (``start`` None: pandas positions map
    to stored rows), or the stored rows [start, stop) of a prefix of ``depth`` levels (``codes``; C locals of the
    formula), held in a local ``name`` (of a prefix cell ``cell``, or None)."""

    def __init__(self, name, call, schema, depth=0, codes=None, start=None, stop=None, cell=None):
        self.name, self.call, self.schema, self.depth, self.codes = name, call, schema, depth, codes or []
        self.start, self.stop, self.cell = start, stop, cell
        self.setup, self.used = None, False


def without_fallbacks(node):
    """``node`` with the pandas branch of every lowered read replaced by ``None`` (for the build report: what is
    left on pandas is what runs there whenever the table is prepared)."""
    from .helpers import module_name
    module = module_name(PROVIDER)

    class Strip(cst.CSTTransformer):
        def leave_IfExp(self, original, updated):
            if f'{module}.slot(' in _code(original.test):
                return updated.with_changes(orelse=cst.Name('None'))
            return updated

        def leave_Call(self, original, updated):     # the frame a binding prepares or looks up is not read
            if isinstance(updated.func, cst.Attribute) and updated.func.attr.value in ('bind_ref', 'bind_ref_checked', 'bind_own') \
                    and _code(updated.func.value) == module:
                return updated.with_changes(args=[])
            return updated

        def leave_If(self, original, updated):       # a lowered loop and its pandas copy
            test = _code(original.test)
            if (f'{module}.slot(' in test or f'{module}.zip_prepared_prefix(' in test) and updated.orelse is not None:
                return updated.with_changes(orelse=None)
            return updated
    return node.visit(Strip())


def audit_subtables(module_infos, modules, resolver):
    """Classify table-producing cells after all of their readers have been lowered."""
    from .usage import RESOLVED, UNKNOWN, _extract_self_chain

    def target(cls, expr):
        chain = _extract_self_chain(expr)
        if isinstance(chain, list) and chain:
            status, fqname = resolver.resolve_chain(cls.fqname, chain)
            if status == RESOLVED:
                return fqname
        return None

    native_receivers, parameterised = set(), set()
    for info in module_infos.values():
        for cls in info.classes.values():
            receivers = set()
            for cell in cls.cells.values():
                parameterised.update(cls.cells[name].fqname for name in getattr(cell, 'read_param_readers', ()))
                receivers.update(read['receiver'] for read in getattr(cell, 'frame_reads', ()) if 'reason' not in read)
            for text in receivers:
                expr = cst.parse_expression(text)
                if isinstance(expr, cst.Call) and not expr.args:
                    native_receivers.add(target(cls, expr.func))

    if parameterised:
        names = {fqname.rsplit('._f_', 1)[-1] for fqname in parameterised}
        remaining = set()

        class Readers(cst.CSTVisitor):
            def __init__(self, cls):
                self.cls = cls

            def visit_Attribute(self, node):
                name = node.attr.value
                if name not in names:
                    return
                chain = _extract_self_chain(node)
                status, fqname = (resolver.resolve_chain(self.cls.fqname, chain)
                                  if isinstance(chain, list) else (UNKNOWN, name))
                if status == RESOLVED:
                    remaining.add(fqname)
                elif status == UNKNOWN:
                    # This can be a call or method capture through a local alias of a model Space.
                    remaining.update(resolver.cells_with_name(name))

        for name, module in modules.items():
            classes = module_infos[name].classes
            for node in module.syntax.body:
                if isinstance(node, cst.ClassDef) and node.name.value in classes:
                    without_fallbacks(node).visit(Readers(classes[node.name.value]))
        parameterised.difference_update(remaining)

    for info in module_infos.values():
        for cls in info.classes.values():
            for cell in cls.cells.values():
                if cell.fqname in parameterised:
                    reason = ('a parameterised sub-table cell: all compiled readers read the parent table directly '
                              'when it is prepared')
                elif cell.fqname in native_receivers:
                    reason = 'a row or sub-table cell: its readers read its value natively when it is prepared'
                else:
                    continue
                for entry in getattr(cell, 'read_audit', ()):
                    entry['reason'] = reason
                    entry['per_step'] = False


class _Rewrite(cst.CSTTransformer):
    """Rewrites reads inside one top-level statement; nested functions and lambdas are not entered."""

    def __init__(self, owner):
        super().__init__()
        self.o = owner
        self.new_holds = []
        self.replaced_names = set()
        self.swaps = {}         # frame or prefix local -> the call it stands for (fallbacks, dropped locals)
        self.key_locals = {}    # name of a key local -> its tuple display (the fallback evaluates the display)
        self.asof_rows = {}     # asof local -> (membership hold, prefix cell call, asof expression)
        self.native = {}        # id(lowered read) -> column kind, for float()/int() around it
        self.taken = []         # T-5c: kinds of the holds taken since the last recorded read (its run-time fallbacks)
        self.generator_walks = set()    # T-5c: ids of lowered generator walks (``gen if test else pandas gen``)
        self.updated_of = {}    # id(original node) -> its rewritten node (key parts are emitted with their reads lowered)
        self.held_string_read = None

    def on_leave(self, original, updated):
        result = super().on_leave(original, updated)
        self.updated_of[id(original)] = result
        if result is not updated and isinstance(original, (cst.Subscript, cst.Call, cst.Comparison, cst.For,
                                                             cst.GeneratorExp, cst.ListComp)):
            self.o.lowered_text.add(normal(_code(original)))
        elif result is not updated and isinstance(original, cst.SimpleStatementLine):
            assigned = self.o._simple_assign(original)      # `row = R.loc[key]`: the statement is rewritten
            if assigned is not None:
                self.o.lowered_text.add(normal(_code(assigned[1])))
        return result

    def visit_FunctionDef(self, node):
        return False

    def visit_Lambda(self, node):
        return False

    def visit_ClassDef(self, node):
        return False

    def _take(self, receiver, label, kind):
        self.taken.append(kind)
        return self.o.hold(receiver, label, kind)

    def _record(self, node, form, receiver, why=None, view=None):
        text = _code(node)
        entry = {'expression': text[:160], 'receiver': _code(receiver), 'form': form}
        if not why:
            schema = self.o._schema_of_call(receiver) or {}
            when = ['the frame at run time has other index level kinds than '
                    f"{list(schema.get('levels', ()))}, or the column read is of another kind"]
            when += [RUN_TIME[kind] for kind in sorted(set(self.taken) & set(RUN_TIME), key=str)]
            if any(code in entry['expression'] for code in self.o.ambiguous):
                when.append(RUN_TIME[-5])
            entry['pandas_when'] = '; '.join(dict.fromkeys(when))
        self.taken = []
        if why:
            entry['reason'] = why
            self.o.declined.append(entry)
        else:
            self.o.lowered.append(entry)
            self.o.lowered_nodes.append(node)

    def _key(self, key, schema):
        """The parts of a complete key on ``schema``: (parts, None) or (None, why). A key held in a local assigned
        once to a tuple display is that display."""
        levels = schema['levels']
        value = self.o._value_of(key.value) if isinstance(key, cst.Name) and len(levels) > 1 else None
        if isinstance(value, cst.Tuple):
            self.key_locals[key.value] = key = value
        if len(levels) == 1:
            if isinstance(key, cst.Tuple):
                return None, 'a tuple key on a single-level index'
            parts = [key]
        else:
            runtime = self.o.tuple_key(key, len(levels)) if not isinstance(key, cst.Tuple) else None
            if runtime is not None:         # T-5c: a tuple held in a local or returned by a cell, checked per read
                return runtime, None
            if not isinstance(key, cst.Tuple) or any(isinstance(e, cst.StarredElement) for e in key.elements):
                return None, 'the key is not a tuple display of every level'
            parts = [e.value for e in key.elements]
            if len(parts) != len(levels):
                return None, f'{len(parts)} key parts on an index of {len(levels)} levels (a prefix: pandas)'
        if len(parts) > MAX_PARTS:
            return None, f'{len(parts)} key parts'
        for part in parts:
            if isinstance(part, SEVERAL_ROWS):
                return None, SEVERAL_ROWS_WHY
        return parts, None

    def _read_args(self, hold, parts, levels):
        """(key arguments, extra native conditions) for a read of ``parts`` through ``hold``: one level code per part
        (T-5: the one read path for every key), ``held - 1`` for a part fixed per item (and its condition that the
        part is present), ``lcode``/``ocode`` of the part otherwise."""
        conditions = self.o.guards(parts)
        codes = [self._held(hold, level, part, conditions) if self.o.item_fixed(part, kind)
                 else self.code_of(hold, level, part, kind, conditions)
                 for level, (part, kind) in enumerate(zip(parts, levels))]
        return self.code_args(codes), conditions

    def _detached(self, expr):
        """T-5c: a key from outside this formula's tree (a prefix or row cell's own key), its reads lowered for a hold
        to evaluate once per item: ``self.R()["f"]``, a field of a row cell, reads the held row instead of building
        the pandas row. ``expr`` itself when it has no such read."""
        found = []

        class Fields(cst.CSTVisitor):
            def visit_Subscript(inner, node):
                if isinstance(node.value, cst.Call) and self.o._fixed_value(node):
                    found.append(node)
        expr.visit(Fields())
        return expr.visit(self) if found else expr

    def _held(self, hold, level, part, conditions):
        """The code of a key part fixed per item, held per item (``held - 1``); ``conditions`` gets its presence."""
        emit = self.updated_of.get(id(part))
        field = self.o.code_field(hold, level, part, emit if emit is not None else self._detached(part))
        conditions.append(cst.parse_expression(f'self.{field} > 0'))
        return cst.parse_expression(f'self.{field} - 1')

    @staticmethod
    def _padded(codes):
        return [*codes, *[cst.Integer('0')] * (5 - len(codes))]

    @staticmethod
    def code_args(codes):
        """(function suffix, arguments) of a read by level codes: ``_1`` with the one code, else ``_c`` with the count
        and six codes (unused ones 0)."""
        if len(codes) == 1:
            return '_1', codes
        return '_c', [cst.Integer(str(len(codes))), *codes, *[cst.Integer('0')] * (MAX_PARTS - len(codes))]

    def _codes(self, hold, prefix, parts, levels, conditions=None):
        """code_args of the codes ``prefix`` followed by those of key ``parts`` on the levels after them."""
        return self.code_args(list(prefix) + [self.code_of(hold, level, part, levels[level], conditions)
                                              for level, part in enumerate(parts, len(prefix))])

    def code_of(self, hold, level, part, kind, conditions=None):
        """The level code of one key part computed per read: ``lcode`` of a C integer on an int level, else ``ocode``
        of the Python value (looked up as pandas does)."""
        o = self.o
        arg, how = o.key_part(part, kind, self.updated_of.get(id(part)))
        if (how == 'typed' and conditions is not None and not o.facts.is_int(part)
                and o.facts.int_argument(part) is None):
            # not known to be a C integer from the text: the read is native only while the part names one label.
            # The test is decided at compile time when Cython gives the part a C type, else it looks at the value;
            # the part is evaluated once, into a local the read then uses
            o.key_count += 1
            name = f'_mx_key_{o.key_count}'
            conditions.append(cst.parse_expression(f'{o.module}.is_label(({name} := {_code(arg)}))'))
            arg = cst.Name(name)
        return self._m({'int': 'lcode', 'typed': 'kcode'}.get(how, 'ocode'), o.ref(hold), cst.Integer(str(level)), arg)

    def _native_test(self, hold, conditions):
        test = self._positive(self.o.slot(hold))
        for condition in conditions:
            test = cst.BooleanOperation(left=test, operator=cst.And(), right=condition)
        return test

    # ------------------------------------------------------------------ T-5: views and index walks
    def view_statement(self, statement, found):
        """``name = R.loc[prefix]``: the prefix codes and the view's stored rows [start, stop) in C locals, before
        the assignment (dropped when every use of ``name`` is lowered). The view is valid while start >= 0."""
        o = self.o
        name, call, schema, parts, cell, *rest = found
        held = rest[0] if rest else None    # T-5c: the cell's parts of a prefix of a prefix cell's value
        hold = self._take(call, _NO_LABEL, -1)
        test = _code(self._native_test(hold, []))
        if cell is None and _code(statement.body[0].value) in o.ambiguous:
            test = f'{test} and {_code(self._native_test(self._take(call, 1, -5), []))}'
        statements, codes, conditions = [], [], []
        if cell is not None:
            codes, conditions = self._prefix_holds(hold, parts if held is None else parts[:held], cell)
        if cell is None or held is not None:
            for level, part in enumerate(parts[len(codes):], len(codes)):
                if o.item_fixed(part, schema['levels'][level]):
                    codes.append(self._held(hold, level, part, conditions))
                    continue
                var = f'_mx_vc_{name}_{level}'
                code = _code(self.code_of(hold, level, part, schema['levels'][level]))
                statements.append(cst.parse_statement(f'{var}: _mx_cy.Py_ssize_t = {code} if {test} else -1'))
                codes.append(cst.Name(var))
        args = ', '.join(_code(c) for c in self._padded(codes))
        guard = ' and '.join([test] + [_code(c) for c in conditions])
        start, stop = f'_mx_vs_{name}', f'_mx_ve_{name}'
        m = o.module
        packed = f'({m}.view_range({_code(o.ref(hold))}, {len(parts)}, {args}) if {guard} else -1)'
        if o.per_item and not statements:       # every prefix code held per item: the rows are too
            # keyed by the prefix (its held codes and conditions), not the local's name: two formulas of the class
            # may name views of different prefixes alike (T-5c defect fix)
            field = hold_table(o.cls_info).field('prepared_row', (hold.table, 'range', packed))
            o.new_items.append(('range', field, hold.table, packed, None))
            packed = f'(self.{field} - 2)'
        statements.append(cst.parse_statement(f'_mx_vr_{name}: _mx_cy.longlong = {packed}'))
        statements.append(cst.parse_statement(f'{start}: _mx_cy.Py_ssize_t = {m}.range_start(_mx_vr_{name})'))
        statements.append(cst.parse_statement(f'{stop}: _mx_cy.Py_ssize_t = {m}.range_stop(_mx_vr_{name})'))
        value = statement.body[0].value
        source = (value.with_changes(value=value.value.with_changes(value=call))
                  if cell is None and isinstance(value, cst.Subscript) else value)      # (T-5c: or P(a) itself)
        view = o.views[name] = _View(name, call, schema, len(parts), codes, cst.Name(start), cst.Name(stop),
                                     cell if held is None else None)
        view.setup = [*statements, statement]     # the setup statements are kept only if a use needs them
        self.swaps[name] = source
        return view.setup

    def _view(self, expr):
        """The view a receiver expression stands for: a view local, or a whole frame or Series (unique index)."""
        o = self.o
        if isinstance(expr, cst.Name) and expr.value in o.views:
            o.views[expr.value].used = True
            return o.views[expr.value]
        if isinstance(expr, cst.Name) and expr.value in o.columns:     # a column's index is its view's
            return self._view(o.columns[expr.value][0])
        prefix = o.prefix_receiver(expr) if isinstance(expr, cst.Call) else None
        if prefix is not None and self._shared(prefix[1]) and o.per_item:
            # T-5c: ``self.P()`` itself (P a prefix cell): a view of its table, P's parts held per item
            _, call, schema, parts, cell = prefix
            hold = self._take(call, _NO_LABEL, -1)
            codes, conditions = self._prefix_holds(hold, parts, cell)
            args = [o.ref(hold), cst.Integer(str(len(parts))), *self._padded(codes)]
            start = self._lowered(self._conjunction(conditions), self._m('view_start', *args), cst.parse_expression('-1'))
            return _View(None, call, schema, len(parts), codes, start, self._m('view_stop', *args))
        inline = o.prefix_of(expr)
        if inline is not None and not self._shared(inline[0][0]):
            return None
        if inline is not None:          # T.loc[prefix] in the expression itself: its codes computed per use
            (call, schema), parts = inline
            hold = self._take(call, _NO_LABEL, -1)
            codes = [self.code_of(hold, level, part, schema['levels'][level]) for level, part in enumerate(parts)]
            args = [o.ref(hold), cst.Integer(str(len(parts))), *self._padded(codes)]
            start = self._m('view_start', *args)
            if _code(expr) in o.ambiguous:
                start = cst.IfExp(test=self._native_test(self._take(call, 1, -5), []), body=start,
                                  orelse=cst.parse_expression('-1'), lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
            return _View(None, call, schema, len(parts), codes, start, self._m('view_stop', *args))
        found = o._receiver(expr)
        if found is None or not found[1].get('unique') or _code(found[0]) in o.mutated or not self._shared(found[0]):
            return None
        if isinstance(expr, cst.Name):
            self.replaced_names.add(expr.value)
        return _View(None, *found)

    def _shared(self, call):
        """In an item, walks read only a table the Data space publishes: a frame the item computes itself would be
        prepared per item for them."""
        return not self.o.per_item or self.o.published(call) is not None

    def _view_parts(self, view, hold):
        """(start, stop, test) of a view through ``hold`` (a read of the given signature)."""
        o = self.o
        test = self._native_test(hold, [])
        if view.start is None:
            return cst.Integer('0'), self._m('nrows', o.ref(hold)), test
        return view.start, view.stop, self._conjunction([test, cst.Comparison(
            view.start, [cst.ComparisonTarget(cst.GreaterThanEqual(), cst.Integer('0'))])])

    def _m(self, name, *args):
        return cst.Call(func=cst.Attribute(cst.Name(self.o.module), cst.Name(name)), args=[cst.Arg(a) for a in args])

    @staticmethod
    def _next(expr):
        return cst.BinaryOperation(expr, cst.Add(), cst.Integer('1'))

    @staticmethod
    def _range(*bounds):
        return cst.Call(func=cst.Name('range'), args=[cst.Arg(b) for b in bounds])

    @staticmethod
    def _assign(name, value):
        return cst.SimpleStatementLine([cst.Assign([cst.AssignTarget(cst.Name(name))], value)])

    @staticmethod
    def _positive(expr):
        return cst.Comparison(expr, [cst.ComparisonTarget(cst.GreaterThan(), cst.Integer('0'))])

    @staticmethod
    def _loop(updated, position, iterator, first):
        """The native loop of a walk: ``for position in iterator:``, the statements ``first`` before the body."""
        body = updated.body
        body = body.with_changes(body=[*first, *body.body]) if isinstance(body, cst.IndentedBlock) else \
            cst.IndentedBlock(body=[*first, cst.SimpleStatementLine(body=body.body)])
        return updated.with_changes(target=position, body=body, iter=iterator)

    def _row_at(self, view, hold, position):
        """The stored row at a pandas position of a view (a stored row of a prefix view, ordered)."""
        if view.start is None:
            return self._m('whole_row', self.o.ref(hold), position)
        return position

    def _label(self, view, hold, row, level=None):
        level = view.depth if level is None else level
        kind = view.schema['levels'][level]
        return self._m('label_ll' if kind == 'int' else 'label_obj', self.o.ref(hold), row, cst.Integer(str(level)))

    def _one_level(self, view):
        return len(view.schema['levels']) - view.depth == 1

    def _walk_hold(self, view):
        """The hold of a walk in pandas order: a whole table maps positions to rows; a prefix view needs an ordered
        table (signature kind -3)."""
        return self._take(view.call, _NO_LABEL, -1) if view.start is None else self._take(view.call, view.depth, -3)

    def _guarded(self, original, updated, view, test, native, kind=None):
        return self._lowered(test, native, self._fallback(updated, view.call), kind)

    def _lowered(self, test, read, fallback, kind=None):
        """``read if test else fallback``: the pandas read cast to the read's C type when the read is typed (both
        branches native), and the lowered read marked as of that kind (``float()``/``int()`` around it dropped)."""
        if kind in CTYPE:
            fallback = _cast(CTYPE[kind], fallback)
        lowered = cst.IfExp(test=test, body=read, orelse=fallback, lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        if kind in CTYPE:
            self.native[id(lowered)] = kind
        return lowered

    def _index_of(self, expr):
        """``X.index`` with X a view, or a local assigned once from one (T-5c): the view, else None."""
        if isinstance(expr, cst.Name):
            value = self.o._value_of(expr.value)
            view = self._index_of(value) if isinstance(value, cst.Attribute) else None
            if view is not None:        # the pandas fallbacks read the index itself
                self.swaps.setdefault(expr.value, value)
            return view
        if isinstance(expr, cst.Attribute) and expr.attr.value == 'index':
            return self._view(expr.value)
        return None

    def index_item(self, original, updated):
        """``X.index[i]``: the label at position i (i an integer; IndexError as pandas)."""
        o = self.o
        parts = _index_parts(original)
        view = self._index_of(original.value)
        if view is None or parts is None or len(parts) != 1 or not self._one_level(view):
            return None
        if not (o.facts.is_int(parts[0]) or self.native.get(id(self.updated_of.get(id(parts[0])))) == 'i64'):
            return None
        if isinstance(parts[0], cst.Name) and parts[0].value in ('True', 'False') or isinstance(parts[0], cst.Float):
            return None         # T-5c: a bool or float is not a position for pandas (it raises)
        hold = self._walk_hold(view)
        start, stop, test = self._view_parts(view, hold)
        i = _cast('_mx_cy.Py_ssize_t', self.updated_of.get(id(parts[0]), parts[0]))
        row = self._row_at(view, hold, self._m('position', start, stop, i))
        self._record(original, 'L8', view.call, view=view)
        int_level = view.schema['levels'][view.depth] == 'int'
        return self._guarded(original, updated, view, test, self._label(view, hold, row), 'i64' if int_level else None)

    def index_call(self, original, updated):
        """``X.index.searchsorted(x, side=...)``, ``int(X.index.asof(x))`` / ``X.index.asof(x)`` as a key part,
        ``X.index.max()`` / ``.min()`` of a prefix view, and len/list/tuple/set/sorted/min/max of ``X.index``."""
        o = self.o
        func = original.func
        if isinstance(func, cst.Name) and len(original.args) == 1 and not original.args[0].keyword:
            arg = original.args[0].value
            if isinstance(arg, cst.Call) and o.facts.int_argument(original) is not None:
                return self._asof(arg, updated, 0, whole_call=original)
            view = self._index_of(arg)
            if view is not None and func.value in ('len', 'list', 'tuple', 'set', 'sorted', 'min', 'max'):
                return self._builtin(func.value, view, original, updated)
            column = self._column_of(arg) if func.value in ('len', 'list', 'tuple') else None
            if column is not None:
                return self._column_builtin(func.value, column, original, updated)
            return None
        if isinstance(func, cst.Attribute) and func.attr.value == 'tolist' and not original.args:
            column = self._column_of(func.value)
            if column is not None:
                return self._column_builtin('list', column, original, updated)
        level = self.level_extreme(original, updated)
        if level is not None:
            return level
        view = self._index_of(func.value) if isinstance(func, cst.Attribute) else None
        if view is None:
            return None
        if (view is not None and view.start is None and func.attr.value in ('max', 'min') and not original.args
                and o.extreme(original) is None):
            hold = self._take(view.call, _NO_LABEL, -2)          # pandas' own max/min of a non-empty index
            self._record(original, 'L5', view.call, view=view)
            native = self._m(func.attr.value + '_obj', o.ref(hold))
            return self._guarded(original, updated, view, self._native_test(hold, []), native)
        levels = view.schema['levels'] if view is not None else ()
        if (view is not None and view.start is not None and func.attr.value in ('max', 'min') and not original.args
                and len(levels) - view.depth > 1):
            # T-5c: lexicographic: rows in pandas order at every remaining level (one order verdict per depth)
            holds = [self._take(view.call, depth, -3) for depth in range(view.depth, len(levels))]
            start, stop, test = self._view_parts(view, holds[0])
            test = self._conjunction([test, *(self._native_test(h, []) for h in holds[1:]),
                                      cst.Comparison(stop, [cst.ComparisonTarget(cst.GreaterThan(), start)])])
            row = start if func.attr.value == 'min' else cst.BinaryOperation(stop, cst.Subtract(), cst.Integer('1'))
            native = cst.Tuple([cst.Element(self._label(view, holds[0], row, level))
                                for level in range(view.depth, len(levels))])
            self._record(original, 'L5', view.call, view=view)
            return self._guarded(original, updated, view, test, native)
        if view is None or not self._one_level(view) or view.schema['levels'][view.depth] != 'int':
            return None
        if func.attr.value == 'searchsorted':
            side = [a for a in original.args if a.keyword is not None]
            plain = [a for a in original.args if a.keyword is None]
            right = side and side[0].keyword.value == 'side' and _literal_str(side[0].value) == 'right'
            left = not side or (side[0].keyword.value == 'side' and _literal_str(side[0].value) == 'left')
            if len(plain) != 1 or len(side) > 1 or not (right or left) or _text(plain[0].value):
                return None         # (a text x is pandas' own TypeError, never a C double)
            if self._vector_argument(original, plain[0].value, view):
                return None
            hold = self._take(view.call, view.depth, -3)
            start, stop, test = self._view_parts(view, hold)
            x = _cast('_mx_cy.double', self.updated_of.get(id(plain[0].value), plain[0].value))
            native = self._m('search', o.ref(hold), start, stop, cst.Integer(str(view.depth)), x,
                             cst.Name('True' if right else 'False'))
            # T-5c: NaN (pandas sorts it last) is pandas' own
            test = self._conjunction([test, cst.Comparison(x, [cst.ComparisonTarget(cst.Equal(), x)])])
            self._record(original, 'L8', view.call, view=view)
            return self._guarded(original, updated, view, test, native, 'i64')
        if func.attr.value in ('max', 'min') and not original.args and view.start is not None:
            hold = self._take(view.call, _NO_LABEL, -1)
            start, stop, test = self._view_parts(view, hold)
            row = start if func.attr.value == 'min' else cst.BinaryOperation(stop, cst.Subtract(), cst.Integer('1'))
            self._record(original, 'L5', view.call, view=view)
            return self._guarded(original, updated, view, test, self._label(view, hold, row), 'i64')
        if func.attr.value == 'asof' and self._key_position(original):
            return self._asof(original, updated, 1)
        if (func.attr.value == 'asof' and len(original.args) == 1 and not original.args[0].keyword
                and not _text(original.args[0].value)):
            # T-5c: a bare asof value: the label when one is <= x, else pandas' own asof (NaN stays pandas')
            if self._vector_argument(original, original.args[0].value, view):
                return None
            hold = self._take(view.call, view.depth, -3)
            start, stop, test = self._view_parts(view, hold)
            x = _cast('_mx_cy.double', self.updated_of.get(id(original.args[0].value), original.args[0].value))
            found = self._m('search', o.ref(hold), start, stop, cst.Integer(str(view.depth)), x, cst.Name('True'))
            row = cst.BinaryOperation(cst.BinaryOperation(start, cst.Add(), found), cst.Subtract(), cst.Integer('1'))
            test = self._conjunction([test, cst.Comparison(found, [cst.ComparisonTarget(cst.GreaterThan(),
                                                                                       cst.Integer('0'))])])
            self._record(original, 'L6', view.call, view=view)
            return self._guarded(original, updated, view, test, self._label(view, hold, row))
        return None

    def level_extreme(self, original, updated):
        """``R.index.get_level_values(L).max()`` / ``.min()`` (L a literal level number, or a level name traced on
        every frame) of an int level: its largest / smallest label, a C integer. Decided per table at prepare
        (signature kind -6: the level still has that name, is an int level, the table is not empty)."""
        o = self.o
        func = original.func
        inner = func.value if isinstance(func, cst.Attribute) else None
        local = inner.value if isinstance(inner, cst.Name) else None
        if local is not None:           # T-5c: ``ages = R.index.get_level_values(L)``, then ``ages.max()``
            inner = o._value_of(local)
        if not (inner is not None and func.attr.value in ('max', 'min') and not original.args and isinstance(inner, cst.Call)
                and isinstance(inner.func, cst.Attribute) and inner.func.attr.value == 'get_level_values'
                and len(inner.args) == 1 and not inner.args[0].keyword
                and isinstance(inner.func.value, cst.Attribute) and inner.func.value.attr.value == 'index'):
            return None
        found = o._receiver(inner.func.value.value)
        if found is None or not found[1].get('unique') or _code(found[0]) in o.mutated or not self._shared(found[0]):
            return None
        call, schema = found
        arg = inner.args[0].value
        names = schema.get('names') or ()
        if isinstance(arg, cst.Integer):
            index, name = int(arg.value, 0), None
        elif _literal_str(arg) is not None and list(names).count(_literal_str(arg)) == 1:
            name = _literal_str(arg)
            index = list(names).index(name)
        else:
            return None
        if not 0 <= index < len(schema['levels']) or schema['levels'][index] != 'int':
            return None
        hold = self._take(call, (name, index), -6)
        if local is not None:           # the pandas branch reads the level values; the local goes once unread
            self.swaps.setdefault(local, inner)
        if isinstance(inner.func.value.value, cst.Name):
            self.replaced_names.add(inner.func.value.value.value)
        native = self._m(f'level_{func.attr.value}_ll', o.ref(hold), cst.Integer(str(index)))
        self._record(original, 'L5', call)
        return self._guarded(original, updated, _View(None, call, schema),
                             self._native_test(hold, []), native, 'i64')

    def level_walk_extreme(self, original, updated):
        """T-5c: ``max(a for a, s in R.index)`` / ``min(...)``: the element is the part of the tuple target naming an
        int level, with no condition, over a whole frame of unique keys: that level's extreme as a C integer, as
        ``R.index.get_level_values(i).max()`` (signature kind -6), not a walk of every row per evaluation."""
        o = self.o
        func = original.func
        if not (isinstance(func, cst.Name) and func.value in ('max', 'min') and len(original.args) == 1
                and not original.args[0].keyword and not original.args[0].star
                and isinstance(original.args[0].value, cst.GeneratorExp)):
            return None
        gen = original.args[0].value
        comp = gen.for_in
        if comp.ifs or comp.inner_for_in is not None or comp.asynchronous is not None or not isinstance(gen.elt, cst.Name):
            return None
        target = comp.target
        if not (isinstance(target, cst.Tuple) and all(isinstance(e, cst.Element) and isinstance(e.value, cst.Name)
                                                       for e in target.elements)):
            return None
        names = [e.value.value for e in target.elements]
        if names.count(gen.elt.value) != 1:
            return None
        level = names.index(gen.elt.value)
        view = self._index_of(comp.iter)
        if view is None or view.start is not None or not view.schema.get('unique'):
            return None
        levels = view.schema['levels']
        if len(levels) != len(names) or levels[level] != 'int' or _code(view.call) in o.mutated:
            return None
        hold = self._take(view.call, (None, level), -6)
        native = self._m(f'level_{func.value}_ll', o.ref(hold), cst.Integer(str(level)))
        self._record(original, 'L5', view.call, view=view)
        arg = updated.args[0].value         # the pandas branch: the generator as written, not its lowered walk
        if id(arg) in self.generator_walks:
            updated = updated.with_changes(args=[updated.args[0].with_changes(value=arg.orelse.with_changes(
                lpar=[], rpar=[]))])
        return self._lowered(self._native_test(hold, []), native, self._fallback(updated, view.call), 'i64')

    def extreme_part(self, original, updated, parts):
        """T-5c: ``R.index.max()[k]`` / ``.min()[k]`` on a whole frame of several levels, ``k`` an integer literal
        naming an int level: that part of pandas' extreme as a C integer (signature kind -2: the index is not
        empty), where the label tuple would be one Python object shared by every thread."""
        o = self.o
        call = original.value
        if not (len(parts) == 1 and isinstance(parts[0], cst.Integer) and isinstance(call, cst.Call) and not call.args
                and isinstance(call.func, cst.Attribute) and call.func.attr.value in ('max', 'min')):
            return None
        view = self._index_of(call.func.value)
        if view is None or view.start is not None:
            return None
        levels = view.schema['levels']
        level = int(parts[0].value, 0)
        if not (1 < len(levels) <= 8 and 0 <= level < len(levels) and levels[level] == 'int'):
            return None
        hold = self._take(view.call, _NO_LABEL, -2)
        inner = updated.value
        pandas = updated.with_changes(value=inner.orelse) if isinstance(inner, cst.IfExp) else updated
        native = self._m(f'{call.func.attr.value}_part_ll', o.ref(hold), cst.Integer(str(level)))
        self._record(original, 'L5', view.call, view=view)
        return self._lowered(self._native_test(hold, []), native, self._fallback(pandas, view.call), 'i64')

    def _key_position(self, node):
        """``node`` is the key (or a key part) of a ``.loc[...]`` / ``.at[...]`` read."""
        parents = self.o.facts.formula.parents
        up = parents.get(id(node))
        if isinstance(up, cst.Element):
            up = parents.get(id(parents.get(id(up))))       # a part of a tuple key
        element = parents.get(id(up)) if isinstance(up, cst.Index) else None
        subscript = parents.get(id(element)) if isinstance(element, cst.SubscriptElement) else None
        return (isinstance(subscript, cst.Subscript) and subscript.slice[0] is element
                and isinstance(subscript.value, cst.Attribute) and subscript.value.attr.value in ('loc', 'at'))

    def _vector_argument(self, call, arg, view):
        """Whether the argument of ``X.index.asof(x)`` / ``.searchsorted(x)`` is not one number, so the call stays on
        pandas (recorded): from the text, a display that selects several rows (as a key part); from tracing, a call
        whose result in the sample was not a single number (an array, a list, a Series). A call with no note is
        not vetoed."""
        why = None
        if isinstance(arg, SEVERAL_ROWS):
            why = 'the argument is a display of several values (an array result, never one label)'
        else:
            note = self.o.note_of(call)
            if note is not None and any(len(seen) == 3 and seen[1] == call.func.attr.value
                                        and seen[2] not in ONE_NUMBER | {RAISES} for seen in note['seen']):
                why = 'the sample returned more than one number from this call (its argument holds several values)'
        if why:
            self._record(call, 'pandas', view.call, why)
        return why is not None

    def _asof(self, call, updated, error, whole_call=None):
        """``X.index.asof(x)`` on an int level of an ordered view, as the label (C long long); no label: the error
        of its consumer (``int()``: ValueError; a key: KeyError)."""
        o = self.o
        func = call.func
        if not (isinstance(func, cst.Attribute) and func.attr.value == 'asof' and isinstance(func.value, cst.Attribute)
                and func.value.attr.value == 'index' and len(call.args) == 1 and not call.args[0].keyword):
            return None
        view = self._view(func.value.value) if not _text(call.args[0].value) else None
        if view is None or not self._one_level(view) or view.schema['levels'][view.depth] != 'int':
            return None
        if self._vector_argument(call, call.args[0].value, view):
            return None
        hold = self._take(view.call, view.depth, -3)
        start, stop, test = self._view_parts(view, hold)
        x = call.args[0].value
        x = _cast('_mx_cy.double', self.updated_of.get(id(x), x))
        # T-5c: pandas' asof of NaN differs by version (NaN on pandas 3, the last label on pandas 2.2): NaN is read
        # on pandas, so the result is live's on every supported version
        test = self._conjunction([test, cst.Comparison(x, [cst.ComparisonTarget(cst.Equal(), x)])])
        row = self._m('asof_at', o.ref(hold), start, stop, cst.Integer(str(view.depth)), x, cst.Integer(str(error)))
        self._record(call, 'L6', view.call, view=view)
        return self._guarded(whole_call or call, updated, view, test, self._label(view, hold, row), 'i64')

    def _values(self, view, hold, value, test, name):
        """``name([value(row) for each position])`` of a one-level view in pandas order (``list``: the list itself)."""
        position = cst.Name(self.o.position_name())
        start, stop, view_test = self._view_parts(view, hold)
        values = cst.ListComp(elt=value(self._row_at(view, hold, position)),
                              for_in=cst.CompFor(target=position, iter=self._range(start, stop)))
        native = values if name == 'list' else cst.Call(func=cst.Name(name), args=[cst.Arg(values)])
        return native, self._conjunction([view_test, *test])

    def _builtin(self, name, view, original, updated):
        if not self._one_level(view):
            return None
        if name == 'len':
            hold = self._take(view.call, _NO_LABEL, -1)
            start, stop, test = self._view_parts(view, hold)
            native = cst.BinaryOperation(stop, cst.Subtract(), start) if view.start is not None else stop
            self._record(original, 'L8', view.call, view=view)
            return self._guarded(original, updated, view, test, native, 'i64')
        if name in ('min', 'max'):      # T-5c: the extreme itself, no list of labels
            if view.start is None and view.schema['levels'] == ('int',):
                # the extreme of a non-empty int index (signature kind -2) as the C integer the builtin's Python
                # ints give; another level kind keeps the list (iteration gives Python floats, pandas numpy ones)
                hold = self._take(view.call, _NO_LABEL, -2)
                self._record(original, 'L5', view.call, view=view)
                return self._guarded(original, updated, view, self._native_test(hold, []),
                                     self._m(name + '_ll', self.o.ref(hold)), 'i64')
        if name in ('min', 'max') and view.start is not None:
            hold = self._walk_hold(view)        # an ordered prefix view: its first or last row (empty: pandas)
            start, stop, test = self._view_parts(view, hold)
            row = start if name == 'min' else cst.BinaryOperation(stop, cst.Subtract(), cst.Integer('1'))
            test = self._conjunction([test, cst.Comparison(stop, [cst.ComparisonTarget(cst.GreaterThan(), start)])])
            self._record(original, 'L5', view.call, view=view)
            int_level = view.schema['levels'][view.depth] == 'int'
            return self._guarded(original, updated, view, test, self._label(view, hold, row),
                                 'i64' if int_level else None)
        hold = self._walk_hold(view)
        native, test = self._values(view, hold, lambda row: self._label(view, hold, row), [], name)
        self._record(original, 'L8', view.call, view=view)
        return self._guarded(original, updated, view, test, native)

    def _column_builtin(self, name, column, original, updated):
        """``C.tolist()``, list/tuple/len of a column of a view: its values in pandas order (Python floats or ints,
        as ``tolist`` gives them)."""
        view, label, kind = column
        if not self._one_level(view):
            return None
        if name == 'len':
            return self._builtin('len', view, original, updated)
        hold = self._walk_hold(view)
        chold = self._take(view.call, label, COLUMN_KIND[kind])
        native, test = self._values(view, hold, lambda row: self._column_value(chold, kind, row),
                                    [self._positive(self.o.slot(chold))], name)
        self._record(original, 'L8', view.call, view=view)
        return self._guarded(original, updated, view, test, native)

    def _column_value(self, chold, kind, row):
        """The value of a float64 or int64 column at a stored row."""
        return self._m(kind + '_row', self.o.ref(chold), self.o.slot(chold), self._next(row))

    def _walk_target(self, target, view):
        """{name: level} of a loop target over a view's index: one name (one remaining level) or a tuple of names,
        one per remaining level; None otherwise."""
        remaining = len(view.schema['levels']) - view.depth
        if isinstance(target, cst.Name) and remaining == 1:
            return {target.value: view.depth}
        if isinstance(target, cst.Name) and 1 < remaining <= MAX_PARTS:      # T-5c: the tuple of the labels
            return {target.value: tuple(range(view.depth, view.depth + remaining))}
        if (isinstance(target, cst.Tuple) and len(target.elements) == remaining
                and all(isinstance(e, cst.Element) and isinstance(e.value, cst.Name) for e in target.elements)):
            names = [e.value.value for e in target.elements]
            return {n: view.depth + j for j, n in enumerate(names)} if len(set(names)) == len(names) else None
        return None

    def _column_of(self, expr):
        """(view, label, kind) of a float64 or int64 column of a view used as a Series: ``V["c"]``, or a local
        assigned once to one (T-5 shape B)."""
        o = self.o
        if isinstance(expr, cst.Name) and expr.value in o.columns:
            source, label = o.columns[expr.value]
        elif isinstance(expr, cst.Subscript):
            parts = _index_parts(expr)
            label = _literal_str(parts[0]) if parts and len(parts) == 1 else None
            if label is None:
                return None
            source = expr.value
        else:
            return None
        view = self._view(source)
        if view is None or view.schema['kind'] != 'frame':
            return None
        kind = dict(view.schema['columns']).get(label)
        return (view, label, kind) if kind in READ else None

    def _walk(self, it, target):
        """A walk of ``X.index``, of a column ``C`` of a view (its values) or of ``C.items()`` (label, value), in
        pandas order: (view, test, start, stop, {target name: its expression at a row} as a function of the row)."""
        view = self._index_of(it)
        if view is not None:
            names = self._walk_target(target, view)
            if names is None:
                return None
            hold = self._walk_hold(view)
            make = lambda row: {n: cst.Tuple([cst.Element(self._label(view, hold, row, level)) for level in lv])
                                if isinstance(lv, tuple) else self._label(view, hold, row, lv) for n, lv in names.items()}
            extra = []
        else:
            items = (isinstance(it, cst.Call) and not it.args and isinstance(it.func, cst.Attribute)
                     and it.func.attr.value == 'items')
            column = self._column_of(it.func.value if items else it)
            if column is None:
                return None
            view, label, kind = column
            if not self._one_level(view):
                return None
            hold = self._walk_hold(view)
            chold = self._take(view.call, label, COLUMN_KIND[kind])
            value = lambda row: self._column_value(chold, kind, row)
            if items:
                if not (isinstance(target, cst.Tuple) and len(target.elements) == 2 and all(
                        isinstance(e, cst.Element) and isinstance(e.value, cst.Name) for e in target.elements)):
                    return None
                k, v = (e.value.value for e in target.elements)
                if k == v:
                    return None
                make = lambda row: {k: self._label(view, hold, row), v: value(row)}
            elif isinstance(target, cst.Name):
                make = lambda row: {target.value: value(row)}
            else:
                return None
            extra = [self._positive(self.o.slot(chold))]
        start, stop, test = self._view_parts(view, hold)
        return view, self._conjunction([test, *extra]), start, stop, lambda row: make(self._row_at(view, hold, row))

    def zip_prefix(self, statements):
        """Walk a string prefix once for a terminal list of int/float pairs from the same view."""
        from .key_facts import _is_docstring
        head = int(bool(statements) and _is_docstring(statements[0]))
        if len(statements) - head != 2:
            return None
        assignment, returned = statements[head:]
        o = self.o
        name, subscript = o._simple_assign(assignment) or (None, None)
        if not (isinstance(subscript, cst.Subscript) and isinstance(subscript.value, cst.Attribute)
                and subscript.value.attr.value == 'loc' and isinstance(subscript.value.value, cst.Call)
                and isinstance(returned, cst.SimpleStatementLine) and len(returned.body) == 1
                and isinstance(returned.body[0], cst.Return) and isinstance(returned.body[0].value, cst.ListComp)):
            return None
        comp = returned.body[0].value
        walk, pair = comp.for_in, comp.elt
        if not (isinstance(pair, cst.Tuple) and len(pair.elements) == 2 and walk.asynchronous is None
                and not walk.ifs and walk.inner_for_in is None and isinstance(walk.target, cst.Tuple)
                and len(walk.target.elements) == 2 and isinstance(walk.iter, cst.Call)
                and isinstance(walk.iter.func, cst.Name) and walk.iter.func.value == 'zip'
                and len(walk.iter.args) == 2 and all(not arg.star and arg.keyword is None for arg in walk.iter.args)
                and name not in ('zip', 'int', 'float') and o._once(name)):
            return None
        names = []
        for target, output, convert in zip(walk.target.elements, pair.elements, ('int', 'float')):
            if not (isinstance(target, cst.Element) and isinstance(target.value, cst.Name)
                    and isinstance(output, cst.Element) and isinstance(output.value, cst.Call)
                    and isinstance(output.value.func, cst.Name) and output.value.func.value == convert
                    and len(output.value.args) == 1):
                return None
            arg = output.value.args[0]
            if (arg.star or arg.keyword is not None or not isinstance(arg.value, cst.Name)
                    or arg.value.value != target.value.value):
                return None
            names.append(target.value.value)
        if names[0] == names[1] or set(names) & {name, 'int', 'float'}:
            return None
        index, column = (arg.value for arg in walk.iter.args)
        if not (isinstance(index, cst.Attribute) and index.attr.value == 'index'
                and isinstance(index.value, cst.Name) and index.value.value == name
                and isinstance(column, cst.Subscript) and isinstance(column.value, cst.Name)
                and column.value.value == name):
            return None
        labels = _index_parts(column)
        label = _literal_str(labels[0]) if labels and len(labels) == 1 else None
        parts = _index_parts(subscript)
        if (label is None or column.slice[0].comma is not cst.MaybeSentinel.DEFAULT
                or not parts or len(parts) != 1 or subscript.slice[0].comma is not cst.MaybeSentinel.DEFAULT):
            return None
        key = parts[0]
        if not (o.per_item and isinstance(key, cst.Call) and not key.args and isinstance(key.func, cst.Attribute)
                and isinstance(key.func.value, cst.Name) and key.func.value.value == 'self'):
            return None
        key_name = key.func.attr.value
        selector = o.cls_info.cells.get(key_name)
        if (selector is None or not selector.has_formula_def or selector.has_args() or selector.is_locked
                or selector.get_rettype_expr() != 'str'):
            return None
        prefix = o.prefix_of(subscript)
        if prefix is None:
            return None
        (call, schema), _ = prefix
        provider = o.published_provider(call)
        owner = call.func.value
        if not (schema['kind'] == 'frame' and schema['levels'] == ('obj', 'int')
                and dict(schema['columns']).get(label) == 'f64' and provider is not None
                and isinstance(owner, cst.Attribute) and isinstance(owner.value, cst.Name)
                and owner.value.value == 'self'):
            return None
        cls, table_name = provider
        ref = o.cls_info.refs.get(owner.attr.value)
        if ref is None or ref.get_type_expr() != cls.name or name == cls.name:
            return None

        occupied = set(o.facts.formula.loads) | set(o.facts.formula.binds) | o.facts._params
        temps = []
        while len(temps) < 6:
            fresh = o.position_name()
            if fresh not in occupied:
                temps.append(fresh)
        data, frame, loc, snapshot, captured_key, result = map(cst.Name, temps)
        key_safe = f'(self._has_{key_name} and type(self._v_{key_name}) is str)'
        cold = getattr(selector, 'held_string_selector', None)
        if cold is not None:
            key_safe = f'({key_safe} or (not self._has_{key_name} and ({cold})))'
        guard = f'type(self) is {o.cls_info.name} and type({data.value}) is {cls.name} and {key_safe}'
        receiver = call.with_changes(func=call.func.with_changes(value=data))
        prefix_lookup = (
            cst.parse_statement(f'{data.value}: {cls.name} = {_code(owner)}'),
            self._assign(frame.value, receiver),
            self._assign(loc.value, cst.Attribute(frame, cst.Name('loc'))),
            cst.parse_statement(f'{snapshot.value} = {data.value}._mx_ptable_{table_name} if {guard} else None'),
            self._assign(captured_key.value, key),
        )
        signature = o.signature('oi', None, label, -7)
        native = self._m('zip_prepared_prefix', snapshot, frame, loc, captured_key, cst.Integer(str(signature)))
        test = cst.parse_expression(f'({result.value} := {_code(native)}) is not None')
        native_return = returned.with_changes(body=[returned.body[0].with_changes(value=result)])
        fallback_read = subscript.with_changes(value=loc, slice=[subscript.slice[0].with_changes(
            slice=subscript.slice[0].slice.with_changes(value=captured_key))])
        fallback_assignment = assignment.with_changes(body=[assignment.body[0].with_changes(value=fallback_read)])
        self._record(subscript, 'L8', call)
        o.lowered[-1]['pandas_when'] = ('the selector was not already proven safe; or the bound frame, metadata, '
                                       'prefix or column cannot use the ordered snapshot')
        o.lowered_text.add(normal(_code(column)))
        return [*statements[:head], *prefix_lookup,
                self._branch(test, [native_return], [fallback_assignment, returned])]

    def comprehension(self, original, updated):
        """A comprehension whose first ``for`` walks ``X.index``, a column of a view or its ``.items()``: positions
        in pandas order, the target names read as the labels (values) of each position's row."""
        comp = original.for_in
        position = cst.Name(f'_mx_p{self.o.local_count}')
        found = self._walk(comp.iter, comp.target)
        if found is None:
            return None
        view, test, start, stop, make = found
        labels = make(position)
        inner, rebound = comp.inner_for_in, set()
        while inner is not None:
            from .key_facts import _bound_names
            rebound |= set(_bound_names(inner.target))
            inner = inner.inner_for_in
        if rebound & set(labels):
            return None
        self.o.position_name()

        class Sub(cst.CSTTransformer):
            def leave_Name(self, o_, new):
                return labels.get(o_.value, new)

            def leave_Attribute(self, o_, new):         # x.k is not the name k
                return new.with_changes(attr=o_.attr)

            def leave_Arg(self, o_, new):
                return new.with_changes(keyword=o_.keyword)
        native = updated.with_changes(for_in=updated.for_in.with_changes(target=position, iter=self._range(start, stop)))
        native = native.with_changes(for_in=native.for_in.with_changes(
            ifs=[i.visit(Sub()) for i in native.for_in.ifs],
            inner_for_in=native.for_in.inner_for_in.visit(Sub()) if native.for_in.inner_for_in else None))
        if isinstance(native, cst.DictComp):
            native = native.with_changes(key=native.key.visit(Sub()), value=native.value.visit(Sub()))
        else:
            native = native.with_changes(elt=native.elt.visit(Sub()))
        parens = {'lpar': [cst.LeftParen()], 'rpar': [cst.RightParen()]}
        if isinstance(original, cst.GeneratorExp):
            native, updated = native.with_changes(**parens), updated.with_changes(**parens)
        self._record(original, 'L8', view.call, view=view)
        lowered = self._guarded(original, updated, view, test, native)
        if isinstance(original, cst.GeneratorExp):
            self.generator_walks.add(id(lowered))
        return lowered

    def leave_ListComp(self, original, updated):
        return self.comprehension(original, updated) or updated

    def leave_SetComp(self, original, updated):
        return self.comprehension(original, updated) or updated

    def leave_DictComp(self, original, updated):       # T-5c
        return self.comprehension(original, updated) or updated

    def leave_GeneratorExp(self, original, updated):
        return self.comprehension(original, updated) or updated

    def leave_For(self, original, updated):
        """``for k in X.index:`` over a view: a loop over positions in pandas order, ``k`` the label of each
        position's row; the pandas loop when the table cannot answer."""
        if original.asynchronous is not None:
            return updated
        starred = _starred_target(original.target)
        if starred is not None:
            return starred(updated, self.o)
        rows = self.iterrows_loop(original, updated)
        if rows is not None:
            return rows
        position = cst.Name(f'_mx_p{self.o.local_count}')
        found = self._walk(original.iter, original.target)
        if found is None:
            return updated
        view, test, start, stop, make = found
        self.o.position_name()
        self.o.declarations.append(position.value)
        native = self._loop(updated, position, self._range(start, stop),
                            [self._assign(n, expr) for n, expr in make(position).items()])
        self._record(original, 'L8', view.call, view=view)
        return self._branch(test, [native], [self._fallback(updated, view.call)])

    @staticmethod
    def _branch(test, native, fallback):
        """``if test: native else: fallback`` (statements)."""
        return cst.If(test=test, body=cst.IndentedBlock(body=native), orelse=cst.Else(body=cst.IndentedBlock(body=fallback)))

    def iterrows_loop(self, original, updated):
        """``for k, row in R.iterrows():`` (T-5 shape D): positions in pandas order, ``k`` the label and ``row`` the
        stored row (``_mx_rw_row``), its fields read as a row local's; the pandas loop (``_mx_rw_row`` -1) when the
        table is not of the traced kinds (every column checked)."""
        o = self.o
        target = original.target         # any other target shape: not this form (row locals found only `k, row`)
        if not (isinstance(target, cst.Tuple) and len(target.elements) == 2
                and all(isinstance(e, cst.Element) and isinstance(e.value, cst.Name) for e in target.elements)):
            return None
        key, name = (e.value.value for e in target.elements)
        found = o.row_locals.get(name)
        if found is None or found[2] is not original:
            return None
        call, schema, _ = found
        hold = self._take(call, _NO_LABEL, -4)
        hold.row_columns = True
        view = _View(None, call, schema)
        var = f'_mx_rw_{name}'
        if var not in o.declarations:
            o.declarations.append(var)
        position = cst.Name(o.position_name())
        o.declarations.append(position.value)
        row = self._m('whole_row', o.ref(hold), position)
        native = self._loop(updated, position, self._range(self._m('nrows', o.ref(hold))),
                            [self._assign(var, row), self._assign(key, self._label(view, hold, row))])
        if isinstance(original.iter.func.value, cst.Name):
            self.replaced_names.add(original.iter.func.value.value)
        fallback = [cst.parse_statement(f'{var} = -1'), self._fallback(updated, call)]
        self._record(original.iter, 'L8', call)
        unbound = cst.parse_statement(f'{name} = None')    # only its guarded pandas reads name it
        return self._branch(self._native_test(hold, []), [unbound, native], fallback)

    def view_read(self, original, updated, value, parts):
        """``V.loc[k, "c"]`` on a view local (not a prefix cell's): the prefix codes and k's code."""
        o = self.o
        view = o.views.get(value.value.value) if isinstance(value.value, cst.Name) else None
        if view is None or view.cell is not None or len(parts) != 2:
            return None
        view.used = True
        label = _literal_str(parts[1])
        if label is None:
            named = self.named_read(original, updated, view.call, view.schema, parts[0], parts[1], view.codes)
            if named is not None:       # the view is valid (start >= 0) whenever its prefix codes are
                start, stop, test = self._view_parts(view, self._take(view.call, _NO_LABEL, -1))
                return named.with_changes(test=self._conjunction([test, named.test]))
            return None
        kind = dict(view.schema['columns']).get(label) if view.schema['kind'] == 'frame' else None
        own = _parts(parts[0])
        levels = view.schema['levels']
        if len(own) == 1 and view.depth + 1 < len(levels):     # T-5c: a key tuple held at run time
            own = o.tuple_key(parts[0], len(levels) - view.depth) or own
        if label is None or kind not in READ or view.depth + len(own) != len(levels):
            return None
        hold = self._take(view.call, label, COLUMN_KIND[kind])
        start, stop, test = self._view_parts(view, hold)
        conditions = o.guards(own)
        codes = list(view.codes) + [self._held(hold, level, part, conditions) if o.item_fixed(part, levels[level])
                                    else self.code_of(hold, level, part, levels[level])
                                    for level, part in enumerate(own, view.depth)]
        suffix, args = self.code_args(codes)
        test = self._conjunction([test, *conditions])
        native = self._m(kind + suffix, o.ref(hold), o.slot(hold), *args)
        self._record(original, 'L3', view.call, view=view)
        return self._guarded(original, updated, view, test, native, None if o.guards(own) else kind)

    # ------------------------------------------------------------------ T-5: row locals
    def leave_SimpleStatementLine(self, original, updated):
        """``row = R.loc[key]``: the stored row in a C local (KeyError as pandas); the pandas row only when the
        table is not of the kinds the build traced (every column checked: signature kind -4)."""
        o = self.o
        if not (len(original.body) == 1 and isinstance(original.body[0], cst.Assign)):
            return updated
        name = original.body[0].targets[0].target
        constants = o.enum_key_locals.get(name.value) if isinstance(name, cst.Name) else None
        if constants:       # T-5c: ``item = E.a if c else E.b`` used only as key parts: bound to C constants

            class Constants(cst.CSTTransformer):
                def leave_Attribute(self, o_, new):
                    constant = constants.get(_code(o_))
                    return cst.Name(constant) if constant is not None else new
            return updated.with_changes(body=[updated.body[0].with_changes(
                value=updated.body[0].value.visit(Constants()))])
        found = o.row_locals.get(name.value) if isinstance(name, cst.Name) else None
        if found is None or found[2] is not original.body[0]:
            return updated
        call, schema, assign = found
        hold = self._take(call, _NO_LABEL, -4)
        hold.row_columns = True
        key = _index_parts(assign.value)[0]
        var = f'_mx_rw_{name.value}'
        if var not in o.declarations:
            o.declarations.append(var)
        fallback = [cst.parse_statement(f'{var} = -1'),
                    original.with_changes(body=[original.body[0].with_changes(
                        value=self._fallback(original.body[0].value, call))])]
        self._record(assign.value, 'L2', call)
        parts = _parts(key)
        if (o.per_item and o.cells.has_args() and len(parts) == len(schema['levels'])
                and all(o.fixed_part(p) for p in parts)):
            # T-5c: a key fixed per item (``row = T.loc[cost_id()]`` in a per-step cell): the row held per item,
            # looked up once (as a row cell's), not at every step
            field = o.row_field(hold, key, None, self._detached(key))
            native = self._assign(var, cst.parse_expression(f'self.{field} - 1'))
            return self._branch(self._native_test(hold, [cst.parse_expression(f'self.{field} > 0')]), [native],
                                fallback)
        labels = []
        suffix, args = self._codes(hold, [], parts, schema['levels'], labels)
        lookup = self._m('row' + suffix, o.ref(hold), *args)
        native = self._assign(var, lookup)
        return self._branch(self._native_test(hold, labels), [native], fallback)

    def row_local_read(self, original, updated, parts):
        """``row["c"]`` of a row local: the field from the prepared blocks, of the kind pandas gives it (as L2)."""
        o = self.o
        value = original.value
        found = o.row_locals.get(value.value) if isinstance(value, cst.Name) else None
        if found is None or len(parts) != 1:
            return None
        call, schema, _ = found
        label = _literal_str(parts[0])
        if label is None:           # float(row[name]): the field chosen at run time
            hold = self._take(call, _NO_LABEL, -4)
            hold.row_columns = True
            read = self._m('field_f64', o.ref(hold), cst.parse_expression(f'_mx_rw_{value.value} + 1'),
                           _cast('object', self.updated_of.get(id(parts[0]), parts[0])))
            self._record(original, 'L2', call)
            return self._lowered(cst.parse_expression(f'_mx_rw_{value.value} >= 0'), read, updated, 'f64')
        read, kind, _ = self._field_read(call, schema, label, lambda hold: (
            cst.parse_expression(f'_mx_rw_{value.value} + 1'), []))
        self._record(original, 'L2', call)
        return self._lowered(cst.parse_expression(f'_mx_rw_{value.value} >= 0'), read, updated, kind)

    def _field_read(self, call, schema, label, position):
        """Field ``label`` of a stored row, of the kind pandas gives it: the column's own on a row with an object
        column, else the row's common kind (an int column of a float row reads as a float). ``position(hold)``:
        (the row + 1, its conditions). (read, kind, conditions)."""
        o = self.o
        kinds = dict(schema['columns'])
        mixed = any(k == 'obj' for k in kinds.values())
        kind = kinds[label] if mixed else ('f64' if any(k == 'f64' for k in kinds.values()) else 'i64')
        stored = kinds[label]           # the block the column is stored in
        hold = self._take(call, label, {'f64': 0, 'i64': 1, 'obj': 2}[stored])
        hold.row_columns = True         # the row's dtype depends on every column: all of them are checked
        row, conditions = position(hold)
        read = self._m({'f64': 'f64_row', 'i64': 'i64_row', 'obj': 'object_field'}[stored], o.ref(hold), o.slot(hold), row)
        if kind == 'f64' and stored == 'i64':
            read = _cast('_mx_cy.double', read)
        return read, kind, self._native_test(hold, conditions)

    def column_keyed(self, original, updated, parts):
        """L4 (T-5): ``R["c"][k]`` / ``R.c[k]`` of a float64 or int64 column: ``R.loc[k, "c"]`` (the same rule for an
        integer key on an index without an integer level as ``S[k]``)."""
        o = self.o
        column = original.value
        if isinstance(column, cst.Subscript):
            inner = _index_parts(column)
            label = _literal_str(inner[0]) if inner and len(inner) == 1 else None
        elif isinstance(column, cst.Attribute):
            label = column.attr.value
        else:
            return None
        found = o._receiver(column.value) if label is not None else None
        if found is None or len(parts) != 1 or found[1]['kind'] != 'frame' or not found[1].get('unique') \
                or _code(found[0]) in o.mutated:
            return None
        call, schema = found
        import pandas
        kind = dict(schema['columns']).get(label)
        if kind not in READ or (isinstance(column, cst.Attribute) and hasattr(pandas.DataFrame, label)):
            return None
        if schema['levels'] != ('int',) and len(schema['levels']) == 1 and o.facts.is_int(parts[0]):
            return None
        return self._column_read(original, updated, call, schema, parts[0], label, kind, column.value, 'L4')

    def named_read(self, original, updated, call, schema, key, name, prefix):
        """T-5 (shape A): ``R.loc[key, name]`` with the column chosen at run time, every column of the frame of one
        kind (float64 or int64; checked for the whole table once, signature kind -4): a typed read whose slot is looked
        up by name (KeyError as pandas). ``prefix``: the codes of a view's prefix, before the key's."""
        o = self.o
        kinds = {kind for _, kind in schema['columns']}
        up = o.facts.formula.parents.get(id(original))
        consumer = o.facts.formula.parents.get(id(up)) if isinstance(up, cst.Arg) else None
        under_float = (isinstance(consumer, cst.Call) and isinstance(consumer.func, cst.Name)
                       and consumer.func.value == 'float' and len(consumer.args) == 1)
        if schema['kind'] != 'frame' or not ((len(kinds) == 1 and kinds <= set(READ)) or under_float):
            return None
        kind = kinds.pop() if len(kinds) == 1 and kinds <= set(READ) else None
        levels = schema['levels']
        own = [e.value for e in key.elements] if isinstance(key, cst.Tuple) and len(levels) - len(prefix) > 1 else [key]
        if len(prefix) + len(own) != len(levels) or any(isinstance(p, (cst.Slice, cst.List, cst.ListComp, cst.Tuple))
                                                       for p in own):
            return None
        hold = self._take(call, _NO_LABEL, -4)
        hold.row_columns = True
        conditions = []
        if prefix:
            suffix, args = self._codes(hold, prefix, own, levels, conditions)
        else:
            (suffix, args), conditions = self._read_args(hold, own, levels)
        label = _cast('object', self.updated_of.get(id(name), name))
        if kind is None:            # float(R.loc[key, name]) on columns of several kinds: float of the stored value
            row = self._next(self._m('row' + suffix, o.ref(hold), *args))
            read, kind = self._m('field_f64', o.ref(hold), row, label), 'f64'
        else:
            read = self._m(kind + suffix, o.ref(hold), self._m('named_slot', o.ref(hold), label), *args)
        self._record(original, 'L1', call)
        return self._lowered(self._native_test(hold, conditions), read, self._fallback(updated, call), kind)

    def column_item(self, original, updated, parts):
        """``C.iloc[i]`` / ``C.loc[k]`` of a column of a view in a local (a float64 or int64 column): the value at
        position i (pandas order) or label k."""
        o = self.o
        value = original.value
        if not (isinstance(value, cst.Attribute) and value.attr.value in ('iloc', 'loc') and len(parts) == 1):
            return None
        column = self._column_of(value.value) if isinstance(value.value, cst.Name) and value.value.value in o.columns \
            else None
        if column is None:
            return None
        view, label, kind = column
        if not self._one_level(view):
            return None
        part = parts[0]
        emit = self.updated_of.get(id(part), part)
        chold = self._take(view.call, label, COLUMN_KIND[kind])
        if value.attr.value == 'iloc':
            if not (o.facts.is_int(part) or self.native.get(id(emit)) == 'i64') or (
                    isinstance(part, cst.Name) and part.value in ('True', 'False')):
                return None
            hold = self._walk_hold(view)
            start, stop, test = self._view_parts(view, hold)
            row = self._row_at(view, hold, self._m('position', start, stop, _cast('_mx_cy.Py_ssize_t', emit)))
            read = self._column_value(chold, kind, row)
        else:
            if view.start is None or isinstance(part, (cst.Slice, cst.List, cst.ListComp, cst.Tuple)):
                return None
            start, stop, test = self._view_parts(view, self._take(view.call, _NO_LABEL, -1))
            labels = []
            suffix, args = self._codes(chold, view.codes, [part], view.schema['levels'], labels)
            read = self._m(kind + suffix, o.ref(chold), o.slot(chold), *args)
            test = self._conjunction([test, *labels])
        test = self._conjunction([test, self._positive(o.slot(chold))])
        self._record(original, 'L8' if value.attr.value == 'iloc' else 'L3', view.call, view=view)
        return self._guarded(original, updated, view, test, read, kind)

    def raised_veto(self, node, receiver):
        """Decision from tracing: a place whose every evaluation raised in the sample is not lowered (it stays as
        written, recorded with its reason). A place that also returned in the sample is not vetoed by this."""
        if not only_raised(self.o.note_of(node)):
            return False
        self._record(node, 'pandas', receiver, ONLY_RAISED_WHY)
        return True

    def leave_Subscript(self, original, updated):
        o = self.o
        value = original.value
        if self.native.get(id(updated.value)) in CTYPE and isinstance(updated.value, cst.IfExp):
            # T-5c: a subscript of a read lowered to a C scalar (``T.index.max()[0]`` on one level): the pandas
            # value, which pandas subscripts (or raises on) as live does
            pandas = updated.value.orelse
            if (isinstance(pandas, cst.Call) and _code(pandas.func) == '_mx_cy.cast' and len(pandas.args) == 2
                    and _code(pandas.args[0].value) in CTYPE.values()):
                pandas = pandas.args[1].value       # the read's own value, not cast to the C type
            return updated.with_changes(value=pandas.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()]))
        parts = _index_parts(original)
        if parts is None:
            return updated
        note = o.note_of(original)
        if note is not None:
            # decisions from tracing: a read that took a mask, a list or a slice in the sample is not a single-cell
            # read, whatever its text looks like; it stays on pandas
            from .read_notes import selects_rows
            if any(len(seen) == 4 and selects_rows(seen[1]) for seen in note['seen']):
                self._record(original, 'pandas', value, SEVERAL_ROWS_WHY)
                return updated
        if (len(parts) == 1 and isinstance(parts[0], cst.Name) and parts[0].value in ('True', 'False')
                and isinstance(value, cst.Attribute) and value.attr.value in PANDAS_INDEXERS):
            # T-5c: a bool literal as the key of a pandas indexer: Cython compiles X[True] as the integer index 1;
            # the call passes the Python bool, as live does (pandas raises or selects by it)
            return cst.Call(func=cst.Attribute(updated.value, cst.Name('__getitem__')), args=[cst.Arg(parts[0])])
        if self.raised_veto(original, value):     # (after the bool key: Cython would read X[False] as X[0])
            return updated
        row = (self.row_local_read(original, updated, parts) or self.row_cell_field(original, updated, parts)
               or self._row_field(original, updated, parts))
        if row is not None:
            return row
        part = self.extreme_part(original, updated, parts)
        if part is not None:
            return part
        walk = (self.index_item(original, updated) or self.column_keyed(original, updated, parts)
                or self.column_item(original, updated, parts))
        if walk is not None:
            return walk
        if isinstance(value, cst.Attribute) and value.attr.value in ('loc', 'at'):
            read = self.view_read(original, updated, value, parts)
            if read is not None:
                return read
        # T-5c (shape G): P(a).loc[k, "c"], P a prefix cell of parameters: T.loc[(key(a), k), "c"] on its parent
        param = o.param_prefix(value.value) if isinstance(value, cst.Attribute) and value.attr.value in (
            'loc', 'at') and len(parts) == 2 else None
        label = _literal_str(parts[1]) if param is not None else None
        if label is not None:
            call, schema, prefix_parts = param
            kind = dict(schema['columns']).get(label)
            if kind in READ:
                key = cst.Tuple([cst.Element(p) for p in [*prefix_parts, *_parts(parts[0])]])
                read = self._column_read(original, updated, call, schema, key, label, kind, value.value, 'L1')
                if read is not None:
                    o.param_readers.add(value.value.func.attr.value)
                    return read
                return updated
        # R.loc[key, "c"] / R.at[key, "c"] on a frame; R.loc[k] / R.at[k] / R[k] on a Series
        if isinstance(value, cst.Attribute) and value.attr.value in ('loc', 'at'):
            prefix = o.prefix_receiver(value.value) if len(parts) == 2 else None
            if prefix is not None:      # before the frame itself: a prefix cell's value is a new frame per item
                return self._prefix_read(original, updated, value, parts, prefix)
            found = o._receiver(value.value)
            form = value.attr.value
        else:
            found = o._receiver(value)
            form = 'item'
        if found is None:
            return updated
        call, schema = found
        if not self._plain(original, call, schema):
            return updated
        if schema['kind'] == 'series':
            if len(parts) != 1:
                return updated
            key, label = parts[0], None
            if form == 'item' and schema['levels'] != ('int',) and o.facts.is_int(key):
                self._record(original, 'pandas', call, 'an integer key on a Series without an integer index')
                return updated
        else:
            if form == 'item' or len(parts) != 2:
                return updated      # a column, a row, or another selection: pandas (T-4/T-5 forms)
            key, label = parts
            label = _literal_str(label)
            if label is None:
                named = self.named_read(original, updated, call, schema, key, parts[1], [])
                if named is not None:
                    return named
                self._record(original, 'pandas', call, 'the column is not a literal name, and the columns are '
                             'not all of one kind (float64 or int64)')
                return updated
        kind = dict(schema['columns']).get(label)
        if kind == 'obj':
            return self.object_read(original, updated, value, form, call, schema, key, label)
        if kind not in READ:
            self._record(original, 'pandas', call, (
                'the Series values are not float64 or int64 in every trace (a row read by field: form L2)'
                if schema['kind'] == 'series' else f'column {label!r} is not float64 or int64 in every trace'))
            return updated
        return self._column_read(original, updated, call, schema, key, label, kind,
                                 value.value if form != 'item' else value, 'L1') or updated

    def _plain(self, original, call, schema):
        """The frame ``call`` is read as stored: not edited by the formula, of a unique index (else the reason)."""
        why = ('the formula edits this frame in place' if _code(call) in self.o.mutated
               else None if schema.get('unique') else 'the index is not unique (pandas)')
        if why:
            self._record(original, 'pandas', call, why)
        return why is None

    def _column_read(self, original, updated, call, schema, key, label, kind, receiver, form):
        """A typed read of column ``label`` at a complete ``key`` of the frame ``call`` (L1, L4); None, with the
        reason recorded for L1, when the key is not one."""
        o = self.o
        key_parts, why = self._key(key, schema)
        if why:
            if form == 'L1':
                self._record(original, 'pandas', call, why)
            return None
        hold = self._take(call, label, COLUMN_KIND[kind])
        if isinstance(receiver, cst.Name):
            self.replaced_names.add(receiver.value)
        (suffix, args), conditions = self._read_args(hold, key_parts, schema['levels'])
        read = self._m(kind + suffix, o.ref(hold), o.slot(hold), *args)
        self._record(original, form, call)
        # T-5c: with a key tuple checked at run time the pandas branch may be a prefix (a Series): the read is boxed
        typed = None if o.guards(key_parts) else kind
        return self._lowered(self._native_test(hold, conditions), read, self._fallback(updated, call), typed)

    def object_read(self, original, updated, value, form, call, schema, key, label):
        """T-5: ``R.loc[key, "c"]`` of an object column (or ``S[k]`` of an object Series): the object pandas returns,
        the stored value (a new reference). A key fixed per item holds its row per item."""
        o = self.o
        key_parts, why = self._key(key, schema)
        if why:
            self._record(original, 'pandas', call, why)
            return updated
        hold = self._take(call, label, 2)
        receiver = value.value if form != 'item' else value
        if isinstance(receiver, cst.Name):
            self.replaced_names.add(receiver.value)
        if o.per_item and all(o.item_fixed(part, kind) for part, kind in zip(key_parts, schema['levels'])):
            field = o.row_field(hold, key, None, self._detached(key))
            read = self._m('object_field', o.ref(hold), o.slot(hold), cst.parse_expression(f'self.{field}'))
            conditions = [cst.parse_expression(f'self.{field} > 0')]
        else:
            (suffix, args), conditions = self._read_args(hold, key_parts, schema['levels'])
            row = self._next(self._m('row' + suffix, o.ref(hold), *args))
            read = self._m('object_field', o.ref(hold), o.slot(hold), row)
        self._record(original, 'L1', call)
        return self._lowered(self._native_test(hold, conditions), read, self._fallback(updated, call))

    def leave_Call(self, original, updated):
        """``float(...)`` around a float64 read and ``int(...)`` around an int64 read are dropped: the read is
        already that C value, and the builtin would box it into a Python object and back. L5: the extremes of a
        one-level int index are read as C integers."""
        target = original
        inner = self.o.facts.int_argument(original)
        if isinstance(inner, cst.Call):
            target = inner         # int(X.index.asof(x)): the place is the asof call
        if (isinstance(target.func, cst.Attribute) and target.func.attr.value in ('searchsorted', 'asof', 'max', 'min')
                and self.raised_veto(target, target.func.value)):
            return updated
        walk = self.index_call(original, updated) or self.level_walk_extreme(original, updated)
        if walk is not None:
            return walk
        found = self.o.extreme(original)
        if found is not None:
            call, which = found
            o = self.o
            hold = self._take(call, _NO_LABEL, -2)
            if isinstance(original.func.value.value, cst.Name):
                self.replaced_names.add(original.func.value.value.value)
            read = self._m(which + '_ll', o.ref(hold))
            self._record(original, 'L5', call)
            return self._lowered(self._native_test(hold, []), read, self._fallback(updated, call), 'i64')
        hoisted = self._hoist_walk(updated)
        if hoisted is not None:
            return hoisted
        if (isinstance(updated.func, cst.Name) and len(updated.args) == 1 and not updated.args[0].keyword
                and not updated.args[0].star):
            kind = self.native.get(id(updated.args[0].value))
            if (kind == 'f64' and updated.func.value == 'float') or (kind == 'i64' and inner is not None):
                return updated.args[0].value.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
        return updated

    def _hoist_walk(self, updated):
        """T-5c: ``f(gen if test else pandas_gen, ...)`` for a builtin f of one iterable -> ``f(gen, ...) if test else
        f(pandas_gen, ...)``: a generator passed straight to the builtin is a loop for Cython (and for the reduction
        lowering), where the generator behind a conditional runs as a generator object."""
        if not (isinstance(updated.func, cst.Name) and updated.func.value in WALK_BUILTINS and updated.args
                and not updated.args[0].keyword and not updated.args[0].star
                and all(a.keyword is not None for a in updated.args[1:])
                and id(updated.args[0].value) in self.generator_walks):
            return None
        walk = updated.args[0].value
        def call(arg):
            if len(updated.args) == 1:  # a lone generator argument needs no parentheses of its own
                arg = arg.with_changes(lpar=[], rpar=[])
            return updated.with_changes(args=[updated.args[0].with_changes(value=arg),
                                              *updated.args[1:]], lpar=[], rpar=[])
        return cst.IfExp(test=walk.test, body=call(walk.body), orelse=call(walk.orelse),
                         lpar=[cst.LeftParen()], rpar=[cst.RightParen()])

    def _prefix_holds(self, hold, prefix_parts, name):
        conditions = []
        codes = [self._held(hold, level, part, conditions) for level, part in enumerate(prefix_parts)]
        return codes, conditions

    def asof_statement(self, statement, asof):
        """``lo = P.index.asof(x)`` -> ``_mx_asof_lo = <row of the label in the parent table, or -1>``, through a
        membership hold of the parent (codes of the prefix held per item)."""
        o = self.o
        (pcall, call, schema, prefix_parts, name), x, original = asof
        target = statement.body[0].targets[0].target.value
        hold = self._take(call, len(prefix_parts), -3)     # T-5: asof needs pandas order within the prefix
        codes, conditions = self._prefix_holds(hold, prefix_parts, name)
        search = self._m('asof_row', o.ref(hold), cst.Integer(str(len(prefix_parts))), *self._padded(codes), _cast('_mx_cy.double', x))
        value = cst.IfExp(test=self._native_test(hold, conditions), body=search, orelse=cst.parse_expression('-1'))
        self.asof_rows[target] = (hold, pcall, original)
        # the fallback of a read evaluates the pandas asof again, on the cell's value
        receiver = original.func.value.value
        self.swaps.setdefault(target, original.with_changes(func=original.func.with_changes(
            value=original.func.value.with_changes(value=pcall))) if isinstance(receiver, cst.Name) else original)
        if isinstance(receiver, cst.Name):
            self.swaps[receiver.value] = pcall
        self._record(original, 'L6', pcall)
        return cst.parse_statement(f'_mx_asof_{target}: _mx_cy.Py_ssize_t = {_code(value)}')

    def _prefix_read(self, original, updated, value, parts, prefix):
        """``P.loc[k, "c"]`` on the value of a prefix cell P (``T.loc[prefix]``): ``T.loc[prefix + k, "c"]`` with
        the prefix parts held as codes per item, while P holds no assigned value (checked once per item)."""
        o = self.o
        pcall, call, schema, prefix_parts, name = prefix
        label = _literal_str(parts[1])
        kind = dict(schema['columns']).get(label)
        if label is None or kind not in READ or not schema.get('unique') or _code(call) in o.mutated:
            self._record(original, 'pandas', pcall, 'a prefix read of a column that is not float64 or int64, '
                         'on a non-unique index, or of a frame the formula edits')
            return updated
        key = parts[0]
        if isinstance(key, cst.Name) and key.value in self.asof_rows:     # a key found by a lowered asof
            return self._asof_read(original, updated, value, label, kind, call, pcall, key.value)
        own = _parts(key)
        levels = schema['levels']
        if len(prefix_parts) + len(own) != len(levels) or not o.per_item:
            self._record(original, 'pandas', pcall, 'the prefix and the key do not make a complete key'
                         if o.per_item else 'a prefix cell outside an item space')
            return updated
        hold = self._take(call, label, COLUMN_KIND[kind])
        codes, conditions = self._prefix_holds(hold, prefix_parts, name)
        conditions = list(conditions)
        suffix, args = self._codes(hold, codes, own, levels, conditions)
        read = self._m(kind + suffix, o.ref(hold), o.slot(hold), *args)
        if isinstance(value.value, cst.Name):
            self.swaps[value.value.value] = pcall
        self._record(original, 'L3', pcall)
        return self._lowered(self._native_test(hold, conditions), read, self._fallback(updated, call), kind)

    def _asof_read(self, original, updated, value, label, kind, call, pcall, name):
        o = self.o
        asof_hold = self.asof_rows[name][0]
        hold = self._take(call, label, COLUMN_KIND[kind])
        row = f'_mx_asof_{name}'
        read = self._m(kind + '_row', o.ref(hold), o.slot(hold), cst.parse_expression(f'{row} + 1'))
        # the row is a row of the table the asof searched: the same prepared table object
        test = self._conjunction([cst.parse_expression(f'{row} >= 0'), self._native_test(hold, []),
                                  cst.parse_expression(f'self.{hold.table} is self.{asof_hold.table}')])
        if isinstance(value.value, cst.Name):
            self.swaps[value.value.value] = pcall
        self._record(original, 'L3', pcall)
        return self._lowered(test, read, self._fallback(updated, call), kind)

    def row_cell_field(self, original, updated, parts):
        """T-5: ``self.P()["f"]`` with P a row cell (``return T.loc[key]``, the key complete and fixed per item) that
        is not inlined (one of its reads takes a computed field): the row held per item, as L2; a computed field under
        ``float()`` reads ``float(row[f])`` from the prepared blocks."""
        o = self.o
        call = original.value
        local = call.value if isinstance(call, cst.Name) else None
        if local is not None:                   # T-5c: ``row = self.P()`` then ``row["f"]``
            call = o._value_of(local) or call
        if not (o.per_item and isinstance(call, cst.Call) and not call.args and isinstance(call.func, cst.Attribute)
                and isinstance(call.func.value, cst.Name) and call.func.value.value == 'self'):
            return None
        name = call.func.attr.value
        row = o.prefix_cells.get(name)
        if row is None or not o.cached_call(call) or len(parts) != 1:
            return None
        if _literal_str(parts[0]) is not None:
            if local is not None:       # the local is P() for the pandas branch, and is dropped once unread
                self.swaps.setdefault(local, call)
            return self._row_field(original.with_changes(value=row), updated, parts, cell=name)
        up = o.facts.formula.parents.get(id(original))
        consumer = o.facts.formula.parents.get(id(up)) if isinstance(up, cst.Arg) else None
        if not (isinstance(consumer, cst.Call) and isinstance(consumer.func, cst.Name) and consumer.func.value == 'float'
                and len(consumer.args) == 1):
            return None
        found = o._receiver(row.value.value)
        key = _index_parts(row)[0]
        if found is None or found[1]['kind'] != 'frame' or not found[1].get('unique') or _code(found[0]) in o.mutated:
            return None
        levels = found[1]['levels']
        from .row_cells import item_constant
        fixed = key.elements if isinstance(key, cst.Tuple) else [key]
        if (len(fixed) if isinstance(key, cst.Tuple) else 1) != len(levels) or not all(
                o.fixed_part(e.value if isinstance(e, cst.Element) else e) for e in fixed):
            return None
        hold = self._take(found[0], _NO_LABEL, -1)
        field = o.row_field(hold, key, name, self._detached(key))
        label = self.updated_of.get(id(parts[0]), parts[0])
        read = self._m('field_f64', o.ref(hold), cst.parse_expression(f'self.{field}'), _cast('object', label))
        self._record(original, 'L2', found[0])
        return self._lowered(self._native_test(hold, [cst.parse_expression(f'self.{field} > 0')]), read, updated, 'f64')

    def _row_field(self, original, updated, parts, cell=None):
        """L2 (T-4): ``R.loc[key]["f"]``, a field of a row of a frame, the key fixed per item (the model-point row,
        also as inlined from its row cell): the row is held per item (``_mx_tw_``) and the field read from the
        prepared table, of the kind pandas gives it: the column's own on a row with an object column, else the
        row's common kind (an int column of a float row reads as a float). None when not this form."""
        o = self.o
        row = original.value
        if not (len(parts) == 1 and isinstance(row, cst.Subscript)
                and isinstance(row.value, cst.Attribute) and row.value.attr.value == 'loc'):
            return None
        label = _literal_str(parts[0])
        key_parts = _index_parts(row)
        found = o._receiver(row.value.value)
        if label is None or key_parts is None or len(key_parts) != 1 or found is None:
            return None
        call, schema = found
        if schema['kind'] != 'frame':
            return None
        if not self._plain(original, call, schema):
            return updated
        key = key_parts[0]
        levels = schema['levels']
        fixed = (key.elements if isinstance(key, cst.Tuple) else [key])
        if (isinstance(key, cst.Tuple) and len(key.elements) != len(levels)) or (
                not isinstance(key, cst.Tuple) and len(levels) != 1):
            self._record(original, 'pandas', call, 'the row key is not complete')
            return updated
        from .row_cells import item_constant
        # T-5: a key not fixed per item finds its row per read (KeyError as pandas when it has none)
        per_read = not (o.per_item and all(o.fixed_part(e.value if isinstance(e, cst.Element) else e)
                                           for e in fixed))
        kinds = dict(schema['columns'])
        if label not in kinds:
            self._record(original, 'pandas', call, f'column {label!r} is not in every traced frame')
            return updated
        if isinstance(row.value.value, cst.Name):
            self.replaced_names.add(row.value.value.value)
        name = cell or (o.row_names.get(_code(row)) if o.row_names else None)

        held_state = None

        def position(hold):
            nonlocal held_state
            if per_read:
                labels = []
                suffix, cargs = self._codes(hold, [], _parts(key), levels, labels)
                return self._next(self._m('row' + suffix, o.ref(hold), *cargs)), labels
            field = o.row_field(hold, key, name, self._detached(key))
            held_state = hold.state
            return cst.parse_expression(f'self.{field}'), [cst.parse_expression(f'self.{field} > 0')]
        read, kind, test = self._field_read(call, schema, label, position)
        if held_state is not None and kind == 'obj' and not o.cells.has_args() and o.cells.get_rettype_expr() == 'str':
            self.held_string_read = read, test, held_state
        self._record(original, 'L2', call)
        return self._lowered(test, read, self._fallback(updated, call), kind)

    @staticmethod
    def _conjunction(conditions):
        test = conditions[0]
        for condition in conditions[1:]:
            test = cst.BooleanOperation(left=test, operator=cst.And(), right=condition)
        return test

    def _fallback(self, updated, call):
        """The original pandas read on the frame itself: a frame local replaced by its Cell call, a key local by
        its tuple display (so neither local is evaluated on the native path)."""
        frames, keys, swaps = self.replaced_names, self.key_locals, self.swaps

        class Swap(cst.CSTTransformer):
            def leave_Name(self, original, new):
                if original.value in swaps:     # a local standing for another (a column of a view local)
                    return swaps[original.value].visit(Swap())
                if original.value in frames:
                    return call
                if original.value in keys:
                    return keys[original.value]
                return new
        return updated.visit(Swap()) if (frames or keys or swaps) else updated

    def prefix_membership(self, original, updated, index, target):
        """T-5: ``k in X.index`` where the key and X's prefix are fewer parts than the levels (pandas: the prefix has
        rows), or X is a view and they make a complete key: through the view's codes and k's."""
        o = self.o
        name = index.value.value if isinstance(index.value, cst.Name) else None
        view = o.views.get(name) if name else None
        if view is None:
            found = o._receiver(index.value)
            if found is None or not found[1].get('unique') or _code(found[0]) in o.mutated:
                return None
            view = _View(None, *found)
        key = original.left
        remaining = len(view.schema['levels']) - view.depth
        runtime = None
        if remaining > 1 and not isinstance(key, cst.Tuple) and not (
                o.facts.is_int(key) or _literal_str(key) is not None or o.facts.enum_constant(key) is not None):
            # T-5c: a key held in a local or returned by a cell: a tuple of every remaining level is a complete key,
            # another tuple is pandas' own, any other value a one-level prefix
            runtime = o.tuple_key(key, remaining)
            if runtime is None:
                return None
        own = [e.value for e in key.elements] if isinstance(key, cst.Tuple) and remaining > 1 else [key]
        if runtime is not None and _code(runtime[0]) not in o.tuple_guards:
            own, runtime = runtime, None        # a local bound once to a tuple display: its parts, as a display
        levels = view.schema['levels']
        count = view.depth + len(own)
        if (view.start is None and count >= len(levels)) or count > len(levels) or any(
                isinstance(p, (cst.StarredElement, cst.Slice, cst.List, cst.ListComp)) for p in own):
            return None         # a complete key on the whole table: L7
        view.used = True
        hold = self._take(view.call, _NO_LABEL, -1)

        def membership(parts):
            codes = list(view.codes) + [self._held(hold, level, part, []) if o.item_fixed(part, levels[level])
                                        else self.code_of(hold, level, part, levels[level])
                                        for level, part in enumerate(parts, start=view.depth)]
            if view.depth + len(parts) == len(levels):
                suffix, args = self.code_args(codes)
                test = self._m('contains' + suffix, o.ref(hold), *args)
            else:
                test = cst.Comparison(self._m('view_start', o.ref(hold), cst.Integer(str(view.depth + len(parts))),
                                              *self._padded(codes)),
                                      [cst.ComparisonTarget(cst.GreaterThanEqual(), cst.Integer('0'))])
            return cst.UnaryOperation(cst.Not(), test) if isinstance(target.operator, cst.NotIn) else test
        test = membership(own)
        shape = []
        if runtime is not None:     # a tuple of another length: pandas (the native test fails)
            k, n = o.tuple_guards[_code(runtime[0])]
            test = self._lowered(cst.parse_expression(f'type({k}) is tuple'), membership(runtime), test)
            shape = [cst.parse_expression(f'(type({k}) is not tuple or len({k}) == {n})')]
        start, stop, native = self._view_parts(view, hold)
        native = self._conjunction([native, *shape])
        if name is None and isinstance(index.value, cst.Name):
            self.replaced_names.add(index.value.value)
        self._record(original, 'L7', view.call, view=view)
        return self._guarded(original, updated, view, native, test)

    def leave_Comparison(self, original, updated):
        o = self.o
        if len(original.comparisons) != 1:
            return updated
        target = original.comparisons[0]
        if not isinstance(target.operator, (cst.In, cst.NotIn)):
            return updated
        index = target.comparator
        if not (isinstance(index, cst.Attribute) and index.attr.value == 'index'):
            return updated
        if self.raised_veto(original, index):
            return updated
        prefix = self.prefix_membership(original, updated, index, target)
        if prefix is not None:
            return prefix
        found = o._receiver(index.value)
        if found is None:
            return updated
        call, schema = found
        if not self._plain(original, call, schema):
            return updated
        key_parts, why = self._key(original.left, schema)
        if why:
            self._record(original, 'pandas', call, why)
            return updated
        hold = self._take(call, _NO_LABEL, -1)
        if isinstance(index.value, cst.Name):
            self.replaced_names.add(index.value.value)
        (suffix, args), conditions = self._read_args(hold, key_parts, schema['levels'])
        test = self._m('contains' + suffix, o.ref(hold), *args)
        if isinstance(target.operator, cst.NotIn):
            test = cst.UnaryOperation(cst.Not(), test)
        self._record(original, 'L7', call)
        # a part fixed per item that is absent from its level: the membership is False without a lookup; a key that
        # is not the tuple the read expects (T-5c) is pandas' own membership
        guards = o.guards(key_parts)
        conditions = conditions[len(guards):]
        # a part that is not one label (the run-time label test of code_of) is pandas' own membership, never False
        labels = [c for c in conditions if '.is_label(' in _code(c)]
        conditions = [c for c in conditions if '.is_label(' not in _code(c)]
        native = self._native_test(hold, guards + labels)
        if conditions:
            test = self._lowered(self._conjunction(conditions), test,
                                 cst.Name('True' if isinstance(target.operator, cst.NotIn) else 'False'))
        return self._lowered(native, test, self._fallback(updated, call))
