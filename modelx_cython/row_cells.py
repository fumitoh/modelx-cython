"""Row fields read without building the row (diag-run8 F3).

A *row cell* is a cached zero-argument Cell whose whole formula is ``return T.loc[key]``: ``T`` a
self-rooted zero-argument call (a table Cell), ``key`` an expression whose only free name is
``self``. Its value is the row the helper builds (a ``Record``): every reader that reads a field
``self.R()["f"]`` takes and releases references to the shared column names and values, once per item.

When no use of the row cell's value can escape, each read of a field is the same read of the
table: ``self.R()["f"]`` is ``T.loc[key]["f"]``. The readers then get the cell's row expression in
place of the call, and the typed table reads lower it to a native read of the row (``H.row_n(key)``,
``H.f64/i64/obj(row, col)``), so no row object is built for them. The decision is made over the
whole module, for each row cell:

- every use of ``R`` in a formula of its own class is ``self.R()["literal"]``, or a local assigned
  once, at the top level of the formula, as ``v = self.R()`` and loaded only as ``v["literal"]``;
- no other class of the module names ``R`` (a reference from another Space, or a Cell of that name
  in a derived class that could override it), and no nested function or lambda uses it.

Anything else (the value returned, stored, passed, iterated, compared, an attribute or a variable
field of it, a use in a nested scope or another class) is an escape, and the row cell and all its
readers keep the current ``Record`` path. The row cell itself is never changed: called from
Python it computes and caches the same ``Record`` as before. A key that is not in the table is
raised by the reader's own row read, with the exception type and message the row cell's read gives.
A reader whose inlined read the table lowering does not turn native is left as written.

The transformer accepts a row cell only when every key part is fixed per item (``item_constant``):
the readers then hold the row position per item (``held_row``, reset with the item's caches), so
the key is looked up once per item, an integer or a str label alike.
"""

import libcst as cst

DOC_TYPES = (cst.SimpleString, cst.ConcatenatedString)
NESTED = (cst.FunctionDef, cst.Lambda, cst.ClassDef)
FORMULA = '_f_'


def _body(func):
    """The statements of a function without its docstring."""
    body = list(func.body.body) if isinstance(func.body, cst.IndentedBlock) else [func.body]
    if (body and isinstance(body[0], cst.SimpleStatementLine) and len(body[0].body) == 1
            and isinstance(body[0].body[0], cst.Expr) and isinstance(body[0].body[0].value, DOC_TYPES)):
        body = body[1:]
    return body


def _self_chain_call(node):
    """``self.a.b()`` with no arguments."""
    if not (isinstance(node, cst.Call) and not node.args):
        return False
    value = node.func
    if not isinstance(value, cst.Attribute):
        return False
    while isinstance(value, cst.Attribute):
        value = value.value
    return isinstance(value, cst.Name) and value.value == 'self'


class _Names(cst.CSTVisitor):
    """Free names of an expression (attribute names excluded); nested scopes and calls with
    keywords are recorded as not plain."""

    def __init__(self):
        self.names, self.plain = set(), True

    def visit_Attribute(self, node):
        node.value.visit(self)
        return False

    def visit_Name(self, node):
        self.names.add(node.value)

    def visit_Lambda(self, node):
        self.plain = False
        return False

    def visit_GeneratorExp(self, node):
        self.plain = False
        return False

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp

    def visit_NamedExpr(self, node):
        self.plain = False
        return False


def _row_expression(func):
    """The ``T.loc[key]`` of a formula ``return T.loc[key]``, else None."""
    params = func.params
    if (len(params.params) != 1 or params.params[0].name.value != 'self' or params.star_arg
            not in (None, cst.MaybeSentinel.DEFAULT) or params.kwonly_params or params.star_kwarg
            or params.posonly_params):
        return None
    body = _body(func)
    if len(body) != 1 or not isinstance(body[0], cst.SimpleStatementLine) or len(body[0].body) != 1:
        return None
    ret = body[0].body[0]
    if not isinstance(ret, cst.Return) or ret.value is None:
        return None
    expr = ret.value
    if not (isinstance(expr, cst.Subscript) and len(expr.slice) == 1
            and isinstance(expr.slice[0].slice, cst.Index) and expr.slice[0].slice.star is None
            and isinstance(expr.value, cst.Attribute) and expr.value.attr.value == 'loc'
            and _self_chain_call(expr.value.value)):
        return None
    names = _Names()
    expr.slice[0].slice.value.visit(names)
    if not names.plain or names.names - {'self'}:
        return None
    return expr.with_changes(lpar=(), rpar=())


def item_constant(node, cached_call):
    """A key part whose value is fixed for an item until its caches are cleared: an integer
    literal, ``self.<name>`` (a Space parameter or Reference), or ``self.<cell>()`` of a cached
    zero-argument Cell (``cached_call(node)``). Such a key's row is held per item (``held_row``)."""
    if isinstance(node, cst.Integer):
        return True
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name) and node.value.value == 'self':
        return True
    return (isinstance(node, cst.Call) and not node.args and isinstance(node.func, cst.Attribute)
            and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'
            and cached_call(node))


def key_parts(expr):
    """The key parts of a row expression ``T.loc[key]``."""
    key = expr.slice[0].slice.value
    return [e.value for e in key.elements] if isinstance(key, cst.Tuple) else [key]


def _literal_field(subscript):
    return (len(subscript.slice) == 1 and isinstance(subscript.slice[0].slice, cst.Index)
            and subscript.slice[0].slice.star is None
            and isinstance(subscript.slice[0].slice.value, cst.SimpleString))


class _Uses(cst.CSTVisitor):
    """Uses of ``self.<name>`` in one function: accepted field reads (direct, or through a local
    assigned once at the top level) and escapes."""

    def __init__(self, name, top):
        self.name, self.top = name, top
        self.escape = None
        self.locals = {}            # local -> [assignments]
        self.local_loads = {}       # local -> [ok?]
        self._parents = []
        self._depth = 0

    def on_visit(self, node):
        if isinstance(node, NESTED):
            self._depth += 1
        self._parents.append(node)
        return super().on_visit(node)

    def on_leave(self, node):
        super().on_leave(node)
        self._parents.pop()
        if isinstance(node, NESTED):
            self._depth -= 1

    def _nested_scope(self):
        return self._depth > 1      # the function itself is depth 1

    def visit_Attribute(self, node):
        if node.attr.value != self.name:
            return
        parents = self._parents[:-1]
        if not (isinstance(node.value, cst.Name) and node.value.value == 'self'):
            self.escape = self.escape or 'used other than through self'
            return
        if self._nested_scope():
            self.escape = self.escape or 'used in a nested function or lambda'
            return
        call = parents[-1] if parents else None
        if not (isinstance(call, cst.Call) and call.func is node and not call.args):
            self.escape = self.escape or 'not called with no arguments'
            return
        outer = parents[-2] if len(parents) > 1 else None
        if isinstance(outer, cst.Subscript) and outer.value is call:
            if _literal_field(outer) and not self._stored(parents[:-2], outer):
                return
            self.escape = self.escape or 'a field read with a non-literal key, or a store'
            return
        if (isinstance(outer, cst.Assign) and outer.value is call and len(outer.targets) == 1
                and isinstance(outer.targets[0].target, cst.Name) and len(parents) > 2
                and isinstance(parents[-3], cst.SimpleStatementLine) and parents[-3] in self.top):
            self.locals.setdefault(outer.targets[0].target.value, []).append(outer)
            return
        self.escape = self.escape or 'the value is used (returned, passed, stored, compared or iterated)'

    @staticmethod
    def _stored(parents, subscript):
        for parent in reversed(parents):
            if isinstance(parent, (cst.AssignTarget, cst.Del, cst.AugAssign)):
                return True
            if isinstance(parent, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
                return False
        return False


class _LocalLoads(cst.CSTVisitor):
    """Every occurrence of the names ``locals`` in a function: each must be its one recorded
    assignment target or a load ``v["literal"]`` in the function's own scope."""

    def __init__(self, names, assignments):
        self.names, self.assignments = set(names), assignments
        self.bad = set()
        self._parents, self._depth = [], 0

    def on_visit(self, node):
        if isinstance(node, NESTED):
            self._depth += 1
        self._parents.append(node)
        return super().on_visit(node)

    def on_leave(self, node):
        super().on_leave(node)
        self._parents.pop()
        if isinstance(node, NESTED):
            self._depth -= 1

    def visit_Attribute(self, node):
        node.value.visit(self)
        return False

    def visit_Name(self, node):
        if node.value not in self.names:
            return
        parents = self._parents[:-1]
        parent = parents[-1] if parents else None
        if isinstance(parent, cst.AssignTarget) and any(parent is a.targets[0] for a in self.assignments.get(node.value, ())):
            return
        if isinstance(parent, cst.Param) and parent.name is node:
            self.bad.add(node.value)
            return
        if self._depth > 1:
            self.bad.add(node.value)
            return
        if (isinstance(parent, cst.Subscript) and parent.value is node and _literal_field(parent)
                and not _Uses._stored(parents[:-1], parent)):
            return
        self.bad.add(node.value)


class _Mentions(cst.CSTVisitor):
    def __init__(self, name):
        self.name, self.found = name, False

    def visit_Attribute(self, node):
        if node.attr.value == self.name:
            self.found = True

    def visit_Name(self, node):
        if node.value == self.name:
            self.found = True


def _mentions(node, name):
    finder = _Mentions(name)
    node.visit(finder)
    return finder.found


def analyse(module_node, cls_name):
    """{row cell name: its row expression} for the class ``cls_name`` of the module: the row cells
    whose every use is a field read (see the module docstring), and {name: why} for the others."""
    classes = [n for n in module_node.body if isinstance(n, cst.ClassDef)]
    own = next((c for c in classes if c.name.value == cls_name), None)
    if own is None:
        return {}, {}
    methods = [s for s in own.body.body if isinstance(s, cst.FunctionDef)]
    rows = {}
    for func in methods:
        if func.name.value.startswith(FORMULA):
            expr = _row_expression(func)
            if expr is not None:
                rows[func.name.value[len(FORMULA):]] = expr
    accepted, refused = {}, {}
    for name, expr in rows.items():
        why = None
        for cls in classes:
            if cls is own:
                continue
            if any(isinstance(s, cst.FunctionDef) and s.name.value in (FORMULA + name, name)
                   for s in cls.body.body):
                why = f'another class defines {name}'
                break
            if _mentions(cls, name):
                why = f'used in another class ({cls.name.value})'
                break
        for func in methods:
            if why:
                break
            if func.name.value == FORMULA + name:
                continue
            uses = _Uses(name, tuple(_body(func)))
            func.visit(uses)
            if uses.escape:
                why = uses.escape
                break
            if uses.locals:
                if any(len(v) != 1 for v in uses.locals.values()):
                    why = 'a local holding the row is assigned more than once'
                    break
                loads = _LocalLoads(uses.locals, uses.locals)
                func.visit(loads)
                if loads.bad:
                    why = f'a local holding the row ({sorted(loads.bad)[0]}) is used other than by a literal field'
                    break
        if why:
            refused[name] = why
        else:
            accepted[name] = expr
    return accepted, refused


class Inline(cst.CSTTransformer):
    """``self.R()`` of an accepted row cell -> its row expression, in one formula."""

    def __init__(self, rows):
        super().__init__()
        self.rows, self.count = rows, 0

    def leave_Call(self, original, updated):
        func = updated.func
        if (not updated.args and isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                and func.value.value == 'self' and func.attr.value in self.rows):
            self.count += 1
            return self.rows[func.attr.value].deep_clone().with_changes(lpar=updated.lpar, rpar=updated.rpar)
        return updated
