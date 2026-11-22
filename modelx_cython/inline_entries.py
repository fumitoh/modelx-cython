"""Emit private cache entries for resolved, same-Space native Cell calls."""

import libcst as cst

from modelx_cython.cell_entries import DenseAxis, FormulaEntry, cache_hit, exact_receiver
from modelx_cython.consts import CY_MOD, FORMULA_PREF, MX_POW, MX_SELF, MX_SYS_MOD
from modelx_cython.scopes import Names
from modelx_cython.syntax import statements, walk
from modelx_cython.typedefs import primitive_kind


_NUMERIC_TYPES = {"bint", "double", "long long"}
_C_CASTS = {"double": "double", "long long": "longlong", "bint": "bint"}


def allows_exact_receiver(node, owner):
    """Check guard names against the authored formula before emitting guards."""
    return {'type', owner}.isdisjoint(Names(node).names)


def _eligible(cell, params):
    # D139: an uncached Cell (no ``_f_`` method: its public method is the formula) takes the same
    # private entry, the formula itself with no cache lookup, storage or readiness flag
    return (
        (cell.has_formula_def or cell.native_provider_binding is None)
        and not cell.is_locked
        and not cell.formula_is_generator
        and not cell.formula_has_captured_default
        and not cell.body_has_closure
        and not cell.called_with_kwargs
        and (cell.get_rettype_expr(c_style=True) in _NUMERIC_TYPES or cell.native_provider_binding is not None
             or not cell.has_args() and cell.get_rettype_expr(c_style=True) in ('object', 'str'))
        and not params.posonly_params
        and not params.kwonly_params
        and not isinstance(params.star_arg, cst.Param)
        and not params.star_kwarg
        and tuple(p.name.value for p in params.params) == (MX_SELF, *cell.params)
        and params.params[0].default is None
        and all(cell.get_argtype_expr(p, c_style=True) in _NUMERIC_TYPES for p in cell.params)
    )


def _entry_domain(cell):
    """The traced argument range of an uncached Cell with integer arguments, or ``()``.

    The tracer keeps each uncached call that widens an argument's observed range, so this is the
    exact sampled domain. It sizes nothing: the entry tests its arguments against it once, and inside
    it the entry's reads of dense Cells take the same bounds proofs as a hoisted cached formula inside
    its dense window (entry_pipeline._bounds); outside it they take the checked calls."""
    rt = getattr(cell, '_rt', None)
    if rt is None or not (cell.has_args() and cell.has_typeinfo()) or any(
            cell.get_argtype_expr(p, c_style=True) != 'long long' for p in cell.params):
        return ()
    lows, highs = getattr(rt, 'observed_min_args', {}), getattr(rt, 'observed_max_args', {})
    if not all(p in lows and p in highs for p in cell.params) or len(cell.params) > 32:
        return ()
    return tuple(DenseAxis(p, lows[p], highs[p] + 1, 'long long') for p in cell.params)


def _literal_defaults(cell, params):
    defaults = tuple(param for param in params.params[1:] if param.default is not None)
    if defaults and not (cell.has_args() and cell.is_arrayable()):
        return None
    values = []
    for param in defaults:
        if cell.get_argtype_expr(param.name.value, c_style=True) != 'long long':
            return None
        default = param.default
        if isinstance(default, cst.Integer):
            value = default.evaluated_value
        elif (isinstance(default, cst.UnaryOperation) and isinstance(default.operator, cst.Minus)
              and isinstance(default.expression, cst.Integer)):
            value = -default.expression.evaluated_value
        else:
            return None
        if primitive_kind(value) != 'longlong':
            return None
        values.append(default)
    return tuple(values)


def _entry_function(method, entry, owner, optional_receiver, body=None):
    params = method.params
    # The public method always has a receiver. Internal calls prove
    # the exact receiver before entering this private C function.
    receiver = params.params[0].with_changes(annotation=cst.Annotation(
        cst.parse_expression(f'{optional_receiver}[{owner}]')))
    keys = tuple(param.with_changes(default=None, equal=cst.MaybeSentinel.DEFAULT)
                 if param.default is not None else param
                 for param in params.params[1:])
    return method.with_changes(
        name=cst.Name(entry),
        params=params.with_changes(params=(receiver, *keys)),
        body=method.body if body is None else body,
        decorators=tuple(cst.Decorator(cst.parse_expression(f"{CY_MOD}.{kind}"))
                         for kind in ("cfunc", "inline")),
    )


class _Calls(cst.CSTTransformer):
    def __init__(self, owner, entries, exact_guards=(), provider_receivers=None, item_entries=None,
                 proven=False):
        self.entries = entries
        # D114 item 5: {typed item local: (its class, that class's entries)}
        self.item_entries = item_entries or {}
        self.provider_receivers = provider_receivers or {}
        self.unlocked_provider_sources = set()
        self.exact_receiver = cst.parse_expression(exact_receiver(owner))
        self.exact_guards = exact_guards
        # D139: the body of an uncached Cell's private entry runs on a proven exact receiver
        self.exact_depth = 1 if proven else 0

    def has_exact_guard(self, node):
        return isinstance(node.test, cst.Name) and node.test.value in self.exact_guards

    def proves_exact_receiver(self, node):
        test = node.test
        return (self.has_exact_guard(node) or test.deep_equals(self.exact_receiver)
                or (isinstance(test, cst.BooleanOperation) and isinstance(test.operator, cst.And)
                    and test.left.deep_equals(self.exact_receiver)))

    def visit_If_body(self, node):
        if self.proves_exact_receiver(node):
            self.exact_depth += 1

    def leave_If_body(self, node):
        if self.proves_exact_receiver(node):
            self.exact_depth -= 1

    def leave_If(self, original, updated):
        if self.has_exact_guard(original):
            return updated.with_changes(test=cst.BooleanOperation(
                self.exact_receiver, cst.And(), updated.test))
        return updated

    # These scopes may bind a different receiver or evaluate later.
    def visit_FunctionDef(self, node):
        return False

    visit_ClassDef = visit_Lambda = visit_GeneratorExp = visit_FunctionDef
    visit_ListComp = visit_SetComp = visit_DictComp = visit_FunctionDef

    def visit_BinaryOperation(self, node):
        # Leave operands at their established Cython power boundary.
        return not isinstance(node.operator, cst.Power)

    def visit_Call(self, node):
        func = node.func
        return not (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                    and func.value.value == MX_SYS_MOD and func.attr.value == MX_POW)

    def leave_Call(self, original, updated):
        func = updated.func
        if (not updated.args and isinstance(func, cst.Attribute)
                and isinstance(func.value, cst.Attribute) and isinstance(func.value.value, cst.Name)
                and func.value.value.value == MX_SELF):
            source = self.provider_receivers.get((func.value.attr.value, func.attr.value))
            if source is not None and not source.is_locked:
                kind = 'table' if source.native_table_result else 'provider'
                self.unlocked_provider_sources.add((source.fqname, kind))
            elif source is not None:
                receiver = cst.Module([]).code_for_node(func.value)
                return cst.IfExp(
                    test=cst.parse_expression(
                        f'type({receiver}) is {source.parent.name} and '
                        f'{MX_SYS_MOD}._mx_load_flag({CY_MOD}.address({receiver}._has_{func.attr.value}))'),
                    body=cst.Attribute(func.value, cst.Name(f'_v_{func.attr.value}')),
                    orelse=updated, lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))
        if (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                and (item := self.item_entries.get(func.value.value)) is not None
                and func.attr.value in item[1]):
            # a call on a typed item local takes the item class's entry behind its own exact-type test
            return self._entry_call(updated, item[1][func.attr.value],
                                    cst.parse_expression(f'type({func.value.value}) is {item[0]}'), False)
        if not (isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name)
                and func.value.value == MX_SELF and func.attr.value in self.entries):
            return updated
        return self._entry_call(updated, self.entries[func.attr.value], self.exact_receiver, self.exact_depth)

    def _entry_call(self, updated, spec, exact_receiver, proven):
        func = updated.func
        entry, arity, defaults, rettype = spec
        missing = arity - len(updated.args)
        if not 0 <= missing <= len(defaults) or any(a.keyword or a.star for a in updated.args):
            return updated
        supplied = tuple(cst.Arg(value) for value in defaults[len(defaults) - missing:])
        direct = updated.with_changes(
            func=cst.Name(entry), args=(cst.Arg(func.value), *updated.args, *supplied))
        if proven:
            return direct
        # D59: the virtual branch converts to the Cell's C return type, as the typed entry
        # returns it, so the whole expression has that type and a local assigned from it
        # is inferred as a C number (infer_types) rather than an object.
        virtual = updated
        if rettype in _C_CASTS:
            virtual = cst.Call(func=cst.parse_expression(f'{CY_MOD}.cast'), args=(
                cst.Arg(cst.parse_expression(f'{CY_MOD}.{_C_CASTS[rettype]}')),
                cst.Arg(updated.with_changes(lpar=(), rpar=()))))
        return cst.IfExp(
            test=exact_receiver,
            body=direct,
            orelse=virtual,
            lpar=(cst.LeftParen(),),
            rpar=(cst.RightParen(),),
        )

    def return_branches(self, expr):
        """Split arithmetic over admitted Cells into exact and virtual expressions."""
        if isinstance(expr, cst.IfExp) and expr.test.deep_equals(self.exact_receiver):
            return expr.body, expr.orelse, 1
        if isinstance(expr, (cst.Integer, cst.Float)):
            return expr, expr, 0
        if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, (cst.Add, cst.Subtract, cst.Multiply)):
            left, right = self.return_branches(expr.left), self.return_branches(expr.right)
            if left is not None and right is not None:
                return (expr.with_changes(left=left[0], right=right[0]),
                        expr.with_changes(left=left[1], right=right[1]),
                        left[2] + right[2])
        return None

    def leave_SimpleStatementLine(self, original, updated):
        if len(updated.body) != 1 or not isinstance(updated.body[0], cst.Return):
            return updated
        statement = updated.body[0]
        branches = self.return_branches(statement.value)
        if branches is None or branches[2] < 2:
            return updated
        fast, fallback, _ = branches
        guard = cst.If(self.exact_receiver, cst.IndentedBlock(body=[
            updated.with_changes(body=[statement.with_changes(value=fast)], leading_lines=[]),
        ]), leading_lines=updated.leading_lines)
        return cst.FlattenSentinel([
            guard, updated.with_changes(body=[statement.with_changes(value=fallback)], leading_lines=[]),
        ])


def _referenced_entries(body, functions, uncached, classes, origins, domains):
    """``functions`` without the private entry of an uncached Cell that no emitted code calls.

    An uncached Cell's entry (D139) exists for its callers on a proven receiver. With none it is
    an unreachable module-level C function and is not emitted; an entry called only by such an
    entry is unreachable too."""
    candidates = {entry: key for key, entry in uncached.items()}
    kept = list(functions)
    while True:
        names = [(fn, {n.value for n in walk(fn) if isinstance(n, cst.Name)}) for fn in kept]
        used = {n.value for node in body if node is not None for n in walk(node) if isinstance(n, cst.Name)}
        for fn, found in names:
            used |= found - {fn.name.value}
        dead = {fn.name.value for fn in kept if fn.name.value in candidates and fn.name.value not in used}
        if not dead:
            return kept
        kept = [fn for fn in kept if fn.name.value not in dead]
        for entry in dead:
            owner, name = candidates[entry]
            classes[owner].cells[name].inline_entry = None
            origins.pop(entry, None)
            domains.pop(entry, None)


def add_inline_entries(module, classes, dense_layouts):
    """Move each admitted cache body and report its dense layout to the build."""
    names = Names(module)
    used_names = names.names
    optional_receiver = names.reserve('_mx_private_receiver')
    functions = []
    dense_entries = {}
    formula_entries = {}
    next_entry = 0
    body = []
    all_entries, pending = {}, []
    uncached = {}          # D139: {(owner, cell name): entry} of uncached Cells
    origins = {}
    domains = {}           # D139b: {entry of an uncached Cell: its traced argument domain}
    for statement in module.body:
        if not isinstance(statement, cst.ClassDef) or statement.name.value not in classes:
            body.append(statement)
            continue

        owner = statement.name.value
        cells = classes[owner].cells
        entries = {}
        methods = []
        for method in statement.body.body:
            cell = cells.get(method.name.value) if isinstance(method, cst.FunctionDef) else None
            if cell is None or cell.is_special() or not _eligible(cell, method.params):
                methods.append(method)
                continue
            defaults = _literal_defaults(cell, method.params)
            if defaults is None:
                methods.append(method)
                continue

            # Reserve a module-level identifier outside all emitted scopes.
            while (entry := f"_mx_cache_entry_{next_entry}") in used_names:
                next_entry += 1
            next_entry += 1
            cell.inline_entry = entry
            entries[cell.name] = entry, len(cell.params), defaults, cell.get_rettype_expr(c_style=True)
            if not cell.has_formula_def:
                # D139: emitted below from the formula, once every class's entries are known
                uncached[owner, cell.name] = entry
                origins[entry] = owner, cell.name
                if domain := _entry_domain(cell):
                    domains[entry] = domain
                methods.append(method)
                continue
            layout = dense_layouts.get((owner, cell.name))
            if layout is not None:
                dense_entries[entry] = layout
                hit = layout.hit if cell.get_rettype_expr(c_style=True) in _NUMERIC_TYPES else None
            else:
                hit = cache_hit(statements(method.body), cell.name) if not cell.has_args() else None
            formula_entries[entry] = FormulaEntry(
                owner, FORMULA_PREF + cell.name, tuple(cell.params),
                getattr(cell, 'formula_allows_exact_receiver', False), hit,
            )
            functions.append(_entry_function(method, entry, owner, optional_receiver))
            # Public methods retain virtual formula dispatch. Private entries
            # are called only behind the exact-receiver proof emitted below.
            # Both entry paths use the same Cell cache fields.
            methods.append(method)

        all_entries[owner] = entries
        pending.append((statement, owner, cells, entries, methods))
        body.append(None)
    # D114 item 5: formulas are rewritten after every class's entries are known, so a call on a
    # typed item local can take the entry of the item's class
    pending_iter = iter(pending)
    for position, placed in enumerate(body):
        if placed is not None:
            continue
        statement, owner, cells, entries, methods = next(pending_iter)
        authored = list(methods)
        if entries or any(getattr(cell, 'typed_item_locals', None) for cell in cells.values()):
            for index, method in enumerate(methods):
                if not isinstance(method, cst.FunctionDef):
                    continue
                name = method.name.value
                if name.startswith(FORMULA_PREF):
                    cell = cells.get(name[len(FORMULA_PREF):])
                    compiled = (cell is not None and not cell.formula_is_generator
                                and not cell.formula_has_captured_default)
                else:
                    cell = cells.get(name)
                    compiled = (cell is not None and not cell.has_formula_def
                                and not cell.body_has_closure and not cell.called_with_kwargs)
                if not compiled or cell.is_special():
                    continue
                # Use authored names when a formula pass has emitted its own guard.
                guard_allowed = getattr(cell, 'formula_allows_exact_receiver', None)
                if guard_allowed is None:
                    guard_allowed = allows_exact_receiver(method, owner)
                if guard_allowed:
                    names_in_formula = Names(method).names
                    provider_receivers = {
                        (ref_name, name): other
                        for ref_name, ref in classes[owner].refs.items()
                        if (target := getattr(ref, 'decl_type_expr', None)) in classes
                        and target not in names_in_formula
                        for name, other in classes[target].cells.items()
                        if (other.native_provider_binding is not None and not other.has_args()
                            and other.has_formula_def)
                    }
                    item_entries = {
                        local: (target, all_entries[target])
                        for local, target in getattr(cell, 'typed_item_locals', {}).items()
                        # the class name is in the formula only as P1's own annotation of the local
                        if all_entries.get(target)}
                    calls = _Calls(owner, entries, getattr(cell, 'exact_reduction_guards', ()), provider_receivers,
                                   item_entries)
                    methods[index] = method.with_changes(body=method.body.visit(calls))
                    cell.unlocked_provider_sources = sorted(calls.unlocked_provider_sources)
                    if (entry := uncached.get((owner, cell.name))) is not None and not name.startswith(FORMULA_PREF):
                        # D139: the entry is the authored formula with its calls taken on the proven receiver
                        proven = _Calls(owner, entries, getattr(cell, 'exact_reduction_guards', ()),
                                        provider_receivers, item_entries, proven=True)
                        functions.append(_entry_function(
                            authored[index], entry, owner, optional_receiver, authored[index].body.visit(proven)))
                        continue
                if (entry := uncached.get((owner, cell.name))) is not None and not name.startswith(FORMULA_PREF):
                    functions.append(_entry_function(authored[index], entry, owner, optional_receiver))
        body[position] = statement.with_changes(body=statement.body.with_changes(body=methods))
    functions = _referenced_entries(body, functions, uncached, classes, origins, domains)
    if functions:
        index = 0
        for index, statement in enumerate(body):
            if not (isinstance(statement, cst.SimpleStatementLine)
                    and len(statement.body) == 1):
                break
            item = statement.body[0]
            if (index == 0 and isinstance(item, cst.Expr)
                    and isinstance(item.value, (cst.SimpleString, cst.ConcatenatedString))):
                continue
            if (isinstance(item, cst.ImportFrom) and isinstance(item.module, cst.Name)
                    and item.module.value == '__future__'):
                continue
            break
        else:
            index = len(body)
        body.insert(index, cst.parse_statement(
            f'from typing import Optional as {optional_receiver}\n'))
    return module.with_changes(body=(*body, *functions)), dense_entries, formula_entries, origins, domains
