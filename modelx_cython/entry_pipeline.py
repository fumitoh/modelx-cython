"""Analyze and emit private Cell entries from the recorded cache layout."""

from collections import Counter
from dataclasses import dataclass, replace

import libcst as cst
from libcst.metadata import ExpressionContext

from .cell_entries import checked_dense_guard, dense_condition, exact_receiver
from .consts import CY_MOD, FORMULA_PREF, MX_SELF
from .scopes import CstFacts, OPAQUE_SCOPES
from .syntax import arguments, dotted_name, literal_value, same_expression, walk
from .typedefs import primitive_kind


def _attribute_counts(tree):
    return Counter(node.attr.value for node in walk(tree)
                   if isinstance(node, cst.Attribute) and node.attr.value.startswith(FORMULA_PREF))


def _argument_range(node, ranges, types, target_type):
    if isinstance(node, cst.Name):
        source_type = types.get(node.value)
        if source_type == target_type or (source_type == 'bint' and target_type == 'long long'):
            return ranges.get(node.value)
    if (target_type == 'long long' and isinstance(node, cst.BinaryOperation)
            and isinstance(node.left, cst.Name) and node.left.value in ranges
            and types[node.left.value] == 'long long' and isinstance(node.right, cst.Integer)
            and isinstance(node.operator, (cst.Add, cst.Subtract))):
        lower, upper = ranges[node.left.value]
        delta = node.right.evaluated_value
        if isinstance(node.operator, cst.Subtract):
            delta = -delta
        if -(1 << 63) < delta < (1 << 63) and -(1 << 63) <= lower + delta < upper + delta <= (1 << 63):
            return lower + delta, upper + delta
    return None


def _exact_calls(node, receiver, exact=False):
    if isinstance(node, (cst.If, cst.IfExp)) and same_expression(node.test, receiver):
        for branch, active in ((node.body, True), (node.orelse, exact)):
            if branch is not None:
                yield from _exact_calls(branch, receiver, active)
        return
    if exact and isinstance(node, cst.Call) and isinstance(node.func, cst.Name):
        yield node
    if not isinstance(node, OPAQUE_SCOPES):
        for child in node.children:
            yield from _exact_calls(child, receiver, exact)


def _public_entry(layout, methods):
    return methods.get((layout.owner, layout.formula[len(FORMULA_PREF):]))


def _formula_ranges(state, layouts, guards, uses):
    spills = {layout.spill_entry for layout in layouts.values() if layout.spill_entry}

    def spill_references(tree):
        return Counter(node.value for node in walk(tree) if isinstance(node, cst.Name)
                       and node.value in spills and state.facts.contexts.get(node) is ExpressionContext.LOAD)

    spill_uses = spill_references(state.syntax) if spills else Counter()
    ranges = {}
    for name, guard in guards.items():
        layout = layouts[name]
        if not 0 < len(layout.axes) <= 32:
            continue
        public = _public_entry(layout, state.methods)
        shared_guard = public is not None and same_expression(guard, checked_dense_guard(public.body, layout))
        uses_here = uses[layout.formula] - (_attribute_counts(public)[layout.formula] if shared_guard else 0)
        if layout.checks_spill_in_helper or (not layout.exact_spill and uses_here == 1):
            ranges[name] = layout.axes
        elif layout.exact_spill and layout.spill_entry and uses_here == 2:
            calls = spill_uses[layout.spill_entry]
            if shared_guard:
                calls -= spill_references(public)[layout.spill_entry]
            if calls == 1:
                ranges[name] = layout.logical_axes
    return ranges


@dataclass(frozen=True)
class _Domain:
    name: str
    axes: tuple


def _bounds(state, layouts, guards, uses):
    guards = {name: guard for name, guard in guards.items() if 0 < len(layouts[name].axes) <= 32}
    ranges = _formula_ranges(state, layouts, guards, uses)
    # Callers whose arguments lie in known axes: a cached Cell's formula, reached inside its dense
    # window (or behind its own domain test when it checks its spill in the helper), and (D139b) an
    # uncached Cell's private entry, which tests its arguments against its traced domain once and runs
    # on a proven receiver. Every caller takes the same proofs below.
    callers = []
    for name, source_axes in ranges.items():
        layout = layouts[name]
        callers.append((state.methods.get((layout.owner, layout.formula)), layout.owner, source_axes,
                        layout.checks_spill_in_helper, False))
    for entry, source_axes in state.entry_domains.items():
        if (origin := state.function_origins.get(entry)) is not None:
            callers.append((state.functions.get(entry), origin[0], source_axes, True, True))
    masks, domains = {}, {}
    for formula, owner, source_axes, checked_domain, proven in callers:
        if not source_axes or formula is None or not _native_formula(formula):
            continue
        nodes = list(walk(formula.body))
        if any(isinstance(node, getattr(cst, 'Match', ())) for node in nodes):
            continue
        mutable = CstFacts(formula.body.body).bindings(nodes)
        intervals = {axis.key: (axis.lower, axis.stop) for axis in source_axes if axis.key not in mutable}
        types = {axis.key: axis.c_type for axis in source_axes}
        receiver = cst.parse_expression(exact_receiver(owner))
        for call in _exact_calls(formula.body, receiver, proven):
            target = call.func.value
            if target not in guards or layouts[target].owner != owner:
                continue
            target_axes = ranges.get(target) or layouts[target].axes
            args = arguments(call, count=len(target_axes) + 1)
            if args is None or not isinstance(args[0], cst.Name) or args[0].value != MX_SELF:
                continue
            mask = 0
            for index, (arg, axis) in enumerate(zip(args[1:], target_axes)):
                interval = _argument_range(arg, intervals, types, axis.c_type)
                if interval and axis.lower <= interval[0] < interval[1] <= axis.stop:
                    mask |= 1 << index
            if mask:
                if checked_domain and formula not in domains:
                    domains[formula] = _Domain(state.names.reserve('_mx_dense_domain', family=True), source_axes)
                masks[call] = mask, domains.get(formula)
    return ranges, masks, domains


def _native_formula(method):
    return any(dotted_name(item.decorator) == f'{CY_MOD}.cfunc' for item in method.decorators)


def _eligible(method, keys, guard_safe):
    if not guard_safe or not _native_formula(method):
        return False
    params = method.params
    if (params.posonly_params or params.kwonly_params or params.star_kwarg
            or isinstance(params.star_arg, cst.Param)
            or tuple(param.name.value for param in params.params) != (MX_SELF, *keys)):
        return False
    native = {f'{CY_MOD}.{name}' for name in ('double', 'longlong', 'bint')}
    if (method.returns is None or dotted_name(method.returns.annotation) not in native
            or any(param.annotation is None or dotted_name(param.annotation.annotation) not in native
                   for param in params.params[1:])):
        return False
    nodes = list(walk(method.body))
    facts = CstFacts(method.body.body)
    if MX_SELF in facts.bindings(nodes):
        return False
    for node in nodes:
        if isinstance(node, (*OPAQUE_SCOPES, cst.Yield)):
            return False
        if isinstance(node, cst.Attribute) or isinstance(node, cst.Name) and node in facts.contexts:
            name = node.value if isinstance(node, cst.Name) else node.attr.value
            if name.startswith("__") or name in {"super", "eval", "exec", "locals", "vars"}:
                return False
    return True


def _unsigned_check(layout):
    if (len(layout.axes) != 1 or (layout.exact_spill and not layout.checks_spill_in_helper)
            or layout.axes[0].c_type != "long long"):
        return None
    axis, = layout.axes
    span = axis.stop - axis.lower
    if 0 < span < 1 << 64:
        # D15: subtract signed, then cast. The difference of two unsigned casts trips
        # Cython's overflow check whenever key < lower (always for a negative lower bound).
        offset = f"{axis.key} - ({axis.lower})" if axis.lower else axis.key
        return cst.parse_expression(f"{CY_MOD}.cast({CY_MOD}.ulonglong, {offset}) < {span}")
    return None


def _guard(method, layout):
    if method is None or tuple(p.name.value for p in method.params.params) != (
            MX_SELF, *(axis.key for axis in layout.axes)):
        return None
    return checked_dense_guard(method.body, layout)


def _with_guard(method, guard):
    return method.with_changes(body=method.body.with_changes(body=(guard,)))


def _param(name, kind):
    return cst.Param(cst.Name(name), annotation=cst.Annotation(cst.parse_expression(f'{CY_MOD}.{kind}')))


def _with_domain(suite, domain):
    declaration = cst.parse_statement(f'{domain.name}: {CY_MOD}.bint = {dense_condition(domain.axes)}\n')
    body = suite.body
    doc = bool(body and isinstance(body[0], cst.SimpleStatementLine) and len(body[0].body) == 1
               and isinstance(body[0].body[0], cst.Expr)
               and isinstance(body[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString)))
    return suite.with_changes(body=(*body[:doc], declaration, *body[doc:]))


def _hit_layout(params, hit):
    test = hit.ready if hit.guard is None else cst.BooleanOperation(hit.guard, cst.And(), hit.ready)
    return tuple((arg.name.value, arg.annotation.annotation) for arg in params), test, hit.value


class _Substitute(cst.CSTTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def leave_Name(self, original, updated):
        return self.mapping.get(original.value, updated)

    def leave_Attribute(self, original, updated):
        return updated.with_changes(attr=original.attr)


def _expose_hits(body, parameters, layouts):
    if not layouts or not parameters or parameters[0].annotation is None:
        return None
    nodes = list(walk(body))
    if (CY_MOD in CstFacts(body.body).bindings(nodes)
            or any(arg.name.value == CY_MOD for arg in parameters)
            or any(isinstance(node, getattr(cst, 'Match', ())) for node in nodes)):
        return None
    receiver = parameters[0]
    types = {arg.name.value: arg.annotation.annotation for arg in parameters if arg.annotation is not None}

    def expose(call):
        layout = layouts.get(call.func.value) if isinstance(call.func, cst.Name) else None
        if layout is None:
            return call
        params, test, value = layout
        values = arguments(call, count=len(params))
        if (values is None or not same_expression(params[0][1], receiver.annotation.annotation)
                or not isinstance(values[0], cst.Name) or values[0].value != receiver.name.value):
            return call
        mapping = {params[0][0]: values[0]}
        for (name, annotation), arg in zip(params[1:], values[1:]):
            if isinstance(arg, cst.Name) and arg.value not in ('True', 'False', 'None'):
                kind = types.get(arg.value)
                if kind is None or not same_expression(kind, annotation):
                    return call
            else:
                try:
                    token = literal_value(arg)
                except (ValueError, TypeError):
                    return call
                if (dotted_name(annotation), primitive_kind(token)) not in (
                        (f'{CY_MOD}.bint', 'bint'), (f'{CY_MOD}.longlong', 'longlong')):
                    return call
            mapping[name] = arg
        substitute = _Substitute(mapping)
        return cst.IfExp(test.visit(substitute), value.visit(substitute), call,
                         lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))

    return expose


class _EmitCalls(cst.CSTTransformer):
    def __init__(self, masks, targets, inbounds, propagate, *, miss=None,
                 guard=None, exact=None, domain=None, methods=None, domains=None, expose=None):
        self.masks, self.targets, self.inbounds = masks, targets, inbounds
        self.propagate = propagate
        self.miss = miss
        self.guard, self.exact, self.domain = guard, exact, domain
        self.methods = methods or {}
        self.domains = domains or {}
        self.expose = expose

    def visit_FunctionDef(self, node):
        return node not in self.methods

    def leave_FunctionDef(self, before, after):
        if before in self.methods:
            return self.methods[before]
        if (domain := self.domains.get(before)) is not None:
            return after.with_changes(body=_with_domain(after.body, domain))
        return after

    def leave_Comparison(self, before, after):
        if self.guard is not None and before.deep_equals(self.guard):
            return self.exact.with_changes(lpar=(*after.lpar, *self.exact.lpar),
                                           rpar=(*self.exact.rpar, *after.rpar))
        return after

    def leave_Call(self, before, after):
        emitted = self._entry_call(before, after)
        if self.expose is None:
            return emitted
        if isinstance(emitted, cst.IfExp):
            return emitted.with_changes(body=self.expose(emitted.body), orelse=self.expose(emitted.orelse))
        return self.expose(emitted)

    def _entry_call(self, before, after):
        if self.miss is not None and isinstance(before.func, cst.Attribute):
            formula, keys, direct = self.miss
            args = arguments(before, f'{MX_SELF}.{formula}', len(keys))
            if args is not None and all(isinstance(arg, cst.Name) and arg.value == key
                                        for arg, key in zip(args, keys)):
                params = (cst.Arg(cst.Name(MX_SELF)), *after.args)
                if self.propagate:
                    params += (cst.Arg(cst.Name('True')),)
                return after.with_changes(func=cst.Name(direct), args=params)
        if not isinstance(before.func, cst.Name) or before.func.value not in self.targets:
            return after
        mask, domain = self.masks.get(before, (0, None))
        conditional = cst.Name(domain.name) if domain is not None and domain is not self.domain else None
        if before.func.value not in self.inbounds:
            value = cst.Integer(str(mask))
            if conditional is not None:
                value = cst.IfExp(conditional, value, cst.Integer('0'))
            return after.with_changes(args=(*after.args, cst.Arg(value)))
        args = (*after.args[:-1], after.args[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT))
        checked = after.with_changes(args=args)
        if not mask:
            return checked
        if self.propagate:
            args += (cst.Arg(cst.Name('True')),)
        direct = after.with_changes(func=cst.Name(self.inbounds[before.func.value]), args=args)
        return (cst.IfExp(conditional, direct, checked, lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))
                if conditional is not None else direct)


def _emit(state, layouts, entries, uses, policy):
    if not (layouts or entries):
        return state
    guards = {name: guard for name, layout in layouts.items()
              if (guard := _guard(state.functions.get(name), layout)) is not None}
    ranges, masks, domains = _bounds(state, layouts, guards, uses) if policy.dense_bounds_proofs else ({}, {}, {})
    targets = {call.func.value for call in masks}
    proof = state.names.reserve('_mx_proven_axes', family=True) if targets else None
    exact = state.names.reserve('_mx_exact_receiver')
    inbounds_exact = state.names.reserve('_mx_inbounds_exact')
    checks = {}
    if policy.dense_one_axis:
        checks = {name: check for name, layout in layouts.items() if (check := _unsigned_check(layout)) is not None}
    inbounds = {name: state.names.reserve(name + '_inbounds') for name, fn in state.functions.items()
                if name in checks and name in targets and name in guards
                and isinstance(guards[name].orelse, cst.Else)}
    propagate = policy.propagate_exact_receiver
    methods, misses, hoisted, proven = {}, {}, [], []
    origins = state.function_origins.copy()
    hit_layouts = {}
    mode = policy.native_cache_hits
    if mode != 'none':
        for name, entry in entries.items():
            hit = entry.hit
            if hit is None or (name in targets and name not in inbounds):
                continue
            params = state.functions[name].params.params
            if name in inbounds:
                hit = replace(hit, guard=None)
                if propagate:
                    params = (*params, _param(inbounds_exact, 'bint'))
            elif name in checks and name in guards and isinstance(guards[name].orelse, cst.Else):
                hit = replace(hit, guard=checks[name])
            if mode in (hit.kind, 'both') or mode == 'auto' and hit.guard is None:
                hit_layouts[inbounds.get(name, name)] = _hit_layout(params, hit)

    def calls(**options):
        return _EmitCalls(masks, targets, inbounds, propagate, **options)

    if policy.dense_miss_binding:
        by_method = {(layout.owner, layout.formula): name for name, layout in entries.items()
                     if name in state.functions}
        for identity, fn in state.methods.items():
            entry = by_method.get(identity)
            if entry is None:
                continue
            layout = entries[entry]
            if not _eligible(fn, layout.keys, layout.guard_safe):
                continue
            direct = state.names.reserve(f'_mx_dense_formula_{len(hoisted)}')
            origins[direct] = identity
            receiver, *params = fn.params.params
            receiver = receiver.with_changes(annotation=state.functions[entry].params.params[0].annotation)
            args = ', '.join((MX_SELF, *layout.keys))
            domain = domains.get(fn)
            options = {}
            if propagate:
                params = [param.with_changes(default=None, equal=cst.MaybeSentinel.DEFAULT) for param in params]
                params.append(_param(exact, 'bint'))
                receiver_test = exact_receiver(layout.owner)
                args += (f', ({receiver_test} and ({dense_condition(domain.axes)}))'
                         if domain is not None else f', {receiver_test}')
                options = {'guard': cst.parse_expression(receiver_test), 'exact': cst.Name(exact), 'domain': domain}
            signature = (receiver, *params)
            formula_body = fn.body.visit(calls(expose=_expose_hits(fn.body, signature, hit_layouts), **options))
            if domain is not None and not propagate:
                formula_body = _with_domain(formula_body, domain)
            hoisted.append(fn.with_changes(
                name=cst.Name(direct), params=fn.params.with_changes(params=signature),
                body=formula_body,
                decorators=(*fn.decorators, cst.Decorator(cst.parse_expression(f'{CY_MOD}.inline')))))
            methods[fn] = fn.with_changes(body=fn.body.with_changes(body=(cst.parse_statement(f'return {direct}({args})\n'),)))
            misses[entry] = layout.formula, layout.keys, direct

    if hit_layouts:
        # D139: the private entry of an uncached Cell is a formula on a proven receiver; its reads of
        # cached Cells expose their hits as the hoisted formulas' reads do
        for name, fn in state.functions.items():
            origin = state.function_origins.get(name)
            if (origin is None or origin[1].startswith(FORMULA_PREF) or name in entries or name in guards
                    or fn in methods):
                continue
            expose = _expose_hits(fn.body, fn.params.params, hit_layouts)
            if expose is not None:
                body = fn.body.visit(calls(expose=expose))
                if (domain := domains.get(fn)) is not None:
                    # D139b: the entry's own domain test, as the module pass adds to other formulas
                    body = _with_domain(body, domain)
                methods[fn] = fn.with_changes(body=body)

    for name, check in checks.items():
        layout = layouts[name]
        method = _public_entry(layout, state.methods)
        if (guard := _guard(method, layout)) is not None and isinstance(guard.orelse, cst.Else):
            params = method.params
            if (not params.posonly_params and not params.kwonly_params and not params.star_kwarg
                    and not isinstance(params.star_arg, cst.Param)
                    and params.params[1].annotation is not None
                    and dotted_name(params.params[1].annotation.annotation) == f'{CY_MOD}.longlong'):
                methods[method] = _with_guard(method, guard.with_changes(test=check))

    for name, fn in state.functions.items():
        if name not in entries and name not in guards:
            continue
        miss = misses.get(name)
        guard = guards.get(name)
        if guard is None:
            if miss is not None:
                methods[fn] = fn.with_changes(body=fn.body.visit(calls(miss=miss)))
            continue
        layout = layouts[name]
        check = checks.get(name) if isinstance(guard.orelse, cst.Else) else None
        if name in inbounds:
            inbounds_name = inbounds[name]
            signature = fn.params.with_changes(params=tuple(
                p.with_changes(comma=cst.MaybeSentinel.DEFAULT) for p in fn.params.params))
            params = signature
            options = {'miss': miss}
            flag = ''
            if propagate:
                params = signature.with_changes(params=(*signature.params, _param(inbounds_exact, 'bint')))
                receiver_test = exact_receiver(layout.owner)
                options.update(guard=cst.parse_expression(receiver_test),
                               exact=cst.parse_expression(f'({inbounds_exact} or {receiver_test})'))
                flag = ', False'
            proven.append(fn.with_changes(
                name=cst.Name(inbounds_name), params=params, body=guard.body.visit(calls(**options))))
            forward = cst.parse_statement(f'return {inbounds_name}({MX_SELF}, {layout.axes[0].key}{flag})\n')
            methods[fn] = _with_guard(fn.with_changes(params=signature),
                                     guard.with_changes(test=check, body=guard.body.with_changes(body=(forward,))))
            continue

        if miss is not None or targets:
            guard = guard.visit(calls(miss=miss))
        if name in targets:
            proved = ranges.get(name, ())
            fallback = guard.orelse
            if proved and layout.logical_axes and not layout.checks_spill_in_helper:
                logical = cst.parse_expression(dense_condition(proved, proof))
                if isinstance(fallback, cst.If):
                    fallback = fallback.with_changes(test=logical)
                else:
                    nested = fallback.body.body[0].with_changes(test=logical)
                    fallback = fallback.with_changes(body=fallback.body.with_changes(body=(nested,)))
            guard = guard.with_changes(test=cst.parse_expression(dense_condition(layout.axes, proof, proved)),
                                       orelse=fallback)
            params = fn.params.with_changes(params=(*fn.params.params, _param(proof, 'uint')))
            methods[fn] = _with_guard(fn.with_changes(params=params), guard)
        elif check is not None:
            methods[fn] = _with_guard(fn, guard.with_changes(test=check))
        elif miss is not None:
            methods[fn] = _with_guard(fn, guard)

    syntax = state.syntax.visit(calls(methods=methods, domains=domains)) if methods or masks else state.syntax
    return replace(state, syntax=syntax.with_changes(body=(*syntax.body, *hoisted, *proven)),
                   function_origins=origins, inbounds_entries={target: source for source, target in inbounds.items()})


def lower_entries(modules, dense, formulas, policy):
    uses = Counter()
    if policy.dense_bounds_proofs and dense:
        for state in modules.values():
            uses.update(_attribute_counts(state.syntax))
    return {name: _emit(state, dense.get(name, {}), formulas.get(name, {}), uses, policy)
            for name, state in modules.items()}
