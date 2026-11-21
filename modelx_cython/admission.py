"""Describe actual Cython representations without claiming nogil execution."""

import re

from .consts import FORMULA_PREF, MODULE_PREF, SPACE_PREF


def _lock_reason(source, kind, resource='table'):
    _, *parents, _, cls, cell = source.split('.')
    space = '.'.join([*(name[len(MODULE_PREF):] for name in parents),
                      cls[len(SPACE_PREF):]])
    name = cell[len(FORMULA_PREF):]
    return f'native {kind} {resource} read from {space}.{name} requires {space} in export locked_spaces'


def describe(module_infos):
    from modelx_cython.numeric_cache import primary_capacity
    result = {}
    for module in module_infos.values():
        for cls in module.classes.values():
            for cell in cls.cells.values():
                if cell.is_special():
                    continue
                formula_python = cell.formula_is_generator or cell.formula_has_captured_default or (
                    not cell.has_formula_def and cell.body_has_closure)
                public_python = cell.body_has_closure or cell.called_with_kwargs
                if not cell.has_formula_def:
                    cache = 'none'
                elif not cell.has_args():
                    cache = 'scalar'
                elif cell.has_typeinfo() and cell.is_arrayable():
                    cache = 'dense'
                elif cell.uses_native_numeric_cache:
                    cache = 'native-numeric-keys'
                elif cell.uses_vector_cache:
                    cache = 'vector-dense'
                else:
                    cache = 'dictionary'
                reasons = []
                if not cell.has_typeinfo():
                    reasons.append('formula not observed by the sample')
                if cell.formula_is_generator:
                    reasons.append('yield requires a Python method')
                if cell.formula_has_captured_default:
                    reasons.append('captured default uses the compiled Python signature')
                if cell.body_has_closure:
                    reasons.append('closure requires a Python public method')
                if cell.called_with_kwargs:
                    reasons.append('keyword call retains the Python public boundary')
                if cell.is_locked:
                    reasons.append('exported shared-Space lock retained')
                if cell.observed_native_binding is not None and cell.native_provider_binding is None:
                    reasons.append('native provider result requires a matching return annotation or direct forwarding from a typed Cell')
                reasons.extend(_lock_reason(source, 'cached', resource)
                               for source, resource in getattr(cell, 'unlocked_provider_sources', ()))
                result[cell.fqname] = {
                    'formula': 'python-method' if formula_python else 'cython-cfunction',
                    'public_entry': 'python-method' if public_python else 'cython-cpdef',
                    'inline_entry': getattr(cell, 'inline_entry', None),
                    'return_type': cell.get_rettype_expr(c_style=True),
                    'arguments': {name: cell.get_argtype_expr(name, c_style=True)
                                  for name in cell.params},
                    'cache': cache,
                    'automatic_policy': module.spec.automatic_optimizations,
                    'dense_window': [[int(axis.lower), int(axis.stop)] for axis in cell.dense_axes]
                    if cache == 'dense' else [],
                    'dense_logical_domain': [[int(axis.lower), int(axis.stop)] for axis in cell.dense_logical_axes]
                    if cache == 'dense' else [],
                    'associative_logical_domain': {
                        axis.key: [int(axis.lower), int(axis.stop)] for axis in cell.associative_logical_axes
                    } if cache in ('native-numeric-keys', 'dictionary') else {},
                    'declared_sizes': _declared_sizes(cls, cell) if cache != 'none' else {},
                    'enum_domains': {name: f'{e[0]}.{e[1]} ({e[2]} members)' for name, e
                                     in getattr(getattr(cell, '_rt', None), 'enum_args', {}).items()},
                    **({'param_bound_checks': [
                        {'parameter': c.parameter, 'limit': c.limit, 'bound': c.bound,
                         'margin': c.margin, 'declared': c.declared}
                        for c in getattr(cls, 'collect_bound_checks', ()) if c.cell == cell.name]}
                       if any(c.cell == cell.name for c in getattr(cls, 'collect_bound_checks', ()))
                       else {}),
                    'numeric_primary_capacity': primary_capacity(cell)
                    if cache == 'native-numeric-keys' else 0,
                    'power_memo': bool(getattr(cell, 'power_memo', None)),
                    'dense_offsets': [int(offset) for offset in cell.cache_offsets]
                    if cache == 'dense' else [],
                    'exact_spill': cache == 'dense' and bool(cell.has_dense_spill),
                    'nogil_qualified': False,
                    'native_helpers': getattr(cell, 'native_helpers', []),
                    'native_numeric_helpers': getattr(cell, 'native_numeric_helpers', []),
                    'table_reads_typed': getattr(cell, 'table_reads_typed', 0),
                    'table_reads_untyped': getattr(cell, 'table_reads_untyped', []),
                    'record_reads_untyped': getattr(cell, 'record_reads_untyped', []),
                    'table_reads_keys': getattr(cell, 'table_reads_keys', []),
                    'native_reductions': getattr(cell, 'native_reductions', 0),
                    **({'collect_outputs': cell.collect_leaves} if getattr(cell, 'collect_leaves', None) else {}),
                    'native_enum_comparisons': getattr(cell, 'native_enum_comparisons', 0),
                    'native_enum_constants': getattr(cell, 'native_enum_constants', 0),
                    'row_fields_inlined': getattr(cell, 'row_fields_inlined', 0),   # diag-run8 F3
                    'row_cell_inlined': getattr(cell, 'row_cell_inlined', False),
                    'row_cell_kept': getattr(cell, 'row_cell_kept', None),
                    'native_tuple_int_loops': getattr(cell, 'native_tuple_int_loops', 0),
                    'native_expressions': sum(getattr(cell, key, 0) for key in (
                        'reductions_native', 'enum_memberships', 'minmax_native', 'math_native')),
                    'expressions_untyped': expressions_untyped(cell),
                    'powers_in_loops': list(getattr(cell, 'powers_in_loops', ())),
                    'guarded_integer_ops': getattr(cell, 'guarded_integer_ops', 0),
                    'object_locals': [{'local': name, 'reason': 'compared by identity (is / is not)'}
                                      for name in getattr(cell, 'python_object_locals', ())],
                    'expressions_object': expressions_object(cell),
                    **({'vectors': cell.vector_report} if getattr(cell, 'vector_report', None) else {}),
                    'cache_detail': cache_detail(cell, cache),
                    'slow_paths': slow_paths(cell, cache),
                    'reasons': reasons or ['original Cython representation'],
                }
    return result


def _declared_sizes(cls, cell):
    """{parameter: where its cells_param_size is declared} for a Cell's cache axes."""
    if not cell.params or not hasattr(cls, 'declared_axes'):
        return {}
    cls.cells_arg_sizes, cls.associative_arg_sizes
    return dict(cls.declared_axes.get(tuple(cell.params), {}))


def _window(axes):
    sizes = [int(axis.stop) - int(axis.lower) for axis in axes]
    count = 1
    for size in sizes:
        count *= size
    return [[int(axis.lower), int(axis.stop)] for axis in axes], sizes, count


OVER_LIMIT_WARNING = ('{cell}: window {window} (traced {ranges}) = {count:,} entries exceeds max_dense_entries '
                      '{limit:,}; its values are kept in the hash store on every route (D135). If the cell is '
                      'read only on a regular grid, key it on the grid index (conversion rule R9, grid key).')


def over_limit_warning(cell):
    """D135: the build warning for a hash-cached Cell whose integer window exceeds the dense limit, else None."""
    if not (cell.has_args() and cell.has_typeinfo() and cell.uses_native_numeric_cache and cell.is_int_args):
        return None
    from .automatic import max_dense_entries
    try:
        bounds, sizes, count = _window(cell.dense_logical_axes)
    except (KeyError, AttributeError, TypeError):
        return None
    limit = max_dense_entries(cell)
    if count <= limit:
        return None
    ranges = ', '.join(f'{name} in [{lo}, {hi})' for name, (lo, hi) in zip(cell.params, bounds))
    return OVER_LIMIT_WARNING.format(cell=getattr(cell, 'model_path', cell.fqname), window=' x '.join(map(str, sizes)), ranges=ranges,
                                     count=count, limit=limit)


USE_WARNING = ('{cell} ({params}): hash store by use: rectangle {ranges} = {count:,} entries, {use} '
               '(dense up to {limit:,} entries, or up to {factor} x the entries one policy fills). Remedies, in '
               'order: re-parametrise the cell if its parameters are tied (one grid key, conversion rule R9); mark '
               'it not cached if each value is read once; tighten or correct the declaration.')


def _use_text(count, filled):
    if filled <= 0:
        return 'no entry filled by any policy in the sample'
    return f'{count / filled:,.1f} x the {filled:,} entries one policy fills in the sample'


def _store_axes(cell):
    """(ranges text, count) of the rectangle the store decision uses: declared axes as declared, sampled ones
    as sampled (observed_axes)."""
    from .automatic import declared_axes, observed_axes
    declared = declared_axes(cell)
    axes = observed_axes(cell, cell.dense_logical_axes)
    count = 1
    for axis in axes:
        count *= int(axis.stop) - int(axis.lower)
    ranges = ' x '.join(f'{axis.key} [{int(axis.lower)}, {int(axis.stop)}) '
                        f'{"declared" if axis.key in declared else "sampled"}' for axis in axes)
    return ranges, count


def use_warning(cell):
    """R-3b: the build warning for a multi-argument Cell put in the hash store because its rectangle exceeds
    the ceiling and DENSE_USE_FACTOR times its use, else None (an explicit max_dense_entries has its own)."""
    from .automatic import DENSE_USE_FACTOR, dense_by_use, filled_per_receiver, max_dense_entries, \
        max_dense_entries_set
    if not (len(cell.params) >= 2 and cell.has_typeinfo() and cell.uses_native_numeric_cache
            and cell.is_int_args) or max_dense_entries_set(cell):
        return None
    try:
        ranges, count = _store_axes(cell)
    except (KeyError, AttributeError, TypeError):
        return None
    limit = max_dense_entries(cell)
    if count <= limit or dense_by_use(cell, count):
        return None
    return USE_WARNING.format(cell=getattr(cell, 'model_path', cell.fqname), params=', '.join(cell.params),
                              ranges=ranges, count=count, use=_use_text(count, filled_per_receiver(cell)),
                              limit=limit, factor=DENSE_USE_FACTOR)


def store_reason(cell, cache):
    """R-3b: per multi-argument integer-keyed Cell, the store chosen and why (build report)."""
    from .automatic import DENSE_USE_FACTOR, filled_per_receiver, max_dense_entries, max_dense_entries_set
    ranges, count = _store_axes(cell)
    limit, filled = max_dense_entries(cell), filled_per_receiver(cell)
    use = _use_text(count, filled)
    source = 'explicit max_dense_entries' if max_dense_entries_set(cell) else 'max_dense_entries'
    if cache == 'dense':
        if count <= limit:
            return f'dense: {ranges} = {count:,} entries, within {source} {limit:,}'
        return f'dense by use: {ranges} = {count:,} entries, {use} (<= {DENSE_USE_FACTOR} x)'
    if count <= limit:
        return f'{cache}: {ranges} = {count:,} entries, within {source} {limit:,} but not dense (see hash_reason)'
    if max_dense_entries_set(cell):
        return f'{cache}: {ranges} = {count:,} entries exceeds the explicit max_dense_entries {limit:,}'
    return f'{cache} by use: {ranges} = {count:,} entries, {use} (> {DENSE_USE_FACTOR} x)'


def cache_detail(cell, cache):
    """The cache of a Cell with parameters: kind, window, entries, bytes, and why it is not dense (D74;
    D135: one storage on every route, no collect-only window)."""
    if not cell.params or cache in ('none', 'scalar'):
        return {'kind': cache}
    detail = {'kind': cache}
    if (cache in ('native-numeric-keys', 'dictionary')
            and getattr(getattr(cell, 'parent', None), 'collect_shared_item', False)):
        detail['hash_reason'] = 'an ItemSpace reached by collect workers uses synchronized storage'
        return detail
    try:
        if len(cell.params) >= 2 and cell.has_typeinfo() and cell.is_int_args and cache in (
                'dense', 'native-numeric-keys'):
            from .automatic import filled_per_receiver
            ranges, count = _store_axes(cell)
            filled = filled_per_receiver(cell)
            detail.update(filled_per_policy=filled, use_ratio=round(count / filled, 1) if filled else None,
                          store_reason=store_reason(cell, cache))
        if cache == 'dense':
            window, sizes, count = _window(cell.dense_axes)
            detail.update(window=window, entries=count, bytes=count * 12)
            return detail
        if cache == 'vector-dense':
            from .vectors import vector_cache_plan
            axes, count, size = vector_cache_plan(cell.parent)[0][cell.name]
            detail.update(window=_window(axes)[0], entries=count, bytes=size)
            return detail
        if cache not in ('native-numeric-keys', 'dictionary'):
            return detail
        from .automatic import max_dense_entries
        limit = max_dense_entries(cell)
        vector_refused = (vector_refusal(cell) if cache == 'dictionary' else None)
        if vector_refused:
            detail['hash_reason'] = vector_refused
        elif cache == 'dictionary':
            detail['hash_reason'] = ('return or parameter types are not native numbers'
                                     if cell.has_typeinfo() else 'formula not observed by the sample')
        elif not cell.is_int_args:
            detail['hash_reason'] = 'key type: the parameters are not all C integers'
        else:
            window, sizes, count = _window(cell.dense_logical_axes)
            detail.update(window=window, entries=count)
            detail['hash_reason'] = (f'window {" x ".join(map(str, sizes))} = {count:,} entries exceeds '
                                     f'max_dense_entries {limit:,}' if count > limit
                                     else 'negative arguments without a shifted window')
    except Exception as error:      # an axis without bounds: the kind alone
        detail.setdefault('hash_reason', f'no bounded window ({type(error).__name__})')
    return detail


def vector_refusal(cell):
    """Why a vector cell with parameters is not on a VecCache (budget or window), else None."""
    if not getattr(cell, 'vector_shape', None) or getattr(cell, 'parent', None) is None:
        return None
    try:
        from .vectors import vector_cache_plan
        return vector_cache_plan(cell.parent)[1].get(cell.name)
    except Exception:       # a class info without cells: no plan
        return None


def slow_paths(cell, cache):
    """diag-jp: work of a compiled formula that still runs on Python objects or a slow cache,
    which the other report keys do not flag as such: a cell with parameters on a hash or
    dictionary cache (and why it is not dense), lowered reductions that iterate or add Python
    objects, table reads whose key parts are Python objects, powers left on Python semantics in
    a loop, and a cell with parameters that returns a Python object. Each entry: kind, detail,
    per_step (the cell has parameters)."""
    per_step = bool(cell.params)
    entries = []
    if per_step and cache in ('native-numeric-keys', 'dictionary'):
        detail = f'{cache} cache'
        refused = vector_refusal(cell) if cache == 'dictionary' else None
        if refused:
            detail += f': {refused}'
        try:
            if refused:
                raise LookupError(refused)
            from .automatic import max_dense_entries
            limit = max_dense_entries(cell)
            axes = list(getattr(cell, 'associative_logical_axes', ()) or ())
            if axes and len(axes) == len(cell.params):
                sizes = [int(axis.stop) - int(axis.lower) for axis in axes]
                count = 1
                for size in sizes:
                    count *= size
                detail += (f': dense window {" x ".join(map(str, sizes))} = {count:,} entries exceeds '
                           f'max_dense_entries {limit:,}' if count > limit
                           else ': the arguments are not all sampled C integers in a bounded range')
        except Exception:       # an axis without bounds: the cache kind alone
            pass
        entries.append({'kind': 'hash-cache', 'detail': detail, 'per_step': per_step})
    for entry in getattr(cell, 'reductions_object', ()) or ():
        entries.append({'kind': 'reduction-on-objects', 'detail': f"line {entry['line']}: {entry['expr']}: "
                        f"{entry['reason']}", 'per_step': per_step})
    for key in getattr(cell, 'table_reads_keys', ()) or ():
        if 'object (' in key:
            entries.append({'kind': 'table-key-object', 'detail': key, 'per_step': per_step})
    for entry in getattr(cell, 'record_reads_untyped', ()) or ():
        # D78: record field reads left on Python objects (no held record handle)
        entries.append({'kind': 'record-read-object', 'detail': f"{entry['read']}: {entry['reason']}",
                        'per_step': per_step})
    for power in getattr(cell, 'powers_in_loops', ()) or ():
        entries.append({'kind': 'power-on-objects-in-loop', 'detail': power, 'per_step': per_step})
    if per_step and cache != 'none' and cell.get_rettype_expr(c_style=True) == 'object' and (
            getattr(cell, 'vector_report', None) is None):
        entries.append({'kind': 'object-return', 'detail': 'returns a Python object on every call '
                        '(boxed at the cell boundary)', 'per_step': per_step})
    return entries


def expressions_untyped(cell):
    """Expression forms (W1.4) left to the interpreter: reductions, n-argument
    min/max, enum membership and math calls, each with its line and reason."""
    entries = [entry for key in ('reductions_untyped', 'enum_untyped', 'minmax_untyped', 'math_untyped')
               for entry in getattr(cell, key, ())]
    return sorted(entries, key=lambda entry: (entry['line'], entry['form'], entry['expr']))


def expressions_object(cell):
    """Built-in forms (W1.8) compiled with comparisons or calls on Python objects, because
    an operand's class is not known at build time; each with its line and the operand
    classes. They no longer call the built-in."""
    entries = [entry for key in ('minmax_object', 'reductions_object', 'enum_object', 'math_object')
               for entry in getattr(cell, key, ())]
    return sorted(entries, key=lambda entry: (entry['line'], entry['form'], entry['expr']))


# ---------------------------------------------------------------- M1 table reads (W1.1)
TABLE_TYPE = 'modelx_helpers.immutable_inputs.Table'
TABLE_READ_ATTRS = ('loc', 'at', 'index', 'iterrows')
NOT_LOWERED = 'not lowered to a typed handle read; runs as a Python call'
KEPT_PYTHON = 'formula kept as a Python method ({why}): runs as a Python call'
PANDAS_RECEIVER = ('receiver is a pandas DataFrame, Series or Index, not a helper Table: never lowered; '
                   'pandas runs on every evaluation')
TABLE_READ_WARNING = '{cell}: per-step table read is not typed: {read} (formula line {line}): {reason}'


def untyped_read_warning(cell, entry):
    """Build warning text for one per-step untyped read (an entry of table_reads_untyped)."""
    return TABLE_READ_WARNING.format(cell=cell, **entry)


def _binds_local(statement, local):
    """Whether an Assign/AnnAssign binds only names tracked as receiver locals."""
    import libcst as cst
    targets = ([t.target for t in statement.targets] if isinstance(statement, cst.Assign)
               else [statement.target])
    return all(isinstance(t, cst.Name) and t.value in local for t in targets)


class TableReadSite:
    """One pandas-spelled read on a traced helper Table in a formula (before lowering)."""
    __slots__ = ('node', 'kind', 'read', 'line')

    def __init__(self, node, kind, read, line):
        self.node, self.kind, self.read, self.line = node, kind, read, line

    def entry(self, per_step, reason=None):
        return {'read': self.read, 'kind': self.kind, 'line': self.line,
                'per_step': per_step, 'reason': reason or NOT_LOWERED}


def table_read_sites(node, resolver, cls_info, pandas=False, receivers=None):
    """The table reads of one formula FunctionDef, in source order. Read-only.

    A receiver is a traced helper Table when it is a self-rooted call to a Cell whose every
    sampled result is a helper Table (``resolver.table_results``), a ``self.<ref>`` whose value
    is a helper Table, or a local name assigned (anywhere in the formula) from such a
    receiver or from ``<receiver>.loc[...]`` / ``.at[...]`` (a sub-table or a row), or bound
    by ``for ... in <receiver>.iterrows()``. A site is ``<receiver>.loc|at|index|iterrows``
    (kind = the attribute) or ``<receiver>[...]`` (kind ``column`` on a table, ``row`` on a
    row local). Reads inside comprehensions and generators count. ``line`` is the line within
    the formula, 1 being the ``def`` line.

    With ``pandas`` true the receivers are pandas objects instead: zero-argument Cells whose
    every sampled result is a pandas DataFrame, Series or Index (``resolver.pandas_results``)
    and ``self.<ref>`` References holding one. Every use of such a receiver is a site, not
    only the table-read spellings: any attribute or method (kind = the attribute name, or
    ``attribute``), a subscript, and the receiver itself used as a value (kind ``use``:
    iterated, passed to a call, returned). Those sites are reported, never lowered.
    """
    import libcst as cst
    from libcst.metadata import MetadataWrapper, PositionProvider
    from .usage import RESOLVED, _extract_self_chain

    if receivers is not None:      # auto-tables: every use of the given cells (admitted tables)
        pandas, tables, refs = True, set(receivers), set()
    elif pandas:
        from .usage import pandas_type_id
        tables = set(getattr(resolver, 'pandas_results', {}))
        refs = {name for name, ref in getattr(cls_info, 'refs', {}).items()
                if getattr(ref, 'type_', None) is not None and pandas_type_id(ref.type_)}
    else:
        tables = {name for name, kind in resolver.table_results.items() if kind == TABLE_TYPE}
        refs = {name for name, ref in getattr(cls_info, 'refs', {}).items()
                if getattr(ref, 'type_', None) is not None
                and f'{ref.type_.__module__}.{ref.type_.__qualname__}' == TABLE_TYPE}
    if not tables and not refs:
        return []

    def receiver(expr, local):
        if isinstance(expr, cst.Call) and not expr.args:
            chain = _extract_self_chain(expr.func)
            if isinstance(chain, list) and chain:
                resolved, name = resolver.resolve_chain(cls_info.fqname, chain)
                return resolved == RESOLVED and name in tables
            return False
        if isinstance(expr, cst.Attribute) and isinstance(expr.value, cst.Name):
            return expr.value.value == 'self' and expr.attr.value in refs
        return isinstance(expr, cst.Name) and expr.value in local

    def derived(expr, local):
        """A value read from a table receiver that is itself table-like (sub-table or row)."""
        if receiver(expr, local):
            return True
        return (isinstance(expr, cst.Subscript) and isinstance(expr.value, cst.Attribute)
                and expr.value.attr.value in ('loc', 'at') and receiver(expr.value.value, local))

    class Locals(cst.CSTVisitor):
        def __init__(self):
            self.assigned = []      # (name, value)
            self.rows = set()

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    self.assigned.append((target.target.value, node.value))

        def visit_AnnAssign(self, node):
            if isinstance(node.target, cst.Name) and node.value is not None:
                self.assigned.append((node.target.value, node.value))

        def _iterrows(self, target, iterable):
            if (isinstance(iterable, cst.Call) and isinstance(iterable.func, cst.Attribute)
                    and iterable.func.attr.value == 'iterrows' and isinstance(target, cst.Tuple)
                    and len(target.elements) == 2 and isinstance(target.elements[1].value, cst.Name)):
                self.assigned.append((target.elements[1].value.value, iterable.func.value))
                self.rows.add(target.elements[1].value.value)

        def visit_For(self, node):
            self._iterrows(node.target, node.iter)

        def visit_CompFor(self, node):
            self._iterrows(node.target, node.iter)

    found = Locals()
    node.visit(found)
    local, rows = set(), set()
    changed = True
    while changed:      # fixed point: locals assigned from other table locals
        changed = False
        for name, value in found.assigned:
            if name not in local and derived(value, local):
                local.add(name)
                changed = True
                if name in found.rows or not receiver(value, local - {name}):
                    rows.add(name)      # .loc/.at result or iterrows row: may be a row

    sites = []

    bare = []      # pandas mode: receivers used as values, filtered once parents are known

    class Sites(cst.CSTVisitor):
        def visit_Attribute(self, expr):
            if receiver(expr.value, local):
                if expr.attr.value in TABLE_READ_ATTRS:
                    sites.append((expr, expr.attr.value))
                elif pandas:
                    sites.append((expr, 'attribute'))
            if pandas and receiver(expr, local):
                bare.append(expr)

        def visit_Subscript(self, expr):
            if receiver(expr.value, local):
                sites.append((expr, 'row' if isinstance(expr.value, cst.Name)
                              and expr.value.value in rows else 'column'))

        def visit_Call(self, expr):
            if pandas and receiver(expr, local):
                bare.append(expr)

        def visit_Name(self, expr):
            if pandas and receiver(expr, local):
                bare.append(expr)

    node.visit(Sites())
    if not sites and not bare:
        return []
    module = cst.Module(body=[node])
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)     # same node objects
    positions = wrapper.resolve(PositionProvider)
    first = positions[node.name].start.line
    parents = {}

    class Parents(cst.CSTVisitor):
        def on_visit(self, child):
            for grandchild in child.children:
                parents[id(grandchild)] = child
            return True

    node.visit(Parents())
    for expr in bare:
        parent = parents.get(id(expr))
        if isinstance(parent, (cst.Attribute, cst.Subscript)) and parent.value is expr:
            continue        # a read already listed (or self.<ref>.x on a non-pandas path)
        if isinstance(parent, cst.Attribute) and parent.attr is expr:
            continue        # an attribute name that happens to spell a local
        if isinstance(parent, cst.AssignTarget) or (
                isinstance(parent, (cst.Assign, cst.AnnAssign)) and getattr(parent, 'value', None) is expr
                and _binds_local(parent, local)):
            continue        # the local itself, or its binding: its uses are the sites
        if isinstance(parent, cst.AnnAssign) and parent.target is expr:
            continue
        holder, up = expr, parent
        while isinstance(up, (cst.Element, cst.Tuple, cst.List)):
            holder, up = up, parents.get(id(up))
        if isinstance(up, (cst.For, cst.CompFor)) and up.target is holder:
            continue        # a loop target (the row of iterrows): its reads are the sites
        sites.append((expr, 'use'))
    if not sites:
        return []
    result = []
    for expr, kind in sites:
        shown = expr
        for _ in range(2):      # T.loc[...]; T.index.max(); k in T.index stays T.index
            parent = parents.get(id(shown))
            if isinstance(parent, (cst.Subscript, cst.Call)) and (
                    getattr(parent, 'value', None) is shown or getattr(parent, 'func', None) is shown):
                shown = parent
            elif isinstance(parent, cst.Attribute) and parent.value is shown:
                shown = parent
            else:
                break
        text = re.sub(r'\s*#[^\'"\n]*$', '', module.code_for_node(shown), flags=re.M)    # comments
        text = ' '.join(text.replace('( ', '(').replace('[ ', '[').split()).replace('( ', '(').replace('[ ', '[')
        result.append(TableReadSite(expr, kind, text, positions[expr].start.line - first + 1))
    result.sort(key=lambda site: site.line)
    return result
