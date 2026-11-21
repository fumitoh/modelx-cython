"""Lower typed reductions using existing CST nodes and binding metadata."""
from functools import cached_property

import libcst as cst
from libcst.metadata import ExpressionContext

from .cell_entries import exact_receiver
from .consts import CY_MOD
from .helpers import module_name
from .helper_members import helper_reference
from .scopes import CstFacts, FormulaTransformer, Names, OPAQUE_SCOPES as OPAQUE, has_scope_effects, lexical_nodes
from .syntax import arguments, code, literal_value, walk
from .typedefs import primitive_kind


def formula_lines(formula):
    """Map a node of ``formula`` to its line in it (1 = the ``def`` line; 0 if not found)."""
    from libcst.metadata import MetadataWrapper, PositionProvider
    try:
        positions = MetadataWrapper(cst.Module(body=[formula]), unsafe_skip_copy=True).resolve(PositionProvider)
        first = positions[formula.name].start.line
    except Exception:        # a formula assembled by an earlier pass may not round-trip
        return lambda node: 0
    return lambda node: positions[node].start.line - first + 1 if node in positions else 0


def untyped_entry(lines, node, form, reason):
    """One admission entry of an expression form left to the interpreter (W1.4)."""
    import re
    text = re.sub(r'\s*#[^\'"\n]*$', '', cst.Module(body=[]).code_for_node(node), flags=re.M)
    text = ' '.join(text.split()).replace('( ', '(').replace('[ ', '[')
    return {'expr': text if len(text) <= 160 else text[:157] + '...', 'form': form,
            'line': lines(node) if lines is not None else 0, 'reason': reason}


BUILTIN_PROVIDER = 'modelx_cython._native_iterables'
STREAMS_PROVIDER = 'modelx_cython.runtime._native_streams'
BUILTIN_REDUCTIONS = frozenset(('all', 'any', 'max', 'min', 'sum'))


def has_reduction_contract(formula, refs, builtin_reductions=frozenset()):
    """Find declared consumers and supported built-in reductions of one iterable."""
    return any(isinstance(node, cst.Call) and isinstance(node.func, cst.Name)
               and node.func.value in builtin_reductions and _one_iterable(node)
               for node in lexical_nodes(formula.body.body))


def _one_iterable(call):
    """``R(iterable)`` (and ``sum(iterable, start)``): the built-in reduction form (W1.4)."""
    if not call.args or any(arg.star for arg in call.args) or call.args[0].keyword is not None:
        return False
    if len(call.args) == 1:
        return True
    return (len(call.args) == 2 and call.func.value == 'sum'
            and (call.args[1].keyword is None or call.args[1].keyword.value == 'start'))


def _call(node, name, count):
    return arguments(node, name, count) is not None


def _parts(generator):
    part = generator.for_in
    while part is not None:
        yield part
        part = part.inner_for_in


def safe_range_increment(iterator):
    """Prove the final C loop increment, including an overshoot of stop."""
    args = arguments(iterator)
    if args is None:
        return False
    if len(args) < 3:
        return True
    try:
        step = literal_value(args[2])
        if type(step) not in (int, bool):
            return False
        if step in (-1, 1):
            return True
        bounds = [literal_value(arg) for arg in args]
        if not all(primitive_kind(value) in ('longlong', 'bint') for value in bounds):
            return False
        values = range(*bounds)
        return not values or primitive_kind(values[-1] + step) == 'longlong'
    except (ValueError, TypeError):
        return False


class _Unsupported(Exception):
    """A form the general reduction lowering leaves to the interpreter (the reason)."""


_ATOMIC = (cst.Name, cst.Attribute, cst.Integer, cst.Float, cst.SimpleString)
_UNROLL_LIMIT = 32            # literal-tuple instances unrolled per reduction
def _empty_message():
    """The running interpreter's own message for min() of an empty iterable, with the function
    name as a placeholder: "iterable argument is empty" from 3.12, "arg is an empty sequence"
    before. The compiled reduction raises exactly the built-in's message."""
    try:
        min(())
    except ValueError as error:
        return str(error).replace('min', '{name}', 1)
    raise AssertionError('min(()) did not raise')


_EMPTY = _empty_message()
_COMPARE = {'min': '<', 'max': '>'}


class _Substitute(cst.CSTTransformer):
    """Replace generator variables (by name, read or store context) with nodes."""
    def __init__(self, mapping, contexts):
        super().__init__()
        self.mapping, self.contexts = mapping, contexts

    def leave_Name(self, before, after):
        if before.value not in self.mapping or before not in self.contexts:
            return after
        value = self.mapping[before.value]
        if after.lpar or not isinstance(value, _ATOMIC):
            value = value.with_changes(lpar=(cst.LeftParen(),), rpar=(cst.RightParen(),))
        return value


def _names(target):
    if isinstance(target, cst.Name):
        return [target.value]
    if isinstance(target, (cst.Tuple, cst.List)) and target.elements and all(
            isinstance(element, cst.Element) and isinstance(element.value, cst.Name)
            for element in target.elements):
        return [element.value.value for element in target.elements]
    raise _Unsupported('a generator target that is not a name or a tuple of names')


class GeneralReductions:
    """W1.4: built-in sum/min/max/all/any over one iterable as native loops.

    Applies where the specialised lowerings above decline. The iterable is a generator
    or list comprehension (any number of ``for``/``if`` clauses over ``range`` with C
    integer bounds, a literal tuple, which is unrolled, or any other iterable: a cell
    value, a tuple-valued cell, ``zip``), or a direct iterable argument. The result is
    the built-in's: ``sum`` through ``_float_sum_state.PySum`` (CPython 3.14 branch by
    branch), ``min``/``max`` keep the first extreme and raise on empty input, ``all``/
    ``any`` short-circuit as the generator would. Generator variables are renamed;
    StopIteration raised in a generator body becomes RuntimeError as in a generator.
    The statement is ``return R(...)``, ``x = R(...)`` or ``x op= R(...)``.
    """

    def general(self, before, after):
        if len(before.body) != 1 or not self.builtin_bounds:
            return after
        statement = before.body[0]
        if isinstance(statement, cst.Return) and statement.value is not None:
            mode, target = 'return', None
        elif (isinstance(statement, cst.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0].target, cst.Name)):
            mode, target = 'assign', statement.targets[0].target.value
        elif isinstance(statement, cst.AugAssign) and isinstance(statement.target, cst.Name):
            mode, target = 'augassign', statement.target.value
        else:
            return after
        call = statement.value
        if not self.builtin_site(call) or id(call) in self.general_done:
            return after
        try:
            text = self.lower_general(statement, call, mode, target)
        except _Unsupported as reason:
            self.general_declined[id(call)] = (call, f'{reason}: built-in call')
            return after
        self.general_done.add(id(call))
        self.general_count += 1
        self.providers.add(STREAMS_PROVIDER)
        return cst.FlattenSentinel(cst.parse_module(text).body)

    # ---------------------------------------------------------------- analysis
    def lower_general(self, statement, call, mode, target):
        name = call.func.value
        source = call.args[0].value
        start = call.args[1].value if len(call.args) == 2 else None
        prefix = self.reserve_prefix()
        if isinstance(source, (cst.GeneratorExp, cst.ListComp)):
            generator = isinstance(source, cst.GeneratorExp)
            parts = [(part.target, part.iter, [condition.test for condition in part.ifs])
                     for part in _parts(source)]
            if any(part.asynchronous is not None for part in _parts(source)):
                raise _Unsupported('an asynchronous generator')
            element, facts = source.elt, CstFacts(source)
            inner = [element, *(test for _, _, tests in parts for test in tests),
                     *(iterator for _, iterator, _ in parts[1:])]
            if any(isinstance(node, OPAQUE) for part in inner for node in lexical_nodes((part,))
                   if node is not part) or any(isinstance(part, OPAQUE) for part in inner):
                raise _Unsupported('a nested function or comprehension in the generator')
        else:
            generator, facts = False, None
            item = cst.Name(f'{prefix}_item')
            parts, element = [(item, source, [])], item
        bound, renamed = [], {}
        for index, (part_target, iterator, tests) in enumerate(parts):
            names = _names(part_target)
            if set(names) & set(bound) or len(set(names)) != len(names):
                raise _Unsupported('a generator variable bound twice')
            if index and facts is not None and set(CstFacts(iterator).reads) & set(names):
                raise _Unsupported('an iterable that reads its own variable')
            bound.extend(names)
            for name_ in names:
                renamed[name_] = f'{prefix}_v_{name_}' if facts is not None else name_
        if facts is not None:
            later = {}
            for index, (part_target, iterator, tests) in enumerate(parts):
                for name_ in _names(part_target):
                    later[name_] = index
            for index, (part_target, iterator, tests) in enumerate(parts):
                reads = set(CstFacts(iterator).reads) if index else set()
                for test in tests:
                    reads |= set(CstFacts(test).reads)
                if any(later.get(name_, -1) > index or (later.get(name_, -1) == index and name_ in
                       (set(CstFacts(iterator).reads) if index else set())) for name_ in reads):
                    raise _Unsupported('a variable read before its for clause binds it')
        if start is not None:
            raise _Unsupported('a second argument')
        self.general_state = dict(prefix=prefix, name=name, generator=generator, facts=facts,
                                  renamed=renamed, mode=mode, target=target,
                                  local_types={}, kinds=set(), alias=module_name(STREAMS_PROVIDER))
        return self.emit_general(statement, call, parts, element)

    def substituted(self, node, mapping):
        state = self.general_state
        if state['facts'] is None or not mapping:
            return node
        return node.visit(_Substitute(mapping, state['facts'].contexts))

    def element_kind(self, node):
        kind = self.kind(node, self.general_state['local_types'])
        return kind if kind in ('double', 'longlong', 'bint') else None

    # ---------------------------------------------------------------- emission
    def emit_general(self, statement, call, parts, element):
        state = self.general_state
        p, name, alias = state['prefix'], state['name'], state['alias']
        # all/any stop at the deciding item, except over a list comprehension, which
        # evaluates every element before the built-in sees the list.
        early = name in ('all', 'any') and not isinstance(call.args[0].value, cst.ListComp)
        state['exit_return'] = early and state['mode'] == 'return'
        state['flag_exit'] = early and not state['exit_return']
        prologue, declarations = [], []
        # The outermost iterable is evaluated where the generator is created.
        classes = [self.classify(index, part) for index, part in enumerate(parts)]
        instances = 1
        for kind_, data in classes:
            if kind_ == 'tuple':
                instances *= len(data)
        if instances > _UNROLL_LIMIT:
            classes = [('iter', part[1]) if kind_ == 'tuple' else (kind_, data)
                       for (kind_, data), part in zip(classes, parts)]
        outer_kind, outer = classes[0]
        mapping0 = {}
        if outer_kind == 'range':
            names = [f'{p}_bound_{i}' for i in range(len(outer))]
            prologue += [f'{n}: _mx_cy.longlong = {code(value)}' for n, value in zip(names, outer)]
            classes[0] = ('range', [cst.Name(n) for n in names])
        elif outer_kind == 'tuple':
            classes[0] = ('tuple', self.tuple_temps(outer, prologue, {}, 0))
        else:
            iterator = code(outer)
            if state['facts'] is None:          # a direct iterable: iter() as the built-in does
                iterator = f'iter({iterator})'
            prologue.append(f'{p}_iterable = ({iterator})')
            classes[0] = ('iter', cst.Name(f'{p}_iterable'))
        # Body
        body = self.emit_level(0, parts, classes, element, mapping0, '')
        kinds = state['kinds']
        # diag-jp: what of this lowered reduction still runs on Python objects (build report)
        objects = []
        if any(kind_ == 'iter' for kind_, _ in classes):
            objects.append('iterates a Python object (an iterable that is not a range of C integers '
                           'or a literal tuple), so its loop variable is an object')
        if name == 'sum' and None in kinds:
            objects.append('an element is not a double or C integer: the sum adds Python objects')
        if objects:
            self.object_loops.append(untyped_entry(self.general_lines, call, name, '; '.join(objects)))
        # Start, state and result
        if name == 'sum':
            init = f'{alias}.sum_init(_mx_cy.address({p}_state))'
            declarations.append(f'{p}_state: {alias}.PySum')
            nonempty = outer_kind == 'tuple' and not any(tests for _, _, tests in parts) and all(
                kind_ == 'tuple' for kind_, _ in classes)
            exact_float = kinds == {'double'} and nonempty
            if state['mode'] == 'return' and self.cells.get_rettype_expr(c_style=True) == 'double':
                exact_float = True
            result = (f'{alias}.sum_result_f64(_mx_cy.address({p}_state))' if exact_float
                      else f'{alias}.sum_result(_mx_cy.address({p}_state))')
            prologue.append(init)
        elif name in ('min', 'max'):
            typed = next(iter(kinds)) if len(kinds) == 1 and None not in kinds else None
            declarations.append(f'{p}_seen: _mx_cy.bint = False')
            if typed:
                declarations += [f'{p}_best: _mx_cy.{typed} = 0', f'{p}_value: _mx_cy.{typed}']
            else:
                declarations += [f'{p}_best: object = None', f'{p}_value: object']
            result = f'{p}_best'
        else:
            declarations.append(f'{p}_result: _mx_cy.bint = {name == "all"}')
            if state['flag_exit']:
                declarations.append(f'{p}_done: _mx_cy.bint = False')
            result = f'{p}_result'
        declarations += state.get('typed_vars', [])
        lines = declarations + prologue
        if state['generator']:
            lines += ['try:'] + ['    ' + line for line in body] + [
                f'except StopIteration as {p}_error:',
                f'    raise RuntimeError("generator raised StopIteration") from {p}_error']
        else:
            lines += body
        if name in ('min', 'max'):
            lines += [f'if not {p}_seen:', f'    raise ValueError({_EMPTY.format(name=name)!r})']
        if state['mode'] == 'return':
            final = f'return {result}'
        elif state['mode'] == 'assign':
            final = f'{state["target"]} = {result}'
        else:
            final = f'{state["target"]} {code(statement.operator).strip()} {result}'
        lines.append(final)
        if name == 'sum':
            lines = (lines[:len(declarations)] + lines[len(declarations):len(declarations) + len(prologue)]
                     + ['try:'] + ['    ' + line for line in lines[len(declarations) + len(prologue):]]
                     + ['finally:', f'    {alias}.sum_clear(_mx_cy.address({p}_state))'])
        return '\n'.join(lines) + '\n'

    def classify(self, index, part):
        part_target, iterator, _ = part
        bounds = arguments(iterator, 'range')
        if (bounds is not None and 1 <= len(bounds) <= 3 and isinstance(part_target, cst.Name)
                and 'range' not in self.body_facts.stores and 'range' not in self.cells.params
                and safe_range_increment(iterator)
                and (index or all(self.bound_kind(bound, {}) in ('longlong', 'bint') for bound in bounds))):
            return 'range', list(bounds)
        if (isinstance(iterator, cst.Tuple) and iterator.elements
                and all(isinstance(element, cst.Element) for element in iterator.elements)):
            values = [element.value for element in iterator.elements]
            if isinstance(part_target, cst.Name) or all(
                    isinstance(value, cst.Tuple) and len(value.elements) == len(_names(part_target))
                    and all(isinstance(item, cst.Element) for item in value.elements) for value in values):
                return 'tuple', values
        return 'iter', iterator

    def constant(self, node):
        if isinstance(node, (cst.Integer, cst.Float, cst.SimpleString)):
            return True
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
            return isinstance(node.expression, (cst.Integer, cst.Float))
        if isinstance(node, cst.Name) and node.value in ('True', 'False', 'None'):
            return True
        return (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute)
                and isinstance(node.value.value, cst.Name) and node.value.value.value == 'self'
                and (self.cls_info.name, node.value.attr.value, node.attr.value) in self.enum_symbols)

    def tuple_temps(self, values, lines, mapping, level):
        """Evaluate a literal tuple's elements in order; constants are substituted."""
        p = self.general_state['prefix']
        result = []
        for index, value in enumerate(values):
            value = self.substituted(value, mapping)
            items = value.elements if isinstance(value, cst.Tuple) else None
            if self.constant(value) or (items is not None and all(self.constant(item.value) for item in items)):
                result.append(value)
                continue
            if items is not None:
                parts = []
                for position, item in enumerate(items):
                    if self.constant(item.value):
                        parts.append(item.value)
                    else:
                        temp = f'{p}_t{level}_{index}_{position}'
                        lines.append(f'{temp} = ({code(item.value)})')
                        parts.append(cst.Name(temp))
                result.append(cst.Tuple([cst.Element(part) for part in parts]))
            else:
                temp = f'{p}_t{level}_{index}'
                lines.append(f'{temp} = ({code(value)})')
                result.append(cst.Name(temp))
        return result

    def emit_level(self, level, parts, classes, element, mapping, indent):
        state = self.general_state
        part_target, _, tests = parts[level]
        kind_, data = classes[level]
        lines, p = [], state['prefix']
        names = _names(part_target)
        if kind_ == 'range' and level and not all(
                self.bound_kind(self.substituted(bound, mapping), state['local_types']) in ('longlong', 'bint')
                for bound in data):
            kind_, data = 'iter', parts[level][1]          # bounds not C integers: a range object
        if kind_ == 'range':
            variable = state['renamed'][names[0]]
            if variable not in state['local_types']:
                state['local_types'][variable] = 'longlong'
                state.setdefault('typed_vars', []).append(f'{variable}: _mx_cy.longlong')
            bounds = ', '.join(code(self.substituted(bound, mapping)) for bound in data)
            lines.append(f'{indent}for {variable} in range({bounds}):')
            inner = dict(mapping, **{names[0]: cst.Name(variable)})
            lines += self.emit_body(level, parts, classes, element, inner, indent + '    ', tests)
        elif kind_ == 'iter':
            iterator = data if level == 0 else self.substituted(data, mapping)
            variables = [state['renamed'][n] for n in names]
            target = (variables[0] if isinstance(part_target, cst.Name)
                      else ', '.join(variables) + (',' if len(variables) == 1 else ''))
            inner = dict(mapping, **{n: cst.Name(v) for n, v in zip(names, variables)})
            if (level == 0 and isinstance(part_target, cst.Name) and isinstance(iterator, cst.Name)
                    and self.integer_argument_only(names[0], parts, element)):
                # diag-scale P3: an exact tuple whose items this loop only passes as C-integer
                # arguments: index loop, items read borrowed; an item that is not an int fitting
                # in 64 bits takes the object path as written, any other iterable the loop as written
                alias, it, index, value = state['alias'], code(iterator), f'{p}_i', f'{variables[0]}_ll'
                for typed in (f'{index}: _mx_cy.Py_ssize_t', f'{value}: _mx_cy.longlong'):
                    if typed not in state.setdefault('typed_vars', []):
                        state['typed_vars'].append(typed)
                state['local_types'][value] = 'longlong'
                lines.append(f'{indent}if {alias}.exact_tuple({it}):')
                lines.append(f'{indent}    for {index} in range({alias}.tuple_size({it})):')
                lines.append(f'{indent}        if {alias}.tuple_int64({it}, {index}, _mx_cy.address({value})):')
                lines += self.emit_body(level, parts, classes, element, dict(mapping, **{names[0]: cst.Name(value)}),
                                        indent + ' ' * 12, tests)
                lines.append(f'{indent}        else:')
                lines.append(f'{indent}            {target} = {it}[{index}]')
                lines += self.emit_body(level, parts, classes, element, inner, indent + ' ' * 12, tests)
                lines.append(f'{indent}else:')
                lines.append(f'{indent}    for {target} in ({it}):')
                lines += self.emit_body(level, parts, classes, element, inner, indent + ' ' * 8, tests)
                self.tuple_int_loops = getattr(self, 'tuple_int_loops', 0) + 1
                return lines
            lines.append(f'{indent}for {target} in ({code(iterator)}):')
            lines += self.emit_body(level, parts, classes, element, inner, indent + '    ', tests)
        else:
            values = data
            wrap = state['flag_exit']
            if level:
                values = self.tuple_temps(values, lines, mapping, level)
                lines[:] = [indent + line for line in lines]
            block = []
            body_indent = indent + ('    ' if wrap else '')
            for value in values:
                if isinstance(part_target, cst.Name):
                    inner = dict(mapping, **{names[0]: value})
                else:
                    inner = dict(mapping, **{n: item.value for n, item in zip(names, value.elements)})
                block += self.emit_body(level, parts, classes, element, inner, body_indent, tests,
                                        loop=False)
            if wrap:
                lines += [f'{indent}while True:'] + block + [f'{indent}    break']
            else:
                lines += block
        return lines

    def integer_argument_only(self, name, parts, element):
        """diag-scale P3: every read of the loop variable ``name`` in the element and in the
        conditions is a positional argument of a call ``self.<cell>(...)`` of a compiled Cell of
        this class whose parameter there is a C ``long long`` or ``bint`` (no keywords, no star
        arguments, no closure). The value then reaches the Cell only through the integer boundary,
        so reading it as a C integer gives the same calls; any other read (returned, compared,
        stored, passed to Python, an attribute of it) keeps the object loop."""
        if len(parts) != 1:
            return False
        state = self.general_state
        facts = state['facts']
        if facts is None:
            return False
        cells = getattr(self.cls_info, 'cells', {})
        nodes = [element, *parts[0][2]]
        allowed, loads = set(), []
        for root in nodes:
            for node in walk(root):
                if isinstance(node, cst.Name) and node.value == name and facts.contexts.get(node) is ExpressionContext.LOAD:
                    loads.append(node)
                if (isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute)
                        and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'
                        and not any(arg.keyword or arg.star for arg in node.args)):
                    cell = cells.get(node.func.attr.value)
                    if cell is None or cell.called_with_kwargs or cell.body_has_closure:
                        continue
                    for arg, param in zip(node.args, cell.params):
                        if (isinstance(arg.value, cst.Name) and arg.value.value == name
                                and cell.get_argtype_expr(param, c_style=True) in ('long long', 'bint')):
                            allowed.add(id(arg.value))
        return bool(loads) and all(id(node) in allowed for node in loads)

    def emit_body(self, level, parts, classes, element, mapping, indent, tests, loop=True):
        state = self.general_state
        lines = []
        for test in tests:
            lines.append(f'{indent}if ({code(self.substituted(test, mapping))}):')
            indent += '    '
        if level + 1 < len(parts):
            lines += self.emit_level(level + 1, parts, classes, element, mapping, indent)
            if state['flag_exit']:
                lines.append(f'{indent}if {state["prefix"]}_done:')
                lines.append(f'{indent}    break')
            return lines
        return lines + [indent + line for line in self.action(self.substituted(element, mapping))]

    def action(self, element):
        state = self.general_state
        p, name, alias = state['prefix'], state['name'], state['alias']
        kind = self.element_kind(element)
        state['kinds'].add(kind)
        value = code(element)
        if name == 'sum':
            add = {'double': 'sum_add_f64', 'longlong': 'sum_add_ll', 'bint': 'sum_add_bint'}.get(
                kind, 'sum_add_object')
            return [f'{alias}.{add}(_mx_cy.address({p}_state), ({value}))']
        if name in ('min', 'max'):
            return [f'{p}_value = ({value})',
                    f'if not {p}_seen or {p}_value {_COMPARE[name]} {p}_best:',
                    f'    {p}_best = {p}_value',
                    f'    {p}_seen = True']
        hit = 'False' if name == 'all' else 'True'
        test = f'not ({value})' if name == 'all' else f'({value})'
        if state['exit_return']:
            return [f'if {test}:', f'    return {hit}']
        if state['flag_exit']:
            return [f'if {test}:', f'    {p}_result = {hit}', f'    {p}_done = True', '    break']
        return [f'if {test}:', f'    {p}_result = {hit}']


    # ---------------------------------------------------------------- W1.8: nested sites
    def hoist(self, statement, root):
        """A reduction nested in ``root`` (the value of ``statement``) as its own statement.

        Python evaluates an expression left to right; the reduction is lifted only where it
        is evaluated unconditionally (not behind ``and``/``or``, a conditional expression's
        branches or a chained comparison's later terms). Everything evaluated before it in
        the statement and not free of effects (a cell call, a local read, arithmetic) is
        first assigned to a temporary in the same order, so evaluation order and the
        exception raised first are those of the original statement. The lifted
        ``tmp = R(...)`` goes through the general lowering; if that declines, nothing is
        lifted. Returns (lines, new root) or None.
        """
        site, preceding = self.first_site(root)
        if site is None:
            return None
        if any(not self.effect_free(node) and _compile_time(node) for node in preceding):
            # Cython syntax put in by an earlier lowering (a cast, a C type, a method of a
            # typed handle) has no run-time value to keep in a temporary: lift nothing
            self.general_declined.setdefault(id(site), (site, (
                'evaluated after compiler-inserted Cython syntax (a cast, a C type or a typed-handle '
                'method) in the same statement, which cannot be held in a temporary: built-in call')))
            return None
        names, replacements, lines = self.names, {}, []
        for node in preceding:
            if self.effect_free(node):
                continue
            temp = names.reserve('_mx_hoist', family=True)
            lines += cst.parse_module(f'{temp} = ({code(node)})\n').body
            replacements[node] = cst.Name(temp)
        result = names.reserve('_mx_hoist', family=True)
        if isinstance(site, cst.BooleanOperation):
            # ``a and R`` / ``a or R``: a evaluated once, its truth tested once, R only when
            # the operator needs it; the result is a or R, as the operator gives it
            site_now = _replaced(site, {id(node): cst.Name(temp.value) for node, temp in replacements.items()})
            left = names.reserve('_mx_hoist', family=True)
            lines += cst.parse_module(f'{left} = ({code(site_now.left)})\n').body
            line = cst.SimpleStatementLine(body=[cst.Assign(
                targets=[cst.AssignTarget(cst.Name(result))], value=site_now.right)])
            lowered = self.leave_SimpleStatementLine(line, line)
            evaluated = list(lowered.nodes) if isinstance(lowered, cst.FlattenSentinel) else [lowered]
            kept = cst.parse_module(f'{result} = {left}\n').body
            first, second = (evaluated, kept) if isinstance(site.operator, cst.And) else (kept, evaluated)
            # one Python object for both branches: C types of the two would be promoted
            lines += cst.parse_module(f'{result}: object\n').body
            lines.append(cst.If(test=cst.Name(left), body=cst.IndentedBlock(body=first),
                                orelse=cst.Else(body=cst.IndentedBlock(body=second))))
        elif isinstance(site, cst.IfExp):
            # a conditional expression whose branch holds a reduction: an if statement
            # assigning each branch, which evaluates the test and one branch as it does
            site_now = _replaced(site, {id(node): cst.Name(temp.value) for node, temp in replacements.items()})
            branches = []
            for value in (site_now.body, site_now.orelse):
                line = cst.SimpleStatementLine(body=[cst.Assign(
                    targets=[cst.AssignTarget(cst.Name(result))], value=value)])
                lowered = self.leave_SimpleStatementLine(line, line)
                branches.append(list(lowered.nodes) if isinstance(lowered, cst.FlattenSentinel)
                                else [lowered])
            lines += cst.parse_module(f'{result}: object\n').body   # no promotion between branches
            lines.append(cst.If(test=site_now.test.with_changes(lpar=(), rpar=()),
                                body=cst.IndentedBlock(body=branches[0]),
                                orelse=cst.Else(body=cst.IndentedBlock(body=branches[1]))))
        else:
            line = cst.SimpleStatementLine(body=[cst.Assign(
                targets=[cst.AssignTarget(cst.Name(result))], value=site)])
            lowered = self.general(line, line)
            if lowered is line:
                return None
            lines += list(lowered.nodes) if isinstance(lowered, cst.FlattenSentinel) else [lowered]
        replacements[site] = cst.Name(result)
        mapping = {id(node): name.with_changes(lpar=node.lpar, rpar=node.rpar)
                   if getattr(node, 'lpar', ()) else name for node, name in replacements.items()}
        return lines, _replaced(root, mapping)

    def first_site(self, root):
        """The first reduction site evaluated unconditionally in ``root``, and the subtrees
        evaluated before it (in order), or (None, ())."""
        preceding, stop = [], []

        def search(node):
            if self.builtin_site(node) and id(node) in self.nested_sites:
                return node
            if isinstance(node, cst.IfExp) and any(
                    id(item) in self.nested_sites for branch in (node.body, node.orelse)
                    for item in _subtree(branch)):
                found = search(node.test)
                return found if found is not None or stop else node
            if isinstance(node, cst.BooleanOperation) and any(
                    id(item) in self.nested_sites for item in _subtree(node.right)):
                found = search(node.left)
                return found if found is not None or stop else node
            for child, conditional in _evaluation_order(node):
                if conditional:
                    stop.append(True)  # later sites run after a conditional part: not lifted
                    self.conditional_sites.update(id(item) for item in _subtree(child)
                                                  if id(item) in self.nested_sites)
                    return None
                found = search(child)
                if found is not None or stop:
                    return found
                preceding.append(child)
            return None

        site = search(root)
        if site is None:
            return None, ()
        # keep the outermost evaluated subtrees: a preceding child of an ancestor covers its own
        covered, result = set(), []
        for node in reversed(preceding):
            if id(node) in covered:
                continue
            result.append(node)
            covered.update(id(item) for item in _subtree(node))
        return site, tuple(reversed(result))

    def effect_free(self, node):
        """Evaluation can neither fail nor run code: literals, parameters, built-in names,
        ``self`` and ``self.<name>`` (a C field or a bound cell method)."""
        if isinstance(node, (cst.Integer, cst.Float, cst.SimpleString)):
            return True
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
            return isinstance(node.expression, (cst.Integer, cst.Float))
        if isinstance(node, cst.Name):
            return (node.value in ('self', 'True', 'False', 'None') or node.value in self.cells.params
                    and node.value not in self.body_facts.stores
                    or node.value in self.builtin_names)
        return (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name)
                and node.value.value == 'self')

    def leave_If(self, before, after):
        if not self.nested_sites or id(before) in self.elifs:
            return after
        hoisted = self.hoist(before, before.test)
        if hoisted is None:
            return after
        lines, test = hoisted
        return cst.FlattenSentinel([*lines, after.with_changes(test=test)])

    def visit_If(self, node):
        if isinstance(node.orelse, cst.If):
            self.elifs.add(id(node.orelse))
        return True

    def nested(self, before, after):
        """``return``/``x = ``/expression statements holding a nested reduction (W1.8)."""
        if not self.nested_sites or len(before.body) != 1:
            return after
        statement = before.body[0]
        if isinstance(statement, cst.Return) and statement.value is not None:
            pass
        elif isinstance(statement, (cst.Assign, cst.Expr)) or (
                isinstance(statement, cst.AnnAssign) and statement.value is not None):
            pass
        else:
            return after
        lines, root = [], statement.value
        while True:
            hoisted = self.hoist(statement, root)
            if hoisted is None:
                break
            new_lines, root = hoisted
            lines += new_lines
        if not lines:
            return after
        return cst.FlattenSentinel([*lines, before.with_changes(
            body=[statement.with_changes(value=root)])])


def _replaced(node, mapping):
    """``node`` with the subtrees whose id is in ``mapping`` replaced; every other subtree
    is kept as the same object (``deep_replace`` copies them), so site ids stay valid."""
    if id(node) in mapping:
        return mapping[id(node)]
    from dataclasses import fields
    changes = {}
    for field in fields(node):
        value = getattr(node, field.name)
        if isinstance(value, cst.CSTNode):
            new = _replaced(value, mapping)
            if new is not value:
                changes[field.name] = new
        elif isinstance(value, (tuple, list)) and any(isinstance(item, cst.CSTNode) for item in value):
            new = [_replaced(item, mapping) if isinstance(item, cst.CSTNode) else item for item in value]
            if any(a is not b for a, b in zip(new, value)):
                changes[field.name] = type(value)(new)
    return node.with_changes(**changes) if changes else node


def _compile_time(node):
    """``node`` holds a reference to the Cython namespace (``_mx_cy``): compiler-inserted
    syntax resolved at C compile time (casts, C types, typed-handle members), never a
    value of the model's own formula."""
    return any(isinstance(item, cst.Name) and item.value == CY_MOD for item in _subtree(node))


def _subtree(node):
    yield node
    for child in node.children:
        yield from _subtree(child)


def _evaluation_order(node):
    """Child expressions of ``node`` in the order Python evaluates them, with a flag for
    those evaluated only conditionally. Scopes (lambda, comprehensions) and other nodes
    give no children: nothing inside them is lifted."""
    if isinstance(node, cst.BinaryOperation):
        return ((node.left, False), (node.right, False))
    if isinstance(node, cst.UnaryOperation):
        return ((node.expression, False),)
    if isinstance(node, cst.BooleanOperation):
        return ((node.left, False), (node.right, True))
    if isinstance(node, cst.Comparison):
        return ((node.left, False), *((part.comparator, index > 0)
                                      for index, part in enumerate(node.comparisons)))
    if isinstance(node, cst.IfExp):
        return ((node.test, False), (node.body, True), (node.orelse, True))
    if isinstance(node, cst.Call):
        return ((node.func, False), *((arg.value, False) for arg in node.args))
    if isinstance(node, cst.Attribute):
        return ((node.value, False),)
    if isinstance(node, cst.Subscript):
        result = [(node.value, False)]
        for element in node.slice:
            part = element.slice
            if isinstance(part, cst.Index):
                result.append((part.value, False))
            else:
                result += [(item, False) for item in (part.lower, part.upper, part.step) if item is not None]
        return tuple(result)
    if isinstance(node, (cst.Tuple, cst.List, cst.Set)):
        return tuple((element.value, False) for element in node.elements)
    return ()


class NativeReductions(GeneralReductions, FormulaTransformer):
    def __init__(self, cls_info, cells, providers, formula, reserve_fallback, builtin_bounds,
                 builtin_reductions=frozenset(), enum_symbols=None, formula_kinds=None):
        super().__init__()
        # diag-jp: kinds of the formula's own locals (powers.formula_local_kinds), for names no
        # comprehension in the formula binds; a term `pols_lapse(s) * (1.0 - rho) ** k` with
        # `rho` a double local is then a double and its sum keeps a double state
        bound = {name.value for part in walk(formula) if isinstance(part, cst.CompFor)
                 for name in walk(part.target) if isinstance(name, cst.Name)}
        self.formula_kinds = {name: kind for name, kind in (formula_kinds or {}).items() if name not in bound}
        self.object_loops = []
        self.cls_info, self.cells, self.providers = cls_info, cells, providers
        self.formula, self.reserve_fallback = formula, reserve_fallback
        self.builtin_bounds = builtin_bounds
        self.builtin_reductions = builtin_reductions
        self.enum_symbols = enum_symbols or {}
        self.fallbacks, self.exact_receiver_guards, self.count = [], set(), 0
        self.general_count, self.general_done, self.general_declined = 0, set(), {}
        self.general_sites, self.general_lines = [], None
        self.nested_sites, self.elifs, self.conditional_sites = set(), set(), set()      # W1.8
        self.builtin_names = {name for name in ('bool', 'int', 'float', 'abs', 'len', 'min', 'max',
                                                'sum', 'all', 'any', 'round')
                              if name in builtin_reductions or name not in self.body_facts.stores}
        self.captures = {}
        self.scope_safe = (not has_scope_effects(formula.body.body)
                           and 'self' not in self.body_facts.stores
                           and not {'locals', 'vars', 'eval', 'exec'} & self.body_facts.reads
                           and not {'StopIteration', 'RuntimeError'} & (
                               self.body_facts.stores.keys() | set(cells.params)))
        self.names = Names(formula)
        if isinstance(formula.body, cst.IndentedBlock):
            lines = formula.body.body
            for assignment, consumer in zip(lines, lines[1:]):
                if (not isinstance(assignment, cst.SimpleStatementLine) or len(assignment.body) != 1
                        or not isinstance(assignment.body[0], cst.Assign)
                        or not isinstance(consumer, cst.SimpleStatementLine) or len(consumer.body) != 1
                        or not isinstance(consumer.body[0], (cst.Return, cst.Assign))):
                    continue
                expression = assignment.body[0]
                if len(expression.targets) != 1:
                    continue
                target, rhs = expression.targets[0].target, expression.value
                if isinstance(target, cst.Name):
                    if self.double_expression(rhs):
                        self.captures[id(consumer)] = target.value

    @cached_property
    def body_facts(self):
        return CstFacts(self.formula.body.body)

    def enum_range(self, tree):
        """Use a native range for a declared enum tuple feeding a typed Cell argument."""
        part = tree.for_in
        if (not self.enum_symbols or not self.cells.formula_allows_exact_receiver
                or part.inner_for_in or part.ifs or part.asynchronous
                or not isinstance(part.target, cst.Name) or not isinstance(part.iter, cst.Tuple)
                or not part.iter.elements):
            return None
        operands = arguments(tree.elt)
        if (operands is None or not isinstance(tree.elt.func, cst.Attribute)
                or not isinstance(tree.elt.func.value, cst.Name) or tree.elt.func.value.value != 'self'):
            return None
        cell = self.cls_info.cells.get(tree.elt.func.attr.value)
        if cell is None or cell.called_with_kwargs or cell.body_has_closure or len(operands) != len(cell.params):
            return None
        name = part.target.value
        positions = [index for index, operand in enumerate(operands) if name in CstFacts(operand).reads]
        if len(positions) != 1:
            return None
        index, = positions
        if (not isinstance(operands[index], cst.Name) or operands[index].value != name
                or cell.get_argtype_expr(cell.params[index], c_style=True) != 'long long'):
            return None
        values = []
        owner = None
        for element in part.iter.elements:
            if not isinstance(element, cst.Element):
                return None
            member = element.value
            if (not isinstance(member, cst.Attribute) or not isinstance(member.value, cst.Attribute)
                    or not isinstance(member.value.value, cst.Name) or member.value.value.value != 'self'):
                return None
            ref = member.value.attr.value
            entry = self.enum_symbols.get((self.cls_info.name, ref, member.attr.value))
            if entry is None or owner is not None and ref != owner:
                return None
            owner = ref
            values.append(entry[1])
        start = values[0]
        if all(value == start + index for index, value in enumerate(values)):
            bounds = str(len(values)) if start == 0 else f'{start}, {start + len(values)}'
            return cst.parse_expression(f'range({bounds})')
        return None

    def double_expression(self, node):
        if isinstance(node, cst.Float):
            return True
        if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name)
                and node.value.value == 'self'):
            ref = self.cls_info.refs.get(node.attr.value)
            return ref is not None and ref.get_type_expr(c_style=True) == 'double'
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, (cst.Plus, cst.Minus)):
            return self.double_expression(node.expression)
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.Divide)):
            return self.double_expression(node.left) and self.double_expression(node.right)
        return False

    def reserve_prefix(self):
        return self.names.reserve(f'_mx_reduce_{self.count}', family=True)

    def callable_name(self, func, prefix):
        """The name the lowered reduction uses for its callable, and the line binding it ('' when none).

        diag-mt: a reduction whose callable is the builtin itself (``sum``, ``min``, ... resolved to the
        builtin by the scope pass) reads that name where it is used. Cython reads a builtin name from
        the module's import-time cache, so the name is the same object at every read and needs no
        local: binding it to an ``object`` local took a reference on the one builtin function per
        call, which threads running the same cells contended on (free-threaded reference counts).
        Any other callable keeps the evaluated-once local."""
        if isinstance(func, cst.Name) and func.value in self.builtin_reductions:
            return func.value, ''
        return f'{prefix}_function', f'{prefix}_function: object = {code(func)}'

    def visit_FunctionDef(self, node):
        self.general_sites = [n for n in lexical_nodes(node.body.body) if self.builtin_site(n)]
        self.general_lines = formula_lines(node)
        whole = {id(statement.value) for statement in lexical_nodes(node.body.body)
                 if isinstance(statement, (cst.Return, cst.Assign, cst.AugAssign))
                 and statement.value is not None}
        self.nested_sites = {id(site) for site in self.general_sites if id(site) not in whole}
        if not self.scope_safe:
            for site in self.general_sites:
                self.general_declined.setdefault(id(site), (site, 'the formula rebinds names or '
                                                 'has nested definitions: built-in call'))
        return self.scope_safe

    def leave_FunctionDef(self, before, after):
        if before is not self._formula_root:
            return after          # a nested def is not this formula
        untyped = []
        for site in self.general_sites:
            if id(site) in self.general_done:
                continue
            node, reason = self.general_declined.get(id(site), (site, None))
            untyped.append(untyped_entry(self.general_lines, node, site.func.value, reason or (
                'evaluated only on one branch (after and/or, in a conditional expression or a later '
                'chained comparison): built-in call' if id(site) in self.conditional_sites else
                'not the whole value of a return, assignment or expression statement or an if test '
                '(an elif or while test, an augmented assignment, a one-line suite): built-in call')))
        self.cells.reductions_native = self.general_count
        self.cells.reductions_untyped = untyped
        self.cells.reductions_object = list(self.object_loops)
        return after

    def builtin_site(self, node):
        return (isinstance(node, cst.Call) and isinstance(node.func, cst.Name)
                and node.func.value in self.builtin_reductions and _one_iterable(node))

    def native_truth(self, node, locals_):
        if self.kind(node, locals_) in ('double', 'longlong', 'bint'):
            return True
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Not):
            return self.native_truth(node.expression, locals_)
        if isinstance(node, cst.BooleanOperation):
            return self.native_truth(node.left, locals_) and self.native_truth(node.right, locals_)
        if isinstance(node, cst.Comparison) and all(isinstance(part.operator, (
                cst.Equal, cst.NotEqual, cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual))
                for part in node.comparisons):
            operands = [node.left, *(part.comparator for part in node.comparisons)]
            kinds = [self.kind(value, locals_) for value in operands]
            if not all(kind in ('double', 'longlong', 'bint') for kind in kinds):
                return False
            return 'double' not in kinds or all(kind != 'longlong' or (
                isinstance(value, cst.Integer) and abs(value.evaluated_value) <= (1 << 53))
                for value, kind in zip(operands, kinds))
        return False

    def kind(self, node, locals_):
        if isinstance(node, (cst.Integer, cst.Float, cst.Imaginary, cst.SimpleString)):
            return primitive_kind(literal_value(node))
        if isinstance(node, cst.Name):
            if node.value in ('True', 'False', 'None'):
                return primitive_kind(literal_value(node))
            if node.value in locals_:
                return locals_[node.value]
            if node.value in getattr(self, 'formula_kinds', {}):
                return self.formula_kinds[node.value]
            if node.value in self.cells.params and self.cells.has_typeinfo():
                return {'double': 'double', 'long long': 'longlong', 'bint': 'bint'}.get(
                    self.cells.get_argtype_expr(node.value, c_style=True, formula=True))
            return None
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, (cst.Plus, cst.Minus)):
            try:
                return primitive_kind(literal_value(node))
            except (ValueError, TypeError):
                kind = self.kind(node.expression, locals_)
                if isinstance(node.operator, cst.Plus) or kind == 'double':
                    return kind
                return 'longlong' if kind == 'bint' else None
        if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name) and node.value.value == 'self':
            ref = self.cls_info.refs.get(node.attr.value)
            if ref is not None:
                return {float: 'double', int: 'longlong', bool: 'bint'}.get(ref.type_)
        absolute = arguments(node, 'abs', 1)
        if (absolute and self.builtin_bounds and 'abs' not in locals_
                and 'abs' not in self.cells.params and 'abs' not in self.body_facts.stores
                and self.kind(absolute[0], locals_) == 'double'):
            return 'double'
        if isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute) and isinstance(
                node.func.value, cst.Attribute):
            member = helper_reference(node.func, self.cls_info.refs)
            if getattr(member, 'module_math', None) and len(node.args) == 1:
                if member.scalar_result == 'int64':      # floor/ceil: C integer of a typed argument
                    argument = self.kind(node.args[0].value, locals_)
                    return 'longlong' if argument in ('double', 'longlong', 'bint') else None
                return {'float64': 'double', 'bool': 'bint'}.get(member.scalar_result)
        if (arguments(node) is not None and isinstance(node.func, cst.Attribute)
                and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'):
            cell = self.cls_info.cells.get(node.func.attr.value)
            if cell is not None and cell.has_typeinfo() and not cell.called_with_kwargs and not cell.body_has_closure:
                return {'double': 'double', 'long long': 'longlong', 'bint': 'bint'}.get(cell.get_rettype_expr(c_style=True))
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.Divide)):
            left, right = self.kind(node.left, locals_), self.kind(node.right, locals_)
            if (left in ('double', 'longlong', 'bint')
                    and right in ('double', 'longlong', 'bint') and 'double' in (left, right)):
                return 'double'
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Power):
            # diag-jp: a double raised to a C integer is a double (Python's float ** int is a
            # float or raises), so a sum of such terms keeps a double state instead of objects
            if (self.kind(node.left, locals_) == 'double'
                    and self.bound_kind(node.right, locals_) in ('longlong', 'bint')):
                return 'double'
        return None

    def bound_kind(self, node, locals_):
        """The kind of a ``range`` bound: as ``kind``, plus C integer arithmetic (``+ - *`` of C
        integers) and the builtin ``min``/``max`` of C integers, which are C integers (diag-jp:
        Term_JP_S ``range(max(0, t - reinstate_window_m()), t)`` stayed a Python range object, so
        its loop variable and every term built from it were objects)."""
        kind = self.kind(node, locals_)
        if kind is not None:
            return kind
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, (cst.Add, cst.Subtract, cst.Multiply)):
            if (self.bound_kind(node.left, locals_) in ('longlong', 'bint')
                    and self.bound_kind(node.right, locals_) in ('longlong', 'bint')):
                return 'longlong'
            return None
        args = arguments(node, 'int')
        if (args and len(args) == 1 and 'int' not in locals_ and 'int' not in self.cells.params
                and 'int' not in self.body_facts.stores
                and self.bound_kind(args[0], locals_) in ('double', 'longlong', 'bint')):
            return 'longlong'       # int() of a number; the bound local's conversion checks the range
        for name in ('min', 'max'):
            args = arguments(node, name)
            if (args and len(args) >= 2 and self.builtin_bounds and name not in locals_
                    and name not in self.cells.params and name not in self.body_facts.stores
                    and all(self.bound_kind(arg, locals_) in ('longlong', 'bint') for arg in args)):
                return 'longlong'
        return None

    def cold_fallback(self, source, generator, capture, function, capture_type='_mx_cy.double'):
        original = f'{function}({source})'
        if not self.cells.has_formula_def or not self.scope_safe:
            return original
        names = CstFacts(generator).reads
        targets = {part.target.value for part in _parts(generator)}
        stores = self.body_facts.stores
        if ((names - targets) & stores.keys()) - {capture}:
            return original
        if capture is not None and stores[capture] != 1:
            return original
        params = self.formula.params
        parameters = [*params.posonly_params, *params.params, *params.kwonly_params]
        parameters += [param for param in (params.star_arg, params.star_kwarg) if isinstance(param, cst.Param)]
        param_names = {param.name.value for param in parameters}
        if targets & param_names or not (names & param_names) <= {'self', *self.cells.params}:
            return original
        args = []
        for name in self.cells.params:
            if name in names:
                kind = self.cells.get_argtype_expr(name, c_style=True, formula=True)
                if kind not in ('double', 'long long', 'bint', 'object'):
                    return original
                args.append((name, self.cells.get_argtype_expr(name, formula=True)))
        if capture is not None:
            args.append((capture, capture_type))
        helper = self.reserve_fallback()
        formals = ', '.join([f'self: {self.cls_info.name}', f'{function}: object']
                            + [f'{name}: {kind}' for name, kind in args])
        self.fallbacks.append(cst.parse_statement(
            f'\n\n@_mx_cy.cfunc\ndef {helper}({formals}) -> object:\n    return {original}\n'))
        actuals = ', '.join(['self', function] + [name for name, _ in args])
        return f'{helper}({actuals})'

    def single_range(self, after, func, provider, operation, source, tree):
        parts = tuple(_parts(tree))
        if len(parts) != 1 or not self.cells.formula_allows_exact_receiver:
            return after
        part, = parts
        if not isinstance(part.target, cst.Name) or part.ifs or part.asynchronous is not None:
            return after
        index = part.target.value
        if index in {'self', 'range', 'abs', 'max', *self.cells.params}:
            return after
        alias, prefix = module_name(provider), self.reserve_prefix()
        counter, guard = f'{prefix}_index', f'{prefix}_native'
        function, binding = self.callable_name(func, prefix)
        predicate = tree.elt
        args = arguments(part.iter, 'range', 1)
        if not (args is not None and arguments(args[0], count=0) is not None
                and self.kind(args[0], {}) == 'longlong'
                and self.cells.get_rettype_expr(c_style=True) == 'bint'):
            return after
        iterator = f'range({prefix}_limit)'
        if isinstance(predicate, cst.Comparison):
            if len(predicate.comparisons) != 1:
                return after
            comparison = predicate.comparisons[0]
            absolute = arguments(predicate.left, 'abs', 1)
            if not (isinstance(comparison.operator, (cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual))
                    and absolute is not None and self.kind(comparison.comparator, {}) == 'double'
                    and isinstance(comparison.comparator, (cst.Float, cst.Attribute))):
                return after
            value = self.indexed_cell(absolute[0], index, 'double', counter)
            if value is None:
                return after
            operator = {cst.LessThan: '<', cst.LessThanEqual: '<=', cst.GreaterThan: '>', cst.GreaterThanEqual: '>='}[type(comparison.operator)]
            test = f'{alias}.f64_abs({value}) {operator} {code(comparison.comparator)}'
        else:
            test = self.indexed_cell(predicate, index, 'bint', counter)
            if test is None:
                return after
        loop = f"""    {prefix}_limit: _mx_cy.longlong = {code(args[0])}
    try:
        for {counter} in {iterator}:
            if not ({test}):
                return False
    except StopIteration as {prefix}_error:
        raise RuntimeError("generator raised StopIteration") from {prefix}_error
    return True"""
        count = len(self.fallbacks)
        fallback = self.cold_fallback(source, tree, None, function)
        if len(self.fallbacks) == count:
            return after
        self.providers.add(provider)
        self.count += 1
        return cst.FlattenSentinel(cst.parse_module(f"""
{binding}
{guard}: _mx_cy.bint = {alias}.{operation}_accepts({function})
if {exact_receiver(self.cls_info.name)} and {guard}:
    {counter}: _mx_cy.longlong
{loop}
return {fallback}
""").body)

    def indexed_cell(self, node, index, kind, counter):
        args = arguments(node, count=1)
        if not (args is not None and isinstance(args[0], cst.Name) and args[0].value == index
                and self.kind(node, {index: 'longlong'}) == kind):
            return None
        cell = self.cls_info.cells[node.func.attr.value]
        if len(cell.params) != 1 or cell.get_argtype_expr(cell.params[0], c_style=True) != 'long long':
            return None
        return f'self.{node.func.attr.value}({counter})'

    def leave_SimpleStatementLine(self, before, after):
        result = self._specialized(before, after)
        if result is not after:
            if len(before.body) == 1 and isinstance(getattr(before.body[0], 'value', None), cst.Call):
                self.general_done.add(id(before.body[0].value))
            return result
        result = self.general(before, after)
        if result is not after:
            return result
        return self.nested(before, after)

    def _specialized(self, before, after):
        if len(before.body) != 1:
            return after
        statement = before.body[0]
        assignment = isinstance(statement, cst.Assign)
        if not (isinstance(statement, cst.Return) or assignment and len(statement.targets) == 1
                and isinstance(statement.targets[0].target, cst.Name)):
            return after
        value = statement.value
        args = arguments(value)
        if not args or not isinstance(args[0], cst.GeneratorExp) or not self.builtin_bounds:
            return after
        func, tree = value.func, args[0]
        source = code(tree)
        builtin = func.value if isinstance(func, cst.Name) and func.value in self.builtin_reductions else None
        if len(args) != 1:
            return after
        if builtin == 'all':
            return after if assignment else self.single_range(
                after, func, BUILTIN_PROVIDER, 'all', source, tree)
        if builtin != 'sum':
            return after
        provider = STREAMS_PROVIDER
        if assignment:
            return after
        if self.cells.get_rettype_expr(c_style=True) != 'double':
            # Ordinary sum keeps integer/mixed/object semantics. Only an existing
            # double return boundary may normalize empty integer zero to double.
            return after
        original_tree = tree
        enum_range = self.enum_range(tree)
        if enum_range is not None:
            tree = tree.with_changes(for_in=tree.for_in.with_changes(iter=enum_range))
        capture = self.captures.get(id(before))
        if capture == 'range':
            return after
        generator_facts = CstFacts(tree)
        if capture is not None and capture not in generator_facts.reads:
            capture = None
        local_types = {capture: 'double'} if capture is not None else {}
        for part in _parts(tree):
            if part.asynchronous is not None or not isinstance(part.target, cst.Name):
                return after
            name, iterator = part.target.value, part.iter
            if name == capture:
                return after
            if name not in local_types and name in CstFacts(iterator).reads:
                return after
            bounds = arguments(iterator, 'range')
            if (bounds is not None and 'range' not in local_types and 1 <= len(bounds) <= 3
                    and all(self.kind(arg, local_types) in ('longlong', 'bint') for arg in bounds)
                    and safe_range_increment(iterator)):
                local_types[name] = 'longlong'
            elif (isinstance(iterator, cst.Tuple)
                    and all(isinstance(element, cst.Element) for element in iterator.elements)):
                kind = (self.kind(iterator.elements[0].value, local_types)
                        if len(iterator.elements) == 1 else None)
                # Literal domains need no sampled category schema or size limit.
                # Keep literal objects intact: even a homogeneous numeric domain
                # can flow into an object-argument Cell. No new loop coercion.
                if kind is None:
                    try:
                        values = [literal_value(element.value) for element in iterator.elements]
                    except (ValueError, TypeError):
                        return after
                    if not all(type(value) in (int, float, bool, str, type(None)) for value in values):
                        return after
                    kind = 'object'
                local_types[name] = kind
            else:
                return after
            if not all(self.native_truth(condition.test, local_types) for condition in part.ifs):
                return after
        if self.kind(tree.elt, local_types) != 'double':
            return after
        alias, prefix = module_name(provider), self.reserve_prefix()
        count = len(self.fallbacks)
        function, binding = self.callable_name(func, prefix)
        fallback = self.cold_fallback(source, original_tree, capture, function)
        if len(self.fallbacks) == count:
            return after
        # Iterator/capture names must not alias the accumulator or call-layout
        # temporaries (e.g. an authored iterator named `state`).
        for role in ('function', 'state', 'iterable', 'error', 'bound_0', 'bound_1', 'bound_2'):
            self.names.reserve(f'{prefix}_{role}')
        renamed = {name: self.names.reserve(f'{prefix}_{name}') for name in local_types}
        contexts = generator_facts.contexts

        class Rename(cst.CSTTransformer):
            def leave_Name(self, before, after):
                if before.value in renamed and before in contexts:
                    return after.with_changes(value=renamed[before.value])
                return after

        tree = tree.visit(Rename())
        declarations = [f'{prefix}_state: {alias}.F64Sum']
        declarations += [f'{renamed[name]}: _mx_cy.{kind}' for name, kind in local_types.items()]
        guard = f'{alias}.f64_accepts({function})'
        if enum_range is not None:
            guard += f' and {exact_receiver(self.cls_info.name)}'
        lines = declarations + [binding] * bool(binding) + [f'if {guard}:',
                               f'    {alias}.f64_init(_mx_cy.address({prefix}_state))']
        if capture is not None:
            lines.append(f'    {renamed[capture]} = {capture}')
        parts = tuple(_parts(tree))
        outer = parts[0]
        bounds = arguments(outer.iter, 'range')
        if bounds is not None:
            bound_names = [f'{prefix}_bound_{index}' for index in range(len(bounds))]
            lines.extend(f'    {name}: _mx_cy.longlong = {code(value)}'
                         for name, value in zip(bound_names, bounds))
            outer_iterator = f'range({", ".join(bound_names)})'
        elif len(outer.iter.elements) == 1:
            # A generator acquires its first iterable before entering its body.
            # Keep that evaluation outside the PEP479 boundary as well.
            lines.append(f'    {outer.target.value} = ({code(outer.iter.elements[0].value)})')
            outer_iterator = None
        else:
            lines.append(f'    {prefix}_iterable: object = {code(outer.iter)}')
            outer_iterator = f'{prefix}_iterable'
        lines.append('    try:')
        indent = '        '
        for index, part in enumerate(parts):
            if index == 0 and outer_iterator is None:
                pass
            elif index == 0:
                lines.append(indent + f'for {part.target.value} in {outer_iterator}:')
                indent += '    '
            elif isinstance(part.iter, cst.Tuple) and len(part.iter.elements) == 1:
                lines.append(indent + f'{part.target.value} = ({code(part.iter.elements[0].value)})')
            else:
                lines.append(indent + f'for {part.target.value} in {code(part.iter)}:')
                indent += '    '
            for condition in part.ifs:
                lines.append(indent + f'if ({code(condition.test)}):')
                indent += '    '
        result = code(statement.targets[0].target) + ' = ' if assignment else 'return '
        lines += [indent + f'{alias}.f64_add(_mx_cy.address({prefix}_state), {code(tree.elt)})',
                  f'    except StopIteration as {prefix}_error:',
                  f'        raise RuntimeError("generator raised StopIteration") from {prefix}_error',
                  f'    {result}{alias}.f64_result(_mx_cy.address({prefix}_state))',
                  'else:', f'    {result}{fallback}']
        self.providers.add(provider)
        self.count += 1
        return cst.FlattenSentinel(cst.parse_module('\n'.join(lines) + '\n').body)
