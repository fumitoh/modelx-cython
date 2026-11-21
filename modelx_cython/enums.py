"""Map enum definitions (References whose traced value is an IntEnum class) to private Cython constants."""
import libcst as cst
from libcst.metadata import ExpressionContext

from .scopes import CstFacts, FormulaTransformer, Names, lexical_nodes
from .syntax import walk
from .consts import MX_SYS_MOD


def storage(name):
    return '_mx_definition_' + name


def prepare(module):
    module.code_enum_symbols = {}
    identifiers = Names(module.visitor.wrapper.module)
    for cls in module.classes.values():
        # An enum definition is a Reference whose traced value is an enum-safe IntEnum class
        # (selected by type, no declaration). It is fixed in the compiled model: read-only, and its
        # members are C constants. A Reference still holding the former declaration marker
        # (modelx_helpers' EnumDefinitions) is an ordinary Reference and is ignored here.
        names = tuple(name for name, ref in cls.refs.items()
                      if name not in getattr(cls, 'spaces', ())
                      and getattr(ref, 'enum', None) and getattr(ref, 'enum_safe', False))
        cls.readonly_refs = names
        for name in names:
            ref = cls.refs[name]
            if storage(name) in cls.refs:
                raise ValueError(f'{cls.fqname}: reserved enum storage name {storage(name)}')
            for member, value in ref.enum[2].items():
                if -(1 << 31) <= value < (1 << 31):
                    symbol = identifiers.reserve(f'_mx_enum_const_{len(module.code_enum_symbols)}')
                    module.code_enum_symbols[cls.name, name, member] = symbol, value


def declarations(module):
    if not getattr(module, 'code_enum_symbols', {}):
        return ''
    return 'cdef enum:\n' + ''.join(f'    {symbol} = {value}\n'
        for symbol,value in module.code_enum_symbols.values())


class Constants(FormulaTransformer):
    def __init__(self, cls, cell, symbols):
        super().__init__()
        self.cls, self.cell, self.symbols = cls, cell, symbols
        self.locals = set()
        self.count = 0
        self.comparisons = 0
        self.untyped = []           # expression forms left as Python (admission report)
        self.rebinds_self = False
        self.member_fields = 0
        self.memberships = 0        # ``in`` / ``not in`` over member tuples lowered
        self.object_memberships = []    # W1.8: ``in`` with an object left, as a chain
        self.lines = None
        self.enum_locals = {}       # D83: local -> enum Ref of a member-valued C integer local

    def visit_FunctionDef(self, node):
        from .reductions import formula_lines
        self.lines = formula_lines(node)
        assignments = {}
        invalid = set()
        nodes = list(lexical_nodes(node.body.body))
        facts = CstFacts(node.body.body)
        self.rebinds_self = 'self' in facts.bindings(nodes)
        if self.rebinds_self:
            return False
        for part in nodes:
            if isinstance(part, cst.Assign):
                for target in part.targets:
                    if isinstance(target.target, cst.Name):
                        assignments.setdefault(target.target.value, []).append(part.value)
            elif isinstance(part, (cst.AugAssign, cst.AnnAssign, cst.NamedExpr, cst.For, cst.CompFor)):
                invalid.update(n.value for n in walk(part.target) if isinstance(n, cst.Name))
            elif not isinstance(part, (cst.Name, cst.Assign)):
                invalid.update(facts.bindings((part,)))
        self.assignments, self.invalid = assignments, invalid     # D98: provenance of locals
        self.loop_members = self._member_loops(nodes, facts, assignments, invalid)
        self.locals.update(self.loop_members)
        for name, values in assignments.items():
            if name not in invalid and all(self.integer_value(v) for v in values):
                self.locals.add(name)
        if node is self._formula_root or self._formula_root is None:
            self.enum_locals = self.member_locals(node, nodes, assignments, invalid)
            self.locals.update(self.enum_locals)

    def integer_value(self, node):
        """A value bound to a C-integer local: a call of a compiled Cell returning ``long long`` or
        ``bint`` (no keywords, no closure), a Reference held in a C ``long long`` field, or (diag-scale
        P2b) a conditional expression whose two branches are such values. The test of a conditional
        is evaluated once, as written, so the local holds exactly what the chosen branch gives."""
        if isinstance(node, cst.IfExp):
            return self.integer_value(node.body) and self.integer_value(node.orelse)
        return ((isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute)
                 and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'
                 and node.func.attr.value in self.cls.cells
                 and not self.cls.cells[node.func.attr.value].called_with_kwargs
                 and not self.cls.cells[node.func.attr.value].body_has_closure
                 and self.cls.cells[node.func.attr.value].get_rettype_expr(c_style=True)
                 in ('long long', 'bint')) or self.integer_ref(node))

    def member_ref(self, node):
        """The Ref of an admitted enum whose member ``node`` is (``self.<Ref>.<member>``, or a
        conditional expression of members of one Ref in both branches), else None."""
        if isinstance(node, cst.IfExp):
            body, orelse = self.member_ref(node.body), self.member_ref(node.orelse)
            return body if body is not None and body == orelse else None
        if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute)
                and isinstance(node.value.value, cst.Name) and node.value.value.value == 'self'
                and (self.cls.name, node.value.attr.value, node.attr.value) in self.symbols):
            return node.value.attr.value
        return None

    def member_locals(self, node, nodes, assignments, invalid):
        """D83: locals held as C integers bound to the enum module constants.

        A local qualifies when every binding is a plain assignment of a member of one admitted enum
        (or a conditional expression of such members), and every load of it is in a C-integer
        position of this formula: one side of ``==`` / ``!=`` with a member or an integer, the left
        of ``in`` / ``not in`` over a literal tuple of members, or a positional argument of a
        compiled Cell whose parameter is a C integer. An IntEnum member compares, hashes and
        converts as its value there, so the result is the one the member object gives. Any other
        load (returned, formatted, an attribute of it, passed to Python, a table key, a nested
        scope) keeps the local an object: no partial typing."""
        candidates = {}
        for name, values in assignments.items():
            if name in invalid or name in self.locals or name in self.cell.params:
                continue
            refs = {self.member_ref(value) for value in values}
            if len(refs) == 1 and None not in refs:
                candidates[name] = refs.pop()
        if not candidates:
            return {}
        targets = {id(target.target) for part in nodes if isinstance(part, cst.Assign)
                   for target in part.targets}
        every = {}          # name -> ids of all its Name nodes that are not assignment targets
        for item in walk(node.body):
            if isinstance(item, cst.Name) and item.value in candidates and id(item) not in targets:
                every.setdefault(item.value, set()).add(id(item))
        integer = {}        # name -> ids of loads in C-integer positions (this formula's scope)

        def ok(name_node, ref):
            integer.setdefault(name_node.value, set()).add(id(name_node))

        for part in nodes:
            if isinstance(part, cst.Comparison) and len(part.comparisons) == 1:
                comp = part.comparisons[0]
                left, right = part.left, comp.comparator
                if isinstance(comp.operator, (cst.Equal, cst.NotEqual)):
                    for one, other in ((left, right), (right, left)):
                        if (isinstance(one, cst.Name) and one.value in candidates
                                and (self.member_ref(other) == candidates[one.value]
                                     and not isinstance(other, cst.IfExp) or self.integer(other))):
                            ok(one, candidates[one.value])
                elif (isinstance(comp.operator, (cst.In, cst.NotIn)) and isinstance(left, cst.Name)
                        and left.value in candidates and isinstance(right, cst.Tuple) and right.elements
                        and all(isinstance(e, cst.Element) and not isinstance(e.value, cst.IfExp)
                                and self.member_ref(e.value) == candidates[left.value]
                                for e in right.elements)):
                    ok(left, candidates[left.value])
            elif (isinstance(part, cst.Call) and isinstance(part.func, cst.Attribute)
                    and isinstance(part.func.value, cst.Name) and part.func.value.value == 'self'
                    and not any(a.keyword or a.star for a in part.args)):
                cell = self.cls.cells.get(part.func.attr.value)
                if cell is None or cell.called_with_kwargs or cell.body_has_closure:
                    continue
                for arg, param in zip(part.args, cell.params):
                    if (isinstance(arg.value, cst.Name) and arg.value.value in candidates
                            and cell.get_argtype_expr(param, c_style=True) in ('long long', 'bint')):
                        ok(arg.value, candidates[arg.value.value])
        return {name: ref for name, ref in candidates.items()
                if every.get(name) and every[name] == integer.get(name)}

    def constant(self, node):
        """A member expression (see member_ref) with each member replaced by its constant."""
        if isinstance(node, cst.IfExp):
            return node.with_changes(body=self.constant(node.body), orelse=self.constant(node.orelse))
        self.count += 1
        return cst.Name(self.symbols[self.cls.name, node.value.attr.value, node.attr.value][0],
                        lpar=node.lpar, rpar=node.rpar)

    def leave_Assign(self, before, after):
        if (len(after.targets) == 1 and isinstance(after.targets[0].target, cst.Name)
                and after.targets[0].target.value in self.enum_locals
                and self.member_ref(after.value) is not None):
            return after.with_changes(value=self.constant(after.value))
        return after

    def _member_loops(self, nodes, facts, assignments, invalid):
        """diag-kr: targets of ``for x in (E.A, E.B, ...)`` over a literal tuple of members.

        Such a target holds only members, which compare and key as their integer values
        (IntEnum), so it is a C ``long long`` when every binding of it is such a loop and every
        read of it is an ``==``/``!=``/``in``/``not in`` operand or an argument of a cell
        parameter typed ``long long``; the tuple's elements become the member constants. A read
        anywhere else (``x is E.A``, an object parameter, a container, ``x.name``) keeps it an
        object, so a member is never replaced by a bare int where its identity could show.
        """
        loops = {}
        for part in nodes:
            if isinstance(part, cst.For) and isinstance(part.target, cst.Name):
                if part.orelse is None and self.member_tuple_constant(part.iter):
                    loops[part.target.value] = loops.get(part.target.value, 0) + 1
        safe = set()
        for part in nodes:
            if isinstance(part, cst.Comparison) and all(
                    isinstance(item.operator, (cst.Equal, cst.NotEqual, cst.In, cst.NotIn))
                    for item in part.comparisons):
                safe.update(id(operand) for operand in (part.left, *(c.comparator for c in part.comparisons))
                            if isinstance(operand, cst.Name))
            elif (isinstance(part, cst.Call) and isinstance(part.func, cst.Attribute)
                    and isinstance(part.func.value, cst.Name) and part.func.value.value == 'self'
                    and not any(a.keyword or a.star for a in part.args)):
                cell = self.cls.cells.get(part.func.attr.value)
                if cell is None or cell.called_with_kwargs or cell.body_has_closure:
                    continue
                for argument, param in zip(part.args, cell.params):
                    if (isinstance(argument.value, cst.Name)
                            and cell.get_argtype_expr(param, c_style=True) in ('long long', 'bint')):
                        safe.add(id(argument.value))
        found = set()
        for name, count in loops.items():
            loads = [node for node, context in facts.contexts.items()
                     if isinstance(node, cst.Name) and node.value == name and context is ExpressionContext.LOAD]
            if (name not in assignments and facts.stores[name] == count and name != 'self'
                    and all(id(node) in safe for node in loads)):
                found.add(name)
        invalid.difference_update(found)
        return found

    def member_tuple_constant(self, node):
        """A non-empty literal tuple whose every element is ``self.<Ref>.<member>`` of a declared
        enum (read only, no count)."""
        return (isinstance(node, cst.Tuple) and bool(node.elements) and all(
            isinstance(element, cst.Element) and isinstance(element.value, cst.Attribute)
            and isinstance(element.value.value, cst.Attribute)
            and isinstance(element.value.value.value, cst.Name) and element.value.value.value.value == 'self'
            and self.symbols.get((self.cls.name, element.value.value.attr.value, element.value.attr.value))
            for element in node.elements))

    def leave_For(self, before, after):
        if (isinstance(before.target, cst.Name) and before.target.value in getattr(self, 'loop_members', ())
                and self.member_tuple_constant(before.iter)):
            members = self.members(after.iter)
            if members is not None:
                return after.with_changes(iter=members)
        return after

    def integer_ref(self, node):
        """``self.<ref>`` of a Reference stored in a C ``long long`` field (an integer or an
        IntEnum member held by the Space, D31): read from the field, so it compares as a C
        integer with a member constant."""
        if not (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name)
                and node.value.value == 'self'):
            return False
        name = node.attr.value
        ref = getattr(self.cls, 'refs', {}).get(name)
        if ref is None or name in getattr(self.cls, 'spaces', ()) or name in getattr(self.cls, 'readonly_refs', ()):
            return False
        try:
            return ref.get_type_expr(c_style=True) == 'long long'
        except Exception:       # a Reference without a scalar type expression
            return False

    def leave_FunctionDef(self, before, after):
        if before is not self._formula_root:
            return after          # a nested def: its own receiver, left as written
        self.cell.enum_untyped = list(self.untyped)
        self.cell.enum_memberships = self.memberships
        self.cell.enum_object = list(self.object_memberships)
        if self.lines is None or self.rebinds_self:
            return after
        self.cell.enum_locals = sorted(self.enum_locals)
        if self.enum_locals:
            # D83: declared C integers (the bindings are module constants); after the docstring
            statements = list(after.body.body)
            start = int(bool(statements) and isinstance(statements[0], cst.SimpleStatementLine)
                        and len(statements[0].body) == 1 and isinstance(statements[0].body[0], cst.Expr)
                        and isinstance(statements[0].body[0].value, (cst.SimpleString, cst.ConcatenatedString)))
            declared = [cst.parse_statement(f'{name}: _mx_cy.longlong\n') for name in sorted(self.enum_locals)]
            after = after.with_changes(body=after.body.with_changes(
                body=(*statements[:start], *declared, *statements[start:])))
        if getattr(self, 'loop_members', None):
            # diag-kr: the member-loop targets are C integers (their tuples hold member constants)
            declarations = tuple(cst.SimpleStatementLine((cst.AnnAssign(
                cst.Name(name), cst.Annotation(cst.Attribute(cst.Name('_mx_cy'), cst.Name('longlong')))),))
                for name in sorted(self.loop_members))
            doc_end = 1 if before.get_docstring() is not None else 0
            body = after.body
            after = after.with_changes(body=body.with_changes(
                body=(*body.body[:doc_end], *declarations, *body.body[doc_end:])))
        # W1.4: a member left as an object (an object position) is read from the definition's
        # C field, not through the Space property: self._mx_definition_<Ref>.<member>.
        fields = _MemberFields(self.cls.name, self.symbols)
        after = after.with_changes(body=after.body.visit(fields))
        self.member_fields = fields.count
        return after

    def member(self, node):
        """``self.<Ref>.<member>`` of an admitted enum as its module constant, else ``node``.

        diag-scale P2a: a conditional expression (``E.A if c else E.B``, nested or with one branch a
        C integer) gets each member branch replaced, but only when both branches are then C integers
        (a member constant or a value ``integer`` accepts); otherwise it is left whole, so a branch
        that is a Python object keeps the object path. Only the branches change: the test is kept and
        evaluated once, and a member converts and compares as its value (``enum_safe``)."""
        if isinstance(node, cst.IfExp):
            count = self.count
            body, orelse = self.member(node.body), self.member(node.orelse)
            if ((body is not node.body or orelse is not node.orelse)
                    and self._c_integer_branch(node.body, body) and self._c_integer_branch(node.orelse, orelse)):
                return node.with_changes(body=body, orelse=orelse)
            self.count = count
            return node
        if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute)
                and isinstance(node.value.value, cst.Name) and node.value.value.value == 'self'):
            entry = self.symbols.get((self.cls.name, node.value.attr.value, node.attr.value))
            if entry:
                self.count += 1
                return cst.Name(entry[0],lpar=node.lpar,rpar=node.rpar)
        return node

    def _c_integer_branch(self, before, after):
        """A branch of a conditional is a C integer after member replacement: replaced (a member
        constant, or a nested conditional that was replaced whole), or a value ``integer`` accepts."""
        return after is not before or self.integer(before)

    def integer(self, node):
        if isinstance(node, cst.Integer):
            return node.evaluated_value < (1 << 63)
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
            return self.integer(node.expression)
        if isinstance(node, cst.BinaryOperation) and isinstance(
                node.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.FloorDivide, cst.Modulo)):
            return self.integer(node.left) and self.integer(node.right)
        if self.integer_ref(node):
            return True
        if isinstance(node, cst.Name):
            return node.value in self.locals or (node.value in self.cell.params and
                self.cell.get_argtype_expr(node.value, c_style=True, formula=True) in ('long long','bint'))
        if (isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute)
                and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'):
            cell = self.cls.cells.get(node.func.attr.value)
            return (cell is not None and not cell.called_with_kwargs and not cell.body_has_closure
                    and cell.get_rettype_expr(c_style=True) in ('long long','bint'))
        return False

    def members(self, node):
        """Every element of a literal tuple as a build-time constant, else None.

        ``x in (E.A, E.B)`` on a typed integer ``x`` then compiles to a chain of
        C ``==`` with ``x`` evaluated once (Cython's literal-tuple membership).
        IntEnum members compare as their int values (``enum_safe``), so the
        result is the one the tuple of member objects gives.
        """
        if not isinstance(node, cst.Tuple) or not node.elements:
            return None
        count, elements = self.count, []
        for element in node.elements:
            member = self.member(element.value) if isinstance(element, cst.Element) else None
            if member is None or member is element.value:
                self.count = count
                return None
            elements.append(element.with_changes(value=member))
        return node.with_changes(elements=elements)

    def leave_Comparison(self, before, after):
        if len(after.comparisons) != 1:
            return after
        comp = after.comparisons[0]
        identity = self.member_identity(after, comp)
        if identity is not None:
            return identity
        if self.integer(after.left):
            if isinstance(comp.operator, (cst.Equal,cst.NotEqual)):
                member = self.member(comp.comparator)
                if member is not comp.comparator:
                    self.comparisons += 1
                return after.with_changes(comparisons=(comp.with_changes(comparator=member),))
            if isinstance(comp.operator, (cst.In, cst.NotIn)):
                members = self.members(comp.comparator)
                if members is None:
                    if self.member_tuple(comp.comparator):
                        self.decline(before, 'a tuple element is not a member of a declared enum')
                    return after
                self.comparisons += len(members.elements)
                self.memberships += 1
                return after.with_changes(comparisons=(comp.with_changes(comparator=members),))
        elif isinstance(comp.operator, (cst.Equal, cst.NotEqual)) and self.integer(comp.comparator):
            # ``E.A == x``: a member on the left has no evaluation to reorder.
            member = self.member(after.left)
            if member is not after.left:
                self.comparisons += 1
                return after.with_changes(left=member)
        elif isinstance(comp.operator, (cst.In, cst.NotIn)):
            chain = self.object_membership(before, after, comp)
            if chain is not None:
                return chain
            if self.member_tuple(comp.comparator):
                self.decline(before, 'the left operand is not a typed integer (an object or untraced '
                             'value) and is not a name, or an element is not a constant or a member')
        return after

    def member_identity(self, after, comp):
        """D98: ``x is E.M`` / ``x is not E.M`` (either order) where ``x`` is typed as a C integer
        and ``E.M`` a member of an admitted enum: lowered exactly as ``==`` / ``!=`` with the
        member's constant. ``x`` holds the integer value of an enum-valued Cell, parameter or
        local, which live holds as the member itself; members are singletons, so live's
        identity test is true exactly when the values are equal. Without this the compiled
        test compared an int with the member object and was always false. Any other identity
        test (no typed integer, or no admitted member) is left as it is, so Python decides it."""
        if not isinstance(comp.operator, (cst.Is, cst.IsNot)):
            return None
        operator = (cst.Equal if isinstance(comp.operator, cst.Is) else cst.NotEqual)(
            whitespace_before=comp.operator.whitespace_before,
            whitespace_after=comp.operator.whitespace_after)
        count = self.count
        if self.integer(after.left) and self.same_enum(after.left, comp.comparator):
            member = self.member(comp.comparator)
            if member is not comp.comparator:
                self.comparisons += 1
                return after.with_changes(comparisons=(comp.with_changes(operator=operator, comparator=member),))
        elif self.integer(comp.comparator) and self.same_enum(comp.comparator, after.left):
            member = self.member(after.left)
            if member is not after.left:
                self.comparisons += 1
                return after.with_changes(left=member, comparisons=(comp.with_changes(operator=operator),))
        self.count = count
        return None

    def same_enum(self, value, member):
        """D98: ``value`` holds a value of the enum class whose member ``member`` is: a call of a
        compiled Cell whose every traced return value was a member of that class, a parameter
        traced as that class, or a local bound only to such values (or a D83 member local of it).
        An integer literal, arithmetic or a plain int parameter is not: ``5 is E.A`` is false in
        Python while ``5 == E.A`` may be true."""
        ref = self.member_ref(member) if not isinstance(member, cst.IfExp) else None
        target = _enum_key(getattr(self.cls, 'refs', {}).get(ref)) if ref else None
        return target is not None and self.enum_of(value, set()) == target

    def enum_of(self, node, seen):
        if (isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute)
                and isinstance(node.func.value, cst.Name) and node.func.value.value == 'self'):
            cell = self.cls.cells.get(node.func.attr.value)
            return _traced_enum(cell, None) if cell is not None else None
        if not isinstance(node, cst.Name):
            return None
        name = node.value
        if name in self.enum_locals:
            return _enum_key(getattr(self.cls, 'refs', {}).get(self.enum_locals[name]))
        if name in self.cell.params:
            return _traced_enum(self.cell, name)
        values = getattr(self, 'assignments', {}).get(name)
        if not values or name in getattr(self, 'invalid', ()) or name in seen:
            return None
        kinds = {self.enum_of(v, seen | {name}) for v in values}
        return kinds.pop() if len(kinds) == 1 else None

    # W1.8: ``x in (a, b, ...)`` with an object ``x``, exactly as tuple membership
    _MEMBERSHIP_LIMIT = 8

    def plain(self, node):
        """Read without running code: a literal, a local or parameter name, ``self.<a>.<b>``."""
        if isinstance(node, (cst.Integer, cst.Float, cst.SimpleString)):
            return True
        if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
            return isinstance(node.expression, (cst.Integer, cst.Float))
        if isinstance(node, cst.Name):
            return node.value not in ('self',)
        return (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute)
                and isinstance(node.value.value, cst.Name) and node.value.value.value == 'self'
                and any(key[:2] == (self.cls.name, node.value.attr.value) for key in self.symbols))

    def object_membership(self, before, after, comp):
        """``x in (a, b)`` -> ``(x is a or not not (a == x) or x is b or ...)``.

        Tuple membership compares each element in order with ``PyObject_RichCompareBool
        (element, x, Py_EQ)``: identity first, then ``element == x`` and its truth. The
        chain makes the same calls in the same order and stops at the same element; it
        reads ``x`` and the elements again, so it applies only where those reads run no
        code (a name, a literal, a member field). The result is a bool, as ``in`` gives.
        """
        tuple_ = comp.comparator
        if (not isinstance(tuple_, cst.Tuple) or not tuple_.elements
                or len(tuple_.elements) > self._MEMBERSHIP_LIMIT
                or not all(isinstance(element, cst.Element) for element in tuple_.elements)
                or not isinstance(after.left, cst.Name) or after.left.value in self.locals
                or not all(self.plain(element.value) for element in tuple_.elements)):
            return None
        left = after.left.value
        parts = []
        for element in tuple_.elements:
            item = cst.Module([]).code_for_node(element.value.with_changes(lpar=(), rpar=())
                                                if hasattr(element.value, 'lpar') else element.value)
            if isinstance(element.value, (cst.Integer, cst.Float, cst.SimpleString, cst.UnaryOperation)):
                # the identity test of a literal is implied by its own reflexive ==
                parts.append(f'not not (({item}) == {left})')
            else:
                parts.append(f'{left} is {item} or not not ({item} == {left})')
        text = ' or '.join(parts)
        if isinstance(comp.operator, cst.NotIn):
            text = f'not ({text})'
        self.object_memberships.append(_untyped(self.lines, before, 'in',
            f'compiled, operands are Python objects: {len(parts)} identity and == tests in tuple order, '
            'no tuple'))
        chain = cst.parse_expression(f'({text})')
        # The chain keeps its own parentheses whatever its position (an operand of and/not, a
        # comparison or arithmetic): without them `A and x in (p, q)` would read as
        # `(A and p == x) or q == x`. The original's own parentheses stay outside.
        return chain.with_changes(lpar=(*after.lpar, *chain.lpar), rpar=(*chain.rpar, *after.rpar))

    def member_tuple(self, node):
        """A literal tuple holding at least one ``self.<Ref>.<name>`` of a declared enum."""
        return isinstance(node, cst.Tuple) and any(
            isinstance(element, cst.Element) and isinstance(element.value, cst.Attribute)
            and isinstance(element.value.value, cst.Attribute)
            and isinstance(element.value.value.value, cst.Name) and element.value.value.value.value == 'self'
            and any(key[:2] == (self.cls.name, element.value.value.attr.value) for key in self.symbols)
            for element in node.elements)

    def decline(self, node, reason):
        from .reductions import untyped_entry
        self.untyped.append(untyped_entry(self.lines, node, 'in', reason))

    def leave_Call(self, before, after):
        func = after.func
        if (not isinstance(func, cst.Attribute) or not isinstance(func.value,cst.Name)
                or func.value.value != 'self' or any(a.keyword or a.star for a in after.args)):
            return after
        cell = self.cls.cells.get(func.attr.value)
        if cell is None or cell.called_with_kwargs or cell.body_has_closure:
            return after
        args = list(after.args)
        for i, param in enumerate(cell.params):
            if i < len(args) and cell.get_argtype_expr(param,c_style=True) in ('long long','bint'):
                args[i] = args[i].with_changes(value=self.member(args[i].value))
        return after.with_changes(args=tuple(args))

    def leave_Return(self, before, after):
        if (self.cell.get_rettype_expr(c_style=True) in ('long long','bint','double')
                and (self.cell.has_formula_def or not self.cell.called_with_kwargs)):
            return after.with_changes(value=self.member(after.value))
        return after


def _untyped(lines, node, form, reason):
    from .reductions import untyped_entry
    return untyped_entry(lines, node, form, reason)


class _MemberFields(cst.CSTTransformer):
    """``self.<Ref>.<member>`` of a declared enum -> ``self.<storage>.<member>`` (lexical scope)."""
    def __init__(self, cls_name, symbols):
        super().__init__()
        self.refs = {ref for owner, ref, _ in symbols if owner == cls_name}
        self.count = 0

    def on_visit(self, node):
        from .scopes import OPAQUE_SCOPES
        return not isinstance(node, OPAQUE_SCOPES) and super().on_visit(node)

    def leave_Attribute(self, before, after):
        inner = after.value
        if (isinstance(inner, cst.Attribute) and isinstance(inner.value, cst.Name)
                and inner.value.value == 'self' and inner.attr.value in self.refs):
            self.count += 1
            return after.with_changes(value=inner.with_changes(attr=cst.Name(storage(inner.attr.value))))
        return after


class Bindings(cst.CSTTransformer):
    def __init__(self, names):
        self.names = names

    def leave_AssignTarget(self, before, after):
        target = after.target
        if (isinstance(target, cst.Attribute) and isinstance(target.value,cst.Name)
                and target.value.value == 'self' and target.attr.value in self.names):
            return after.with_changes(target=target.with_changes(attr=cst.Name(storage(target.attr.value))))
        return after


def bind_properties(cls, body):
    names = getattr(cls, 'readonly_refs', ())
    result = []
    for statement in body:
        if names and isinstance(statement,cst.FunctionDef) and statement.name.value=='__init__':
            statements=list(statement.body.body)
            docstring=(statements and isinstance(statements[0],cst.SimpleStatementLine)
                and len(statements[0].body)==1 and isinstance(statements[0].body[0],cst.Expr)
                and isinstance(statements[0].body[0].value,(cst.SimpleString,cst.ConcatenatedString)))
            check=cst.parse_statement(f'{MX_SYS_MOD}._mx_check_enum_schema(self, {cls.name}, {names!r})\n')
            statements.insert(int(bool(docstring)),check)
            statement=statement.with_changes(body=statement.body.with_changes(body=tuple(statements)))
        if isinstance(statement, cst.FunctionDef) and statement.name.value in ('_mx_assign_refs','_mx_copy_refs'):
            statement = statement.visit(Bindings(names))
        result.append(statement)
    for name in names:
        result.extend(cst.parse_module(f'@property\ndef {name}(self):\n    return self.{storage(name)}\n').body)
    return tuple(result)


def _enum_key(ref):
    """(module, qualname) of an admitted IntEnum Reference, else None."""
    enum = getattr(ref, 'enum', None)
    return tuple(enum[:2]) if enum and getattr(ref, 'enum_safe', False) else None


def _traced_enum(cell, param):
    """(module, qualname) of the IntEnum class every traced value of a Cell's return (``param``
    None) or parameter had, else None. A spec return type or a mixed trace (merged to
    ``numbers.Integral``) gives None."""
    import enum
    trace = getattr(cell, '_rt', None)
    if trace is None or not cell.has_typeinfo():
        return None
    if param is None:
        if getattr(cell, '_spec_ret_t', None) or getattr(trace.ret_type, 'is_array', False):
            return None
        typ = trace.ret_type.value_type
    else:
        typ = getattr(trace, 'arg_types', {}).get(param)
    if isinstance(typ, type) and issubclass(typ, enum.IntEnum) and '<locals>' not in typ.__qualname__:
        return (typ.__module__, typ.__qualname__)
    return None
