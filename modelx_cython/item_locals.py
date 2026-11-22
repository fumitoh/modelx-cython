"""D110 P1: a local bound to an ItemSpace of a typed Reference is declared that compiled class.

``item = self.projection[key]`` (or ``self.projection(key)``) in a compiled formula, where the
Reference ``projection`` is declared as a compiled ItemSpace class of this module, becomes
``item: _c_Projection = self.projection[key]``. Calls on ``item`` are then C calls of the
class's typed entries instead of Python attribute calls. Conditions, otherwise nothing changes:

* the local has exactly one binding in the formula (no other assignment, loop or ``with`` target,
  ``del``, augmented assignment, comprehension target or parameter), so it never holds anything else;
* the class's ``__getitem__`` is the exported ``return self.__call__(key)`` and every ``return``
  of its ``__call__`` is ``self._mx_itemspaces[...]`` or the root built there by
  ``<self or the base>.__class__(self)``: the value is an instance of the class (proven, not
  sampled). Cython's type test at the typed assignment remains, so a value of another type
  raises TypeError and is never cast wrongly.

Integer arguments of calls on such a local keep the checks of calls on ``self``
(``integer_boundaries._GuardCalls``), so errors are the same as through the Python gate.
"""
import libcst as cst

from .syntax import walk

MX_SELF = 'self'
_SCOPES = (cst.FunctionDef, cst.Lambda, cst.ClassDef)


def _method(cls_node, name):
    return next((item for item in cls_node.body.body
                 if isinstance(item, cst.FunctionDef) and item.name.value == name), None)


def _returns(node):
    return [part for part in walk(node.body) if isinstance(part, cst.Return)]


def _is_self_attr(node, attr=None):
    return (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name)
            and node.value.value == MX_SELF and (attr is None or node.attr.value == attr))


def returns_own_items(cls_node):
    """True when ``cls[key]`` and ``cls(key)`` of the exported class always return its own items."""
    getitem, call = _method(cls_node, '__getitem__'), _method(cls_node, '__call__')
    if getitem is None or call is None:
        return False
    body = getitem.body.body
    if len(body) != 1 or not isinstance(body[0], cst.SimpleStatementLine):
        return False
    ret = body[0].body[0]
    if not (isinstance(ret, cst.Return) and isinstance(ret.value, cst.Call)
            and _is_self_attr(ret.value.func, '__call__')):
        return False
    stores = {}
    for part in walk(call.body):
        if isinstance(part, cst.Assign):
            for target in part.targets:
                if isinstance(target.target, cst.Name):
                    stores.setdefault(target.target.value, []).append(part.value)
    base = stores.get('_mx_base', [])
    if base and not all(isinstance(v, cst.Name) and v.value == MX_SELF for v in base):
        return False

    def built_here(value):      # <self or _mx_base>.__class__(self)
        return (isinstance(value, cst.Call) and isinstance(value.func, cst.Attribute)
                and value.func.attr.value == '__class__' and isinstance(value.func.value, cst.Name)
                and value.func.value.value in (MX_SELF, '_mx_base') and len(value.args) == 1
                and isinstance(value.args[0].value, cst.Name) and value.args[0].value.value == MX_SELF)
    roots = {name for name, values in stores.items() if values and all(built_here(v) for v in values)}
    returns = _returns(call)
    if not returns:
        return False
    for ret in returns:
        value = ret.value
        if isinstance(value, cst.Subscript) and _is_self_attr(value.value, '_mx_itemspaces'):
            continue
        if isinstance(value, cst.Name) and value.value in roots:
            continue
        return False
    # every store into _mx_itemspaces anywhere in the class is such a root, made in __call__
    for part in walk(cls_node.body):
        targets = (part.targets if isinstance(part, cst.Assign) else
                   (cst.AssignTarget(part.target),) if isinstance(part, (cst.AnnAssign, cst.AugAssign)) else ())
        for target in targets:
            node = target.target
            if isinstance(node, cst.Subscript) and _is_self_attr(node.value, '_mx_itemspaces'):
                if not (isinstance(part, cst.Assign) and isinstance(part.value, cst.Name)
                        and part.value.value in roots and any(part is p for p in walk(call.body))):
                    return False
    return True


def _bound_names(target):
    if isinstance(target, cst.Name):
        yield target.value
    elif isinstance(target, (cst.Tuple, cst.List)):
        for element in target.elements:
            yield from _bound_names(element.value)
    elif isinstance(target, cst.StarredElement):
        yield from _bound_names(target.value)


def _binding_counts(func):
    counts = {}

    def add(target):
        for name in _bound_names(target):
            counts[name] = counts.get(name, 0) + 1
    for param in (*func.params.posonly_params, *func.params.params, *func.params.kwonly_params):
        add(param.name)
    for star in (func.params.star_arg, func.params.star_kwarg):
        if isinstance(star, cst.Param):
            add(star.name)
    for part in walk(func.body):
        if isinstance(part, cst.Assign):
            for target in part.targets:
                add(target.target)
        elif isinstance(part, (cst.AnnAssign, cst.AugAssign)):
            add(part.target)
        elif isinstance(part, (cst.For, cst.CompFor)):
            add(part.target)
        elif isinstance(part, cst.WithItem) and part.asname is not None:
            add(part.asname.name)
        elif isinstance(part, cst.NamedExpr):
            add(part.target)
        elif isinstance(part, cst.Del):
            add(part.target)
        elif isinstance(part, cst.ExceptHandler) and part.name is not None:
            add(part.name.name)
        elif isinstance(part, (cst.Global, cst.Nonlocal)):
            for item in part.names:
                counts[item.name.value] = counts.get(item.name.value, 0) + 2
        elif isinstance(part, (cst.Import, cst.ImportFrom)) and not isinstance(part.names, cst.ImportStar):
            for alias in part.names:
                add(alias.asname.name if alias.asname else cst.Name(alias.evaluated_name.split('.')[0]))
        elif isinstance(part, _SCOPES) and part is not func:
            if isinstance(part, (cst.FunctionDef, cst.ClassDef)):
                add(part.name)
    return counts


def item_classes(module):
    """{class name: True} for the module's ItemSpace classes whose items are proven of the class."""
    visitor = module.visitor
    found = {}
    for node in visitor.wrapper.module.body:
        if isinstance(node, cst.ClassDef) and node.name.value in visitor.itemspace_classes:
            found[node.name.value] = returns_own_items(node)
    return found


class _TypeItemLocals(cst.CSTTransformer):
    def __init__(self, eligible, counts):
        self.eligible, self.counts, self.typed = eligible, counts, {}

    def on_visit(self, node):
        return not (isinstance(node, _SCOPES) and getattr(self, '_entered', False)) and (
            setattr(self, '_entered', True) or True)

    def _owner(self, value):
        if isinstance(value, cst.Subscript):
            if len(value.slice) != 1 or not isinstance(value.slice[0].slice, cst.Index):
                return None
            receiver = value.value
        elif isinstance(value, cst.Call):
            if any(arg.star or arg.keyword is not None for arg in value.args):
                return None
            receiver = value.func
        else:
            return None
        if _is_self_attr(receiver):
            return self.eligible.get(receiver.attr.value)
        return None

    def leave_SimpleStatementLine(self, before, after):
        if len(after.body) != 1 or not isinstance(after.body[0], cst.Assign):
            return after
        assign = after.body[0]
        if len(assign.targets) != 1 or not isinstance(assign.targets[0].target, cst.Name):
            return after
        name = assign.targets[0].target.value
        owner = self._owner(assign.value)
        if owner is None or self.counts.get(name) != 1:
            return after
        self.typed[name] = owner
        return after.with_changes(body=(cst.AnnAssign(
            target=cst.Name(name), annotation=cst.Annotation(cst.Name(owner)), value=assign.value),))


def type_item_locals(node, cls_info, module):
    """Return (node, {local: class name}) for one compiled formula ``node`` (a FunctionDef)."""
    classes = getattr(module, '_mx_item_classes', None)
    if classes is None:
        classes = module._mx_item_classes = item_classes(module)
    eligible = {name: ref.decl_type_expr for name, ref in cls_info.refs.items()
                if getattr(ref, 'is_relative', False) and classes.get(getattr(ref, 'decl_type_expr', ''))}
    if not eligible:
        return node, {}
    transformer = _TypeItemLocals(eligible, _binding_counts(node))
    node = node.visit(transformer)
    return node, transformer.typed
