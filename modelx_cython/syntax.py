"""Small operations on the compiler's existing LibCST representation."""
from dataclasses import fields

import libcst as cst


def code(node):
    return cst.Module([]).code_for_node(node)


def statements(suite):
    """Expose statements without formatting-only line/suite wrappers."""
    body = suite.body if isinstance(suite, (cst.BaseSuite, cst.Module)) else suite
    return tuple(small for item in body for small in
                 (item.body if isinstance(item, cst.SimpleStatementLine) else (item,)))


def single_statement(line):
    if isinstance(line, cst.SimpleStatementLine) and len(line.body) == 1:
        return line.body[0]
    return None


def local_assignment(node):
    if (isinstance(node, cst.Assign) and len(node.targets) == 1
            and isinstance(target := node.targets[0].target, cst.Name)):
        return target
    return None


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def dotted_name(node):
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        parent = dotted_name(node.value)
        return parent + '.' + node.attr.value if parent else None
    return None


def arguments(node, name=None, count=None):
    if (not isinstance(node, cst.Call) or name is not None and dotted_name(node.func) != name
            or count is not None and len(node.args) != count
            or any(arg.keyword is not None or arg.star for arg in node.args)):
        return None
    return tuple(arg.value for arg in node.args)


def literal_value(node):
    """Decode CST literals, never execute names, calls or expressions.

    Decode numeric tokens directly and let LibCST decode string escapes.
    """
    if isinstance(node, cst.Integer):
        return int(node.value, 0)
    if isinstance(node, cst.Float):
        return float(node.value)
    if isinstance(node, cst.Imaginary):
        return complex(node.value)
    if isinstance(node, (cst.SimpleString, cst.ConcatenatedString)):
        value = node.evaluated_value
        if value is None:
            raise ValueError('formatted strings are not literals')
        return value
    if isinstance(node, cst.Name) and node.value in ('True', 'False', 'None'):
        return {'True': True, 'False': False, 'None': None}[node.value]
    if isinstance(node, cst.Ellipsis):
        return Ellipsis
    if isinstance(node, (cst.Tuple, cst.List, cst.Set)):
        if any(not isinstance(item, cst.Element) for item in node.elements):
            raise ValueError('starred values are not specification literals')
        values = (literal_value(item.value) for item in node.elements)
        return (tuple if isinstance(node, cst.Tuple) else
                list if isinstance(node, cst.List) else set)(values)
    if isinstance(node, cst.Dict):
        if any(not isinstance(item, cst.DictElement) for item in node.elements):
            raise ValueError('dictionary unpacking is not a specification literal')
        return {literal_value(item.key): literal_value(item.value) for item in node.elements}
    if (isinstance(node, cst.UnaryOperation) and isinstance(node.operator, (cst.Plus, cst.Minus))
            and isinstance(node.expression, (cst.Integer, cst.Float, cst.Imaginary))):
        value = literal_value(node.expression)
        return -value if isinstance(node.operator, cst.Minus) else value
    if (isinstance(node, cst.BinaryOperation) and isinstance(node.operator, (cst.Add, cst.Subtract))
            and isinstance(node.right, cst.Imaginary)):
        left, right = literal_value(node.left), literal_value(node.right)
        if type(left) in (int, float) and type(right) is complex:
            return left + right if isinstance(node.operator, cst.Add) else left - right
    if arguments(node, 'set', 0) is not None:
        return set()
    raise ValueError('expected a Python literal')


def read_literal(source):
    module = cst.parse_module(source.lstrip(' \t'))
    body = statements(module)
    if (len(body) != 1 or not isinstance(body[0], cst.Expr)
            or body[0].semicolon is not cst.MaybeSentinel.DEFAULT):
        raise ValueError('expected one literal expression')
    return literal_value(body[0].value)


_FORMATTING = frozenset((
    'lpar', 'rpar', 'comma', 'semicolon', 'leading_lines', 'lines_after_decorators',
    'header', 'footer', 'indent', 'trailing_whitespace', 'default_indent',
    'default_newline', 'encoding', 'has_trailing_newline', 'equal', 'dot',
))


def semantic_key(node):
    """Compare existing CST structure independently of comments and layout."""
    if isinstance(node, (cst.Integer, cst.Float, cst.Imaginary,
                         cst.SimpleString, cst.ConcatenatedString)):
        value = node.evaluated_value
        if value is not None:
            return 'literal', type(value).__name__, value
    if isinstance(node, cst.CSTNode):
        return type(node).__name__, tuple(
            (field.name, semantic_key(getattr(node, field.name)))
            for field in fields(node)
            if field.name not in _FORMATTING and not field.name.startswith('whitespace'))
    if isinstance(node, (tuple, list)):
        return tuple(semantic_key(item) for item in node)
    return node


def same_expression(left, right):
    return semantic_key(left) == semantic_key(right)
