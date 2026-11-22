"""The emitted text of a module must parse back to the tree the passes built.

libcst prints exactly the parentheses a node holds and never adds any for precedence. A pass
that puts a node of lower precedence (``or``, ``and``, ``not``, a comparison, an arithmetic
operator, if-else, lambda, ``:=``) where a higher one is needed without parentheses emits text
that Python reads differently from the lowering, silently: ``A and (p == x or q == x)`` printed
bare is ``(A and p == x) or q == x``.

The check is independent of any one pass and stays on the compiler's one frontend (libcst).
The tree is printed a second time with parentheses around every operator node, which is
always legal there and states the tree's own grouping. Both texts are parsed again and their
shapes compared, ignoring parentheses and layout (and the nesting of ``a or b or c``, which
evaluates the same operands in the same order however it is grouped). A difference is a
defect of the compiler and stops the build, naming the function and the expression.
"""
import dataclasses

import libcst as cst

# Expression nodes whose grouping depends on operator precedence; parentheses are legal
# wherever they can appear.
_OPERATORS = (cst.BooleanOperation, cst.BinaryOperation, cst.UnaryOperation, cst.Comparison,
              cst.IfExp, cst.Lambda, cst.NamedExpr, cst.Await, cst.Yield)

# Fields that hold layout only (parentheses, whitespace, separators, comments).
_LAYOUT = ('lpar', 'rpar', 'comma', 'semicolon', 'header', 'footer', 'leading_lines',
           'lines_after_decorators', 'trailing_whitespace', 'newline', 'indent', 'encoding',
           'default_indent', 'default_newline', 'has_trailing_newline', 'equal', 'colon', 'dot',
           'arrow', 'asynchronous')
_PRINTER = cst.Module([])


class PrecedenceError(RuntimeError):
    """The emitted text parses to a different expression than the tree the compiler built."""


class _Group(cst.CSTTransformer):
    def on_leave(self, original, updated):
        if isinstance(updated, _OPERATORS):
            return updated.with_changes(lpar=(cst.LeftParen(), *updated.lpar),
                                        rpar=(*updated.rpar, cst.RightParen()))
        return updated


def _layout(name):
    return name in _LAYOUT or name.startswith('whitespace')


def _operands(node, operator):
    """The operands of a chain of one boolean operator, in evaluation order."""
    if isinstance(node, cst.BooleanOperation) and type(node.operator) is operator:
        return [*_operands(node.left, operator), *_operands(node.right, operator)]
    return [node]


def _children(node):
    """(field name, value) pairs of ``node`` that carry meaning, a boolean chain flattened."""
    if isinstance(node, cst.BooleanOperation):
        return [('operator', type(node.operator).__name__),
                ('operands', _operands(node, type(node.operator)))]
    return [(field.name, getattr(node, field.name)) for field in dataclasses.fields(node)
            if not _layout(field.name)]


def _first_difference(a, b, function=None, expression=None):
    """(function name, emitted expression node) where the shapes first differ, or None."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return function, expression
        for x, y in zip(a, b):
            found = _first_difference(x, y, function, expression)
            if found:
                return found
        return None
    if type(a) is not type(b):
        return function, b if isinstance(b, cst.BaseExpression) else expression
    if not isinstance(a, cst.CSTNode):
        return None if a == b else (function, expression)
    if isinstance(b, cst.FunctionDef):
        function = b.name.value
    if isinstance(b, cst.BaseExpression):
        expression = b
    for (_, x), (_, y) in zip(_children(a), _children(b)):
        found = _first_difference(x, y, function, expression)
        if found:
            return found
    return None


def check(module, text=None, name='<module>'):
    """Raise :class:`PrecedenceError` if ``text`` (default ``module.code``) does not parse to
    the grouping held by ``module``; return the text."""
    if text is None:
        text = module.code
    emitted = cst.parse_module(text)
    built = cst.parse_module(module.visit(_Group()).code)
    found = _first_difference(built, emitted)
    if found is None:
        return text
    function, node = found
    where = f'function {function}' if function else 'module level'
    expression = ' '.join(_PRINTER.code_for_node(node).split()) if node is not None else '<no expression>'
    raise PrecedenceError(
        f'{name}: {where}: the generated text reads differently from the compiled tree '
        f'(a lowering dropped parentheses): {expression[:300]}')
