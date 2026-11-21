"""Registered functions behind module References, using the existing registry.

These records contain capabilities, not frozen callable bindings. Emission must
still evaluate the current member callable and let the provider check identity.
The shared tracer/merged-Reference wiring is a separately owned integration.
"""
import math
from math import ceil, erf, exp, floor, isclose, isinf, log, sqrt
from types import BuiltinFunctionType, FunctionType, ModuleType, SimpleNamespace

import libcst as cst

from .helpers import CAPABILITIES, helper_capability, helper_parameters, helper_result_type, registered_helper


_SCALAR_MATH = {erf: 'float64', exp: 'float64', isinf: 'bool', log: 'float64', sqrt: 'float64'}
MATH_PROVIDER = 'modelx_cython.runtime._native_math'
# W1.4: members of a module Reference resolved at build time (``_native_math.m_<name>``
# receives the Reference itself). floor/ceil return the module's int; see m_floor_ll.
_MODULE_MATH = {erf: 'float64', exp: 'float64', isinf: 'bool', log: 'float64', sqrt: 'float64',
                floor: 'int64', ceil: 'int64', isclose: 'bool'}      # isclose: W1.8
MATH_CONSTANTS = ('inf', 'nan', 'pi', 'e', 'tau')     # index order of _native_math.m_constant


def module_helpers(value):
    """Snapshot declared member metadata without executing module descriptors."""
    if type(value) is not ModuleType:
        return {}
    result = {}
    stdlib = value is math
    for name, function in vars(value).copy().items():
        if type(function) is BuiltinFunctionType and function in _MODULE_MATH:
            if function not in _SCALAR_MATH and not stdlib:
                continue          # floor/ceil only on the math module itself
            result[name] = SimpleNamespace(
                helper=(MATH_PROVIDER, function.__name__),
                helper_parameters=None, scalar_result=_MODULE_MATH[function],
                module_math=function.__name__ if stdlib or name == function.__name__ else None,
            )
            continue
        if stdlib and name in MATH_CONSTANTS and type(function) is float:
            result[name] = SimpleNamespace(helper=None, module_constant=MATH_CONSTANTS.index(name))
            continue
        if type(function) is not FunctionType:
            continue
        helper = registered_helper(function)
        if helper is None:
            continue
        capabilities = {field: helper_capability(function, suffix)
                        for suffix, (field, _) in CAPABILITIES.items()}
        result[name] = SimpleNamespace(helper=helper, helper_parameters=helper_parameters(function),
                                       helper_result_type=helper_result_type(function),
                                       module_math=(helper[1] if helper[0] == MATH_PROVIDER
                                                    and helper[1] == name else None),
                                       **capabilities)
    return result


def helper_reference(function, refs):
    """Resolve direct References or one declared module member in this scope."""
    if not isinstance(function, cst.Attribute):
        return None
    receiver = function.value
    if isinstance(receiver, cst.Name) and receiver.value == 'self':
        return refs.get(function.attr.value)
    if (isinstance(receiver, cst.Attribute) and isinstance(receiver.value, cst.Name)
            and receiver.value.value == 'self'):
        namespace = refs.get(receiver.attr.value)
        return getattr(namespace, 'helper_members', {}).get(function.attr.value)
    return None


_KINDS = {'double': 'double', 'long long': 'longlong', 'bint': 'bint'}


def argument_kind(node, current, cells, refs):
    """C kind of a math argument from declared types only: 'double', 'longlong', 'bint' or None.

    ``current`` is the formula's cells (parameter types), ``cells`` the class cells by name,
    ``refs`` the class References. Unknown or object operands give None.
    """
    if isinstance(node, cst.Float):
        return 'double'
    if isinstance(node, cst.Integer):
        return 'longlong' if node.evaluated_value < (1 << 63) else None
    if isinstance(node, cst.Name):
        if current is not None and node.value in getattr(current, 'params', ()) and current.has_typeinfo():
            return _KINDS.get(current.get_argtype_expr(node.value, c_style=True, formula=True))
        return None
    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, (cst.Minus, cst.Plus)):
        kind = argument_kind(node.expression, current, cells, refs)
        return 'longlong' if kind == 'bint' else kind
    if isinstance(node, cst.BinaryOperation) and isinstance(
            node.operator, (cst.Add, cst.Subtract, cst.Multiply, cst.Divide)):
        left = argument_kind(node.left, current, cells, refs)
        right = argument_kind(node.right, current, cells, refs)
        if left is None or right is None:
            return None
        if isinstance(node.operator, cst.Divide) or 'double' in (left, right):
            return 'double'
        return 'longlong'    # C integer arithmetic, as Cython compiles the operand
    if (isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name)
            and node.value.value == 'self'):
        ref = refs.get(node.attr.value)
        return {float: 'double', int: 'longlong', bool: 'bint'}.get(getattr(ref, 'type_', None))
    if isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute):
        func = node.func
        if isinstance(func.value, cst.Name) and func.value.value == 'self':
            cell = cells.get(func.attr.value) if hasattr(cells, 'get') else None
            if (cell is not None and cell.has_typeinfo() and not cell.called_with_kwargs
                    and not cell.body_has_closure):
                return _KINDS.get(cell.get_rettype_expr(c_style=True))
            return None
        member = helper_reference(func, refs)
        if getattr(member, 'module_math', None):
            return {'float64': 'double', 'bool': 'bint'}.get(member.scalar_result)
    return None
