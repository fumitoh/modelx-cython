"""Recognize the exporter-owned collection evaluator without changing it."""

import libcst as cst

from .syntax import dotted_name, literal_value, statements


_VERSION = '_mx_collect_adapter_version'
PROTOCOL = 2
_HOOKS = {
    'collect_from_space': {'evaluate_serial', 'reuse_in_parallel', 'private_ordered'},
    '_mx_collect_parallel': {'private', 'private_ordered', 'workers'},
    '_mx_collect_chunk': {'private'},
    '_mx_collect_native_coords': set(),
}


def _current(tree):
    declarations = []
    for node in statements(tree):
        if isinstance(node, cst.Assign) and any(
                dotted_name(target.target) == _VERSION for target in node.targets):
            declarations.append(node.value)
        elif isinstance(node, (cst.AnnAssign, cst.AugAssign)) and dotted_name(node.target) == _VERSION:
            return False
    if len(declarations) != 1:
        return False
    try:
        version = literal_value(declarations[0])
    except (ValueError, TypeError):
        return False
    if type(version) is not int or version != PROTOCOL:
        return False
    for name, arguments in _HOOKS.items():
        functions = [node for node in tree.body
                     if isinstance(node, cst.FunctionDef) and node.name.value == name]
        if (len(functions) != 1 or functions[0].asynchronous is not None
                or not arguments <= {param.name.value for param in functions[0].params.params}):
            return False
    return True


def _flag(tree, name):
    for node in statements(tree):
        if isinstance(node, cst.Assign) and any(dotted_name(target.target) == name for target in node.targets):
            try:
                return literal_value(node.value) is True
            except (ValueError, TypeError):
                return False
    return False


class CollectRuntime:
    def __init__(self, source):
        self.syntax = cst.parse_module(source)
        self.hook = _current(self.syntax)
        # protocol 2 extension: the evaluator may take the collect's other Space parameters
        self.evaluator_args = self.hook and _flag(self.syntax, '_mx_collect_evaluator_args')

    def require_current(self, modules=(), workers=1):
        """The one check of the modelx that wrote the export: its runtime (``_mx_sys.py``) must
        declare export protocol 2 with its collect entries, whatever the build uses."""
        if not self.hook:
            raise ValueError(
                f'This export was written by a modelx without export protocol {PROTOCOL} '
                f'({_VERSION} = {PROTOCOL} in _mx_sys.py, with its collect entries), which '
                f'modelx-cython requires. Re-export the model with a modelx that writes protocol {PROTOCOL}.')
