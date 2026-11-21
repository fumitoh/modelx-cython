"""Only registered helper kernels specialize, through the shared CST frontend."""
import ast
from types import SimpleNamespace
import libcst as cst

from modelx_cython.reductions import NativeReductions


def test_unregistered_builtin_preserves_source_without_an_ast_frontend(monkeypatch):
    source = "def formula(self, n):\n    return all(self.term(t) for t in range(n))\n"
    formula = cst.parse_module(source)
    def forbidden(*args, **kwargs):
        raise AssertionError("retired generator AST parser")
    with monkeypatch.context() as patch:
        patch.setattr(ast, "parse", forbidden)
        retired = NativeReductions(SimpleNamespace(refs={}), SimpleNamespace(params=()),
                                   set(), formula.body[0], lambda: '_cold', False)
        result = formula.visit(retired).code
    assert result == source
    assert retired.count == 0 and retired.fallbacks == []
