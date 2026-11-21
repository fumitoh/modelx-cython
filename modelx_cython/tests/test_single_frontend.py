"""Model analysis stays on the original LibCST frontend."""
import ast
from pathlib import Path


def test_compiler_does_not_import_a_second_ast_frontend():
    package = Path(__file__).parents[1]
    offenders = []
    for path in package.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import) and any(alias.name == 'ast' for alias in node.names):
                offenders.append(path.name)
            if isinstance(node, ast.ImportFrom) and node.module == 'ast':
                offenders.append(path.name)
    assert not offenders
