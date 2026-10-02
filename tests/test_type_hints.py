"""Every function in the package carries a full type declaration.

An AST check rather than a type checker: it asks only that the signatures are
annotated, which is cheap, has no dependency, and is what silently rots as new
code lands. `self` and `cls` are exempt; everything else - every parameter,
*args, **kwargs - and every return type must be annotated.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "home_energy_optimizer"


def _unannotated(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        params = [*a.posonlyargs, *a.args, *a.kwonlyargs]
        exempt = set()
        if isinstance(parents.get(node), ast.ClassDef) and params:
            decorators = {d.id for d in node.decorator_list if isinstance(d, ast.Name)}
            if "staticmethod" not in decorators:
                exempt.add(params[0].arg)          # self / cls
        missing = [p.arg for p in params if p.annotation is None and p.arg not in exempt]
        if a.vararg is not None and a.vararg.annotation is None:
            missing.append("*" + a.vararg.arg)
        if a.kwarg is not None and a.kwarg.annotation is None:
            missing.append("**" + a.kwarg.arg)
        if node.returns is None:
            missing.append("-> return")
        if missing:
            out.append(f"{path.name}:{node.lineno} {node.name}: {', '.join(missing)}")
    return out


def test_every_function_is_annotated() -> None:
    files = sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)
    assert files, "no package sources found"
    missing = [m for p in files for m in _unannotated(p)]
    assert not missing, "unannotated signatures:\n" + "\n".join(missing)
