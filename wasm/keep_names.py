#!/usr/bin/env python3
"""Print the names make_standalone.py must not rename, one per line.

`obfuscate()` renames every FunctionDef it finds and rewrites the name
wherever it appears -- including after a dot. Two ways that goes wrong:

  * A method or @property is reached as `horizon.times()`. Renaming the `def`
    without the attribute access gives "'Horizon' object has no attribute
    'times'" at runtime.
  * A module-level function can share a name with a dataclass FIELD.
    `dp_battery.terminal_price` and `BatteryConfig.terminal_price` collide, and
    renaming the function rewrote `batt.terminal_price` with it.

Both are the same underlying rule: a name that is ever reached through an
attribute must not be renamed. So keep every class member AND every name used
anywhere as an attribute; rename only module-level functions that are always
reached as a plain Name.

Both failures build cleanly and die at call time, which is why the standalone
is exercised in tests rather than eyeballed.

Run:  wasm/keep_names.py [bundle.py]
"""

from __future__ import annotations

import ast
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def keep(src: str) -> set[str]:
    names: set[str] = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        # anything reached through a dot, anywhere
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        # every class member: methods, properties, annotated fields, plain
        # assignments. Dataclass fields are AnnAssign and are read as
        # attributes, so they belong here even when nothing in this file
        # happens to read them.
        elif isinstance(node, ast.ClassDef):
            for inner in ast.walk(node):
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(inner.name)
                elif isinstance(inner, ast.AnnAssign) and isinstance(inner.target, ast.Name):
                    names.add(inner.target.id)
                elif isinstance(inner, ast.Assign):
                    for t in inner.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
    return names


def main() -> None:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "admm" / "wasm" / "admm_bundle.py"
    for name in sorted(keep(path.read_text())):
        print(name)


if __name__ == "__main__":
    main()
