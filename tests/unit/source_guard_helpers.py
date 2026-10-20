"""Keep static guards on original code and the selected mutmut variant only."""

import ast
import os
import re


def active_source_tree(source: str, *, module: str = "") -> ast.Module:
    tree = ast.parse(source)
    if "MUTANT_UNDER_TEST" not in os.environ:
        return tree
    active_module, _, active_name = os.environ["MUTANT_UNDER_TEST"].rpartition(".")

    class ActiveFunctions(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST | None:
            if re.search(r"__mutmut_\d+$", node.name) and (module != active_module or node.name != active_name):
                return None
            return self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    ActiveFunctions().visit(tree)
    return tree
