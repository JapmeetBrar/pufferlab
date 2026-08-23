from __future__ import annotations

import ast
from pathlib import Path


def test_eval_package_has_no_service_provider_persistence_or_frontend_imports() -> None:
    eval_package = Path(__file__).parents[2] / "pufferlab" / "evals"
    forbidden_prefixes = (
        "fastapi",
        "sqlalchemy",
        "turbopuffer",
        "pufferlab.api",
        "pufferlab.jobs",
        "pufferlab.persistence",
        "pufferlab.providers",
        "pufferlab.retrieval",
        "web",
    )
    violations: list[str] = []

    for path in sorted(eval_package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = [node.module]
            for module in modules:
                if module.startswith(forbidden_prefixes):
                    violations.append(f"{path.name}:{node.lineno}: {module}")

    assert violations == []
