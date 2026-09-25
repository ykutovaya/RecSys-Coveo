"""Сведения о фактически запущенной среде, а не о нагрузочном узле."""

import hashlib
import importlib.metadata
import platform
import sys
import sysconfig
from pathlib import Path
from typing import Any


def environment_report(project_root: Path) -> dict[str, Any]:
    import numpy as np

    show_config: Any = np.show_config  # NumPy 2.1.3: параметр mode отсутствует в .pyi.

    distributions = (
        "numpy",
        "scipy",
        "pyarrow",
        "lightgbm",
        "jsonschema",
        "fastapi",
        "uvicorn",
        "pytest",
        "hypothesis",
        "mypy",
        "ruff",
        "hatchling",
    )
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    files = {}
    for name in ("uv.lock", "pyproject.toml", ".python-version"):
        path = project_root / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {
        "scope": "development_environment",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "compiler": platform.python_compiler(),
        "python_build": platform.python_build(),
        "soabi": sysconfig.get_config_var("SOABI"),
        "libraries": versions,
        "numpy_build": show_config(mode="dicts"),
        "lock_files_sha256": files,
        "torch_imported": "torch" in sys.modules,
        "load_measurements": None,
    }
