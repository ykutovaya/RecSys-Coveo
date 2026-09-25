import ast
import tomllib
from pathlib import Path

from pathspec import GitIgnoreSpec

ROOT = Path(__file__).resolve().parents[2]


def imports(path):
    result = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return result


def test_pure_modules_do_not_import_adapters_or_training():
    forbidden = {
        "fastapi",
        "uvicorn",
        "sqlite3",
        "requests",
        "httpx",
        "subprocess",
        "time",
        "torch",
        "coveo_recsys.cli",
        "coveo_recsys.artifacts",
        "coveo_recsys.config",
        "coveo_recsys.verification",
    }
    for directory in ("contracts", "numerics"):
        for path in (ROOT / "src/coveo_recsys" / directory).glob("*.py"):
            assert not any(
                name == item or name.startswith(item + ".")
                for name in imports(path)
                for item in forbidden
            ), path
    for path in (
        ROOT / "src/coveo_recsys/contracts/__init__.py",
        ROOT / "src/coveo_recsys/contracts/records.py",
    ):
        assert not any("training" in name for name in imports(path))


def test_validator_does_not_execute_verification():
    for path in (ROOT / "src/coveo_recsys/config").glob("*.py"):
        assert not any("verification" in name or name == "subprocess" for name in imports(path))


def test_e1_composition_and_future_targets_stay_out_of_prefix():
    package = ROOT / "src/coveo_recsys"
    for directory in ("data", "state"):
        for path in (package / directory).glob("*.py"):
            assert not any(
                name.startswith(
                    ("coveo_recsys.targets", "coveo_recsys.contracts.training", "coveo_recsys.cli")
                )
                for name in imports(path)
            ), path
            if directory == "data":
                assert not any(name.startswith("coveo_recsys.state") for name in imports(path)), (
                    path
                )
    for directory in ("state", "targets"):
        for path in (package / directory).glob("*.py"):
            assert not any(
                name.startswith(
                    (
                        "coveo_recsys.artifacts",
                        "coveo_recsys.config",
                        "coveo_recsys.verification",
                        "fastapi",
                        "subprocess",
                    )
                )
                or name in {"time", "sqlite3"}
                for name in imports(path)
            ), path


def test_environment_lock_has_hashes_and_no_base_torch():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert all("==" in name for name in project["project"]["dependencies"])
    assert not any(name.startswith("torch") for name in project["project"]["dependencies"])
    assert project["project"]["optional-dependencies"]["neural"] == ["torch==2.6.0"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    for package in lock["package"]:
        if "registry" in package["source"]:
            distributions = package.get("wheels", []) + (
                [package["sdist"]] if "sdist" in package else []
            )
            assert distributions, package["name"]
            assert all(item["hash"].startswith("sha256:") for item in distributions)


def test_normative_template_keeps_exact_bytes_across_checkouts():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.json text eol=lf" in attributes
    assert "*.tex text eol=lf" in attributes


def test_data_exclusion_does_not_hide_package_from_git_or_build():
    spec = GitIgnoreSpec.from_lines((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert spec.match_file("data/01_raw/browsing_train.csv")
    for path in (ROOT / "src/coveo_recsys").rglob("*.py"):
        assert not spec.match_file(path.relative_to(ROOT).as_posix()), path
