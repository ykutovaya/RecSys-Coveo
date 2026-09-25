"""Хеширование исходников с явной областью; результаты запусков не входят."""

from pathlib import Path

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes


def source_inventory(root: Path) -> dict[str, str]:
    """Состав кода и тестов. Не включает .git, runs, окружение и документацию."""
    result = {}
    for directory, suffixes in (
        ("src", {".py"}),
        ("tests", {".py", ".json", ".csv"}),
        ("schemas", {".json"}),
        ("configs", {".json"}),
    ):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                result[path.relative_to(root).as_posix()] = sha256_bytes(path.read_bytes())
    for name in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".gitattributes",
        ".gitignore",
        ".github/workflows/checks.yml",
    ):
        path = root / name
        if path.is_file():
            result[name] = sha256_bytes(path.read_bytes())
    return result


def source_sha256(root: Path) -> str:
    return sha256_bytes(canonical_json_bytes(source_inventory(root)))


def assert_source_binding(root: Path) -> None:
    """Паспорт расчёта должен описывать исполняемый пакет, в том числе установленный wheel."""
    actual = Path(__file__).resolve().parents[1]
    expected = (root / "src/coveo_recsys").resolve()
    if not expected.is_dir():
        raise ValueError("project-root не содержит исходников пакета")
    if actual == expected:
        return

    def inventory(directory: Path) -> dict[str, str]:
        return {
            p.relative_to(directory).as_posix(): sha256_bytes(p.read_bytes())
            for p in directory.rglob("*.py")
            if "__pycache__" not in p.parts
        }

    if inventory(actual) != inventory(expected):
        raise ValueError("Исполняемый пакет не соответствует project-root")
