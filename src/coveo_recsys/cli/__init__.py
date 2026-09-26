"""Подготовка, исторический пакет, признаки и обучение ранжировщика."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from coveo_recsys import __version__


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _read_json(path: Path) -> dict[str, Any]:
    from coveo_recsys.artifacts import strict_json_loads

    value = strict_json_loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Ожидался JSON-объект")
    return value


def _parser() -> argparse.ArgumentParser:
    from coveo_recsys.contracts import Stage

    parser = argparse.ArgumentParser(prog="coveo-recsys")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("environment", help="Сведения о фактической среде")
    manifest = commands.add_parser("manifest", help="Проект паспорта и условия стадий")
    actions = manifest.add_subparsers(dest="action", required=True)
    init = actions.add_parser("init")
    init.add_argument("--run-id", required=True)
    init.add_argument("--run-dir", required=True, type=Path)
    init.add_argument(
        "--scope",
        choices=("synthetic", "actual", "offline_first_run", "production", "research"),
        default="synthetic",
    )
    validation = actions.add_parser("validate")
    validation.add_argument("--manifest", type=Path, required=True)
    validation.add_argument("--run-dir", type=Path, required=True)
    validation.add_argument("--stage", choices=[s.value for s in Stage], required=True)
    validation.add_argument("--plan-reference", action="append", default=[], metavar="NAME=SHA256")
    validation.add_argument("--input-file", action="append", default=[], metavar="NAME=PATH")
    validation.add_argument("--operation", choices=("evaluate",))
    data = commands.add_parser("data", help="Подготовка искусственных данных Э1")
    data_actions = data.add_subparsers(dest="action", required=True)
    inspect = data_actions.add_parser("inspect")
    prepare = data_actions.add_parser("prepare")
    supply = data_actions.add_parser(
        "inspect-supply", help="Проверка поставки Coveo без подготовки"
    )
    for action in (inspect, prepare, supply):
        action.add_argument("--input-dir", type=Path, required=True)
        action.add_argument("--profile", type=Path, required=True)
        action.add_argument("--run-dir", type=Path, required=True)
        action.add_argument("--run-id", required=True)
        action.add_argument("--batch-size", type=int, default=1024)
        action.add_argument("--max-batch-chars", type=int, default=4_194_304)
    prepare.add_argument("--plan", type=Path, required=True)
    supply.add_argument("--max-header-bytes", type=int, default=4096)
    supply.add_argument("--max-record-bytes", type=int, default=1_048_576)
    supply.add_argument("--max-field-chars", type=int, default=1_048_576)
    supply.add_argument("--max-seconds", type=int, default=7200)
    supply.add_argument("--max-temp-bytes", type=int, default=67_108_864)
    historical = commands.add_parser("historical", help="Исторический пакет A_G")
    historical_actions = historical.add_subparsers(dest="action", required=True)
    historical_build = historical_actions.add_parser("build")
    historical_build.add_argument("--config", type=Path, required=True)
    historical_build.add_argument("--run-dir", type=Path, required=True)
    historical_build.add_argument("--run-id", required=True)
    features = commands.add_parser("features", help="Кандидаты и признаки на тестовых наборах")
    feature_actions = features.add_subparsers(dest="action", required=True)
    feature_build = feature_actions.add_parser("build")
    feature_build.add_argument("--config", type=Path, required=True)
    feature_build.add_argument("--run-dir", type=Path, required=True)
    feature_build.add_argument("--run-id", required=True)
    feature_build.add_argument("--period", choices=("B",), required=True)
    train = commands.add_parser("train", help="Обучение ранжировщика")
    train_actions = train.add_subparsers(dest="action", required=True)
    ranker = train_actions.add_parser("ranker")
    ranker.add_argument("--config", type=Path, required=True)
    ranker.add_argument("--run-dir", type=Path, required=True)
    ranker.add_argument("--run-id", required=True)
    for name in (
        "replay",
        "select",
        "evaluate",
        "serve",
        "load",
        "experiment",
    ):
        command = commands.add_parser(name, help="Ещё не реализовано")
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _plans(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, digest = value.partition("=")
        if (
            not separator
            or name not in {"before_C_select", "before_D", "before_AB"}
            or name in result
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Ожидалась уникальная пара имени плана и SHA-256")
        result[name] = digest
    return result


def _inputs(values: list[str]) -> dict[str, Path] | None:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("Ожидалась уникальная пара NAME=PATH")
        result[name] = Path(path)
    return result or None


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command not in {"environment", "manifest", "data", "historical", "features", "train"}:
        print(f"Команда {args.command} ещё не реализована", file=sys.stderr)
        return 2
    try:
        if args.command == "train":
            from contextlib import redirect_stdout

            from coveo_recsys.cli.training import run_training

            try:
                # LightGBM сообщает ход обучения в stdout; JSON команды остаётся отдельным.
                with redirect_stdout(sys.stderr):
                    result = run_training(args.project_root, args.config, args.run_dir, args.run_id)
            except KeyboardInterrupt:
                _emit({"status": "failed", "reason": "interrupted"})
                return 130
            _emit(result)
            return 0 if result["status"] == "completed" else 1
        if args.command == "environment":
            from coveo_recsys.environment import environment_report

            _emit(environment_report(args.project_root))
            return 0
        if args.command == "historical":
            from coveo_recsys.cli.historical import run_historical

            try:
                result = run_historical(args.project_root, args.config, args.run_dir, args.run_id)
            except KeyboardInterrupt:
                _emit({"status": "failed", "reason": "interrupted"})
                return 130
            _emit(result)
            return 0
        if args.command == "features":
            from coveo_recsys.cli.features import run_features

            try:
                result = run_features(
                    args.project_root, args.config, args.run_dir, args.run_id, period=args.period
                )
            except KeyboardInterrupt:
                _emit({"status": "failed", "reason": "interrupted"})
                return 130
            _emit(result)
            return 0
        from coveo_recsys.artifacts import LocalArtifactStore, canonical_json_bytes

        store = LocalArtifactStore(args.run_dir)
        if args.command == "data":
            if args.action == "inspect-supply":
                from coveo_recsys.cli.supply import inspect_supply_command
                from coveo_recsys.data.supply import SupplyScanError

                try:
                    result = inspect_supply_command(
                        store,
                        args.project_root,
                        args.run_id,
                        args.input_dir,
                        args.profile,
                        batch_size=args.batch_size,
                        max_batch_chars=args.max_batch_chars,
                        max_header_bytes=args.max_header_bytes,
                        max_record_bytes=args.max_record_bytes,
                        max_field_chars=args.max_field_chars,
                        max_seconds=args.max_seconds,
                        max_temp_bytes=args.max_temp_bytes,
                    )
                except KeyboardInterrupt:
                    _emit({"status": "failed", "reason": "interrupted"})
                    return 130
                except SupplyScanError as exc:
                    _emit(
                        {
                            "status": "failed",
                            "reason": "supply_validation_failed",
                            "error": {
                                "code": exc.code,
                                "file": exc.file,
                                "column": exc.column,
                                "line": exc.line,
                            },
                        }
                    )
                    return 1
                except (
                    ValueError,
                    OSError,
                    TypeError,
                    KeyError,
                    ArithmeticError,
                    RecursionError,
                ) as exc:
                    # Ошибки поставки не должны раскрывать содержимое CSV или локальные пути.
                    _emit({"status": "failed", "reason": type(exc).__name__})
                    return 1
                _emit(result)
                return 0 if result["format_status"] == "confirmed" else 1
            from coveo_recsys.cli.preparation import Preparation

            preparation = Preparation(store, args.project_root, args.run_id)
            try:
                options = dict(batch_size=args.batch_size, max_batch_chars=args.max_batch_chars)
                if args.action == "inspect":
                    result = preparation.inspect(args.input_dir, args.profile, **options)
                else:
                    result = preparation.prepare(args.input_dir, args.profile, args.plan, **options)
            except KeyboardInterrupt:
                preparation.progress("failed", reason="interrupted")
                _emit({"status": "failed", "reason": "interrupted"})
                return 130
            except (
                ValueError,
                OSError,
                TypeError,
                KeyError,
                ArithmeticError,
                RecursionError,
            ) as exc:
                if preparation.common["scope"] == "actual":
                    preparation.progress("failed", reason=type(exc).__name__)
                    _emit({"status": "failed", "reason": type(exc).__name__})
                    return 2
                preparation.progress("failed", reason=type(exc).__name__, message=str(exc))
                raise
            _emit(result)
            return 0
        from coveo_recsys.config import init_manifest, validate_manifest

        if args.action == "init":
            manifest = init_manifest(args.run_id, scope=args.scope)
            store._atomic_write("manifest.json", canonical_json_bytes(manifest))
            _emit(
                {
                    "path": "manifest.json",
                    "manifest_version": manifest["manifest_version"],
                    "status": manifest["status"],
                }
            )
            return 0
        manifest = _read_json(args.manifest)
        validation = validate_manifest(
            manifest,
            args.stage,
            store,
            plan_references=_plans(args.plan_reference),
            input_paths=_inputs(args.input_file),
            operation=args.operation,
        )
        _emit(asdict(validation))
        return 0 if validation.gate_passed else 1
    except (ValueError, OSError, TypeError, KeyError, ArithmeticError, RecursionError) as exc:
        _emit({"status": "failed", "reason": type(exc).__name__, "message": str(exc)})
        return 2
