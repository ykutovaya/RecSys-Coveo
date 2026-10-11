"""Настройки и проверки малого пути A_G → B → C_smoke; расчёты запускаются отдельно."""

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from coveo_recsys.artifacts import canonical_json_bytes, sha256_file, strict_json_loads
from coveo_recsys.contracts import InputValidationError, require_int64
from coveo_recsys.contracts.preparation import PreparationPlan
from coveo_recsys.contracts.retrieval import RetrievalConfig
from coveo_recsys.contracts.serialization import from_json, to_json
from coveo_recsys.data.prepared import read_report
from coveo_recsys.verification.provenance import assert_source_binding, source_sha256

CSV_FILES = ("browsing_train.csv", "search_train.csv", "sku_to_content.csv")
DIRECTORIES = {"A_G": "a_g", "B": "b", "C_smoke": "c_smoke"}
PROFILE = "configs/coveo_supply_v1.json"


def _work(root: Path, work: Path) -> Path:
    if any(path.is_symlink() for path in (work, *work.parents)):
        raise InputValidationError("Каталог результата не должен содержать символические ссылки")
    allowed = (root / "runs").resolve()
    work = work.resolve()
    if work == allowed or allowed not in work.parents:
        raise InputValidationError("Нужен отдельный каталог внутри runs")
    return work


def _publish(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise InputValidationError("Ссылка вместо файла результата")
    if path.exists():
        if path.read_bytes() != payload:
            raise InputValidationError(f"Нельзя изменить готовый файл: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def initialize(
    root: Path,
    work: Path,
    input_dir: Path,
    *,
    ag_sessions: int,
    b_sessions: int,
    smoke_sessions: int,
    max_seconds: int,
    max_memory_bytes: int,
    max_temp_bytes: int,
) -> None:
    assert_source_binding(root)
    work = _work(root, work)
    counts = dict(zip(DIRECTORIES, (ag_sessions, b_sessions, smoke_sessions)))
    if any(
        require_int64(value, nonnegative=True) == 0
        for value in (*counts.values(), max_seconds, max_memory_bytes, max_temp_bytes)
    ):
        raise InputValidationError("Нужны положительные объёмы и пределы")
    if any(not (input_dir / name).is_file() for name in CSV_FILES):
        raise InputValidationError("Нужны три исходных CSV Coveo")
    original = from_json((root / "configs/coveo_preparation.json").read_bytes(), PreparationPlan)
    plans = {
        name: replace(
            original,
            periods=tuple(
                replace(p, sample_size=size if p.name == name else 0) for p in original.periods
            ),
            max_sessions=size,
            diagnostic_sessions=0,
            max_seconds=max_seconds,
            max_memory_bytes=max_memory_bytes,
            max_temp_bytes=max_temp_bytes,
        )
        for name, size in counts.items()
    }
    work.mkdir(parents=True, exist_ok=False)
    for name, plan in plans.items():
        _publish(work / f"{name}-plan.json", to_json(plan))


def smoke_evaluation_plan() -> dict[str, Any]:
    names = ("GCS+Rk", "P", "R", "GCS+Rk-P", "GCS+Rk-R")
    return {
        "period": "C_smoke",
        "policies": ["GCS+Rk", "P", "R"],
        "k": 20,
        "diagnostic_depth": 50,
        "intervals": {
            "grid_step_ms": 1_800_000,
            "block_cells": 12,
            "sensitivity_block_cells": [6, 24],
            "mc_error_method": "binomial_order_statistic_95pct",
            "dependence": {name: {"supported": None, "assumptions": []} for name in names},
            "half_width_fraction": dict.fromkeys(names),
        },
    }


def _pin(root: Path, directory: Path, report: str) -> dict[str, str]:
    path = directory / report
    if report != "historical/package.json":
        if strict_json_loads(path.read_bytes()).get("status") != "completed":
            raise InputValidationError(f"Нет завершённого результата: {path}")
    field = "package_sha256" if report == "historical/package.json" else "report_sha256"
    return {"path": directory.relative_to(root).as_posix(), field: sha256_file(path)}


def check_data(root: Path, work: Path, input_dir: Path, period: str) -> Path:
    from coveo_recsys.verification.preparation import verify_preparation

    work = _work(root, work)
    plan_path = work / f"{period}-plan.json"
    plan = from_json(plan_path.read_bytes(), PreparationPlan)
    prepared = work / DIRECTORIES[period]
    peers = tuple(
        work / name
        for key, name in DIRECTORIES.items()
        if key != period and (work / name / "reports/preparation.json").is_file()
    )
    result = verify_preparation(
        root,
        prepared,
        input_dir,
        root / PROFILE,
        plan_path,
        ag_dir=None if period == "A_G" else work / "a_g",
        historical_dir=None if period == "A_G" else work / "historical",
        other_preparations=peers,
        max_seconds=plan.max_seconds,
        max_memory_bytes=plan.max_memory_bytes,
    )
    output = prepared / "reports/data-check.json"
    _publish(output, canonical_json_bytes(result))
    return output


def make_config(
    root: Path, work: Path, input_dir: Path, step: str, *, max_result_bytes: int
) -> Path:
    from coveo_recsys.verification.admission import assemble_admission, checked_preparation

    root = root.resolve()
    work = _work(root, work)
    assert_source_binding(root)
    if require_int64(max_result_bytes, nonnegative=True) == 0:
        raise InputValidationError("Нужен предел размера результата")
    b_plan = from_json((work / "B-plan.json").read_bytes(), PreparationPlan)
    c_plan = from_json((work / "C_smoke-plan.json").read_bytes(), PreparationPlan)
    ag_plan = from_json((work / "A_G-plan.json").read_bytes(), PreparationPlan)
    metadata = (input_dir.resolve() / "sku_to_content.csv").as_posix()
    limits = {
        "max_seconds": b_plan.max_seconds,
        "max_memory_bytes": b_plan.max_memory_bytes,
        "max_result_bytes": max_result_bytes,
    }
    if step == "historical":
        settings = strict_json_loads((root / "configs/coveo_historical.json").read_bytes())
        settings["preparation"] = _pin(root, work / "a_g", "reports/preparation.json")
        settings["metadata_path"] = metadata
        settings["config"].update(
            {
                **limits,
                "max_seconds": ag_plan.max_seconds,
                "max_memory_bytes": ag_plan.max_memory_bytes,
                "max_temp_bytes": ag_plan.max_temp_bytes,
            }
        )
    elif step == "replay":
        settings = strict_json_loads((work / "evaluation-preparation.json").read_bytes())
        settings["preparation"] = _pin(root, work / "c_smoke", "reports/preparation.json")
        checked = checked_preparation(
            work / "c_smoke",
            read_report(work / "c_smoke", verify_tables=False),
            source_sha256(root),
            "C_smoke",
        )
        if (
            checked["bindings"].get("ag_report_sha256")
            != _pin(root, work / "a_g", "reports/preparation.json")["report_sha256"]
            or checked["bindings"].get("historical_package_sha256")
            != settings["historical"]["package_sha256"]
        ):
            raise InputValidationError("Нет текущей независимой проверки C_smoke")
    elif step in {"features", "train", "evaluation-preparation"}:
        base = {
            "preparation": _pin(root, work / "b", "reports/preparation.json"),
            "historical": _pin(root, work / "historical", "historical/package.json"),
            "feature_schema": "configs/feature_schema.json",
        }
        if step == "train":
            feature_settings = strict_json_loads((work / "features.json").read_bytes())
            if any(feature_settings[name] != value for name, value in base.items()):
                raise InputValidationError("Изменены входы признаков")
            settings = {
                **base,
                "features": _pin(root, work / "b_features", "reports/features.json"),
                "recipe": "configs/ranker_recipe.json",
                "admission": feature_settings["admission"],
                "limits": {
                    **limits,
                    "max_requests": b_plan.max_sessions,
                    "max_rows": b_plan.max_sessions * 300,
                },
            }
        else:
            stage = "preflight" if step == "features" else "post_training"
            admission = assemble_admission(
                root,
                work / stage,
                run_id=f"{work.name}-{stage}",
                prepared_dir=work / "b",
                ag_dir=work / "a_g",
                historical_dir=work / "historical",
                test_run=work / "tests",
                input_dir=input_dir,
                stage=stage,
                features_dir=None if step == "features" else work / "b_features",
                model_dir=None if step == "features" else work / "ranker",
                evaluation_plan_path=None if step == "features" else work / "C_smoke-plan.json",
            )
            settings = {
                **base,
                "retrieval": asdict(RetrievalConfig()),
                "metadata_path": metadata,
                "admission": admission,
                "limits": {
                    **limits,
                    "max_requests": b_plan.max_sessions,
                    "batch_size": 16,
                    "max_observed_skus": 100_000,
                },
            }
            if step == "evaluation-preparation":
                settings["training_preparation"] = settings["preparation"]
                settings.update(
                    preparation=None,
                    training_features=_pin(root, work / "b_features", "reports/features.json"),
                    model=_pin(root, work / "ranker", "reports/ranker.json"),
                    preparation_plan=(work / "C_smoke-plan.json").relative_to(root).as_posix(),
                    recipe="configs/ranker_recipe.json",
                    evaluation_plan=smoke_evaluation_plan(),
                )
                settings["limits"].update(
                    max_requests=c_plan.max_sessions,
                    max_seconds=c_plan.max_seconds,
                    max_memory_bytes=c_plan.max_memory_bytes,
                )
    else:
        raise InputValidationError(f"Неизвестный шаг: {step}")
    output = work / f"{step}.json"
    _publish(output, canonical_json_bytes(settings))
    return output


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=(
            "init",
            "tests",
            "check-ag",
            "check-b",
            "check-csmoke",
            "historical",
            "features",
            "train",
            "evaluation-preparation",
            "replay",
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("data/01_raw"))
    for name in (
        "ag-sessions",
        "b-sessions",
        "smoke-sessions",
        "max-seconds",
        "max-memory-bytes",
        "max-temp-bytes",
        "max-result-bytes",
    ):
        parser.add_argument(f"--{name}", type=int)
    args = parser.parse_args()
    root = args.project_root.resolve()
    work, inputs = root / args.work_dir, root / args.input_dir
    try:
        assert_source_binding(root)
        work = _work(root, work)
        if args.step == "init":
            initialize(
                root,
                work,
                inputs,
                **{
                    key: getattr(args, key)
                    for key in (
                        "ag_sessions",
                        "b_sessions",
                        "smoke_sessions",
                        "max_seconds",
                        "max_memory_bytes",
                        "max_temp_bytes",
                    )
                },
            )
            print(work)
        elif args.step == "tests":
            from coveo_recsys.verification.run_tests import run_tests

            result = run_tests(root, work / "tests")
            print(result)
            return 0 if result["status"] == "passed" else 1
        elif args.step.startswith("check-"):
            period = {"check-ag": "A_G", "check-b": "B", "check-csmoke": "C_smoke"}[args.step]
            print(check_data(root, work, inputs, period))
        else:
            print(
                make_config(root, work, inputs, args.step, max_result_bytes=args.max_result_bytes)
            )
        return 0
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
