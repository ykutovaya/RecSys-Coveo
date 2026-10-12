"""Поточная оценка сохраняет численный результат и не удерживает подробные строки."""

import weakref
from contextlib import ExitStack
from dataclasses import replace

import pytest

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes, strict_json_loads
from coveo_recsys.contracts.evaluation import POLICIES, EvaluationRecord, ReplayRecord
from coveo_recsys.contracts.preparation import PreparedRequest
from coveo_recsys.contracts.serialization import to_json
from coveo_recsys.contracts.training import TargetRecord
from coveo_recsys.data.prepared import TableWriter, iter_sessions, table_info, write_report
from coveo_recsys.evaluation import reporting
from tests.integration import test_e5_reporting as fixtures

# Сняты до изменения накопления rows/sensitivity_rows на трёх независимых малых
# случаях. Исключены лишь отчёт подготовки и физические порции выходной таблицы.
BASELINE = {
    "normal": (
        "9b07636f2975d04f0dbee00a0582283f5598060fef3dba3f3dbfc244047eb461",
        "80f70fd821bf2696d3e40c62d38d65aff77d097978db49fefa6e060ba1a545c9",
    ),
    "unknown": (
        "426b23ceb53984b8b6b5a5aa2868aca80f5e7f60a253e790f122c785738024f5",
        "a56941dd781afc692d578eb47acebc5f9e35cc413552f0da6f0a8a93006389e7",
    ),
    "fallback": (
        "eade4e540b37d990877b4ce5ee834be74590fe4982c45c1ebc61c05bf16c7e54",
        "f7412f98618851b323e6a8b7085100984d98019daa6b4bb70e1e55184338f8ef",
    ),
}


@pytest.mark.parametrize("case", BASELINE)
@pytest.mark.parametrize("chunk_rows", [1, 2, 64])
def test_streaming_preserves_exact_prechange_results_across_chunks(
    tmp_path, monkeypatch, case, chunk_rows
):
    original_plan = fixtures.preparation_plan
    monkeypatch.setattr(
        fixtures, "preparation_plan", lambda: replace(original_plan(), chunk_size=chunk_rows)
    )
    inputs = fixtures.saved_inputs(tmp_path, **({case: True} if case != "normal" else {}))
    summary = reporting.summarize_replay(*inputs, check_budget=lambda: None)
    semantic = {
        key: value for key, value in summary.items() if key not in {"preparation", "tables"}
    }
    records = b"\n".join(
        to_json(record)
        for _, group in iter_sessions(inputs[-1] / "replay/sessions.parquet", EvaluationRecord)
        for record in group
    )
    assert (sha256_bytes(canonical_json_bytes(semantic)), sha256_bytes(records)) == BASELINE[case]


def many_saved_inputs(tmp_path, count):
    original, prepared, answers_path, history, plan, evaluation_plan, _ = fixtures.saved_inputs(
        tmp_path / "templates"
    )
    prefixes = [
        rows[0]
        for _, rows in iter_sessions(
            original / "datasets/C_smoke/prefixes.parquet", PreparedRequest
        )
    ]
    targets = [
        rows[0]
        for _, rows in iter_sessions(original / "datasets/C_smoke/targets.parquet", TargetRecord)
    ]
    answers = [rows for _, rows in iter_sessions(answers_path, ReplayRecord)]
    source, destination = tmp_path / "many-prepared", tmp_path / "many-evaluation"
    plan = replace(
        plan,
        chunk_size=2,
        periods=tuple(
            replace(part, sample_size=count if part.name == "C_smoke" else 0)
            for part in plan.periods
        ),
    )
    paths = (
        (source / "datasets/C_smoke/prefixes.parquet", PreparedRequest),
        (source / "datasets/C_smoke/targets.parquet", TargetRecord),
        (destination / "replay/answers.parquet", ReplayRecord),
    )
    ids = [f"s{i:05d}" for i in range(count)]
    with ExitStack() as stack:
        writers = [
            stack.enter_context(TableWriter(path, kind, batch_rows=2)) for path, kind in paths
        ]
        for index, sid in enumerate(ids):
            template = index % len(prefixes)
            request, qid = prefixes[template], "q-" + sid
            blocks = tuple(
                replace(
                    block,
                    session_id=sid,
                    events=tuple(replace(event, session_id=sid) for event in block.events),
                )
                for block in request.prefix.blocks
            )
            prefix = replace(request.prefix, session_id=sid, blocks=blocks)
            writers[0].append(
                sid, (replace(request, request_id=qid, session_id=sid, prefix=prefix),)
            )
            writers[1].append(sid, (replace(targets[template], request_id=qid),))
            group = []
            for record in answers[template]:
                diagnostic = strict_json_loads(record.diagnostics_json)
                if record.policy == "GCS+Rk":
                    diagnostic["sensitivity"]["request_id"] = qid
                result = replace(
                    record.result,
                    request_id=qid,
                    candidates=replace(record.result.candidates, request_id=qid),
                )
                group.append(
                    replace(
                        record,
                        session_id=sid,
                        result=result,
                        diagnostics_json=canonical_json_bytes(diagnostic).decode(),
                    )
                )
            writers[2].append(sid, group)
    plan_json = to_json(plan).decode()
    prepared.update(
        plan_json=plan_json,
        inputs={**prepared["inputs"], "plan": sha256_bytes(plan_json.encode())},
        tables={
            str(path.relative_to(source)).replace("\\", "/"): table_info(
                path, kind, "C_smoke", role
            )
            for (path, kind), role in zip(paths[:2], ("prefix", "targets"), strict=True)
        },
        selection={"C_smoke": ids},
        period_counts={"C_smoke": dict(eligible=count + 2, admitted=count, selected=count)},
        N_obs=count,
        N_eligible=count + 2,
    )
    write_report(source, prepared)
    return source, prepared, paths[2][0], history, plan, evaluation_plan, destination


def test_reporting_releases_rich_details_after_each_session(tmp_path, monkeypatch):
    inputs = many_saved_inputs(tmp_path, 120)
    original = reporting.evaluate_session
    references = []
    peak = 0

    class TrackedDetails(dict):
        __slots__ = ("__weakref__",)

    def tracked(*args, **kwargs):
        nonlocal peak
        evaluation, details = original(*args, **kwargs)
        result = TrackedDetails(details)
        references.append(weakref.ref(result))
        peak = max(peak, sum(reference() is not None for reference in references))
        return evaluation, result

    monkeypatch.setattr(reporting, "evaluate_session", tracked)
    summary = reporting.summarize_replay(*inputs, check_budget=lambda: None)
    assert summary["data"]["N_obs"] == 120
    assert len(references) == 120 * len(POLICIES)
    # Допустима одна сессия из трёх политик плюс текущий объект следующей.
    assert peak <= len(POLICIES) + 1
    assert all(reference() is None for reference in references)
