"""Фиксированный публичный текст и проверенная числовая опора из трассы."""

from dataclasses import replace
from types import MappingProxyType

from coveo_recsys.artifacts import canonical_json_bytes, sha256_bytes
from coveo_recsys.contracts import InputValidationError, id_bytes
from coveo_recsys.contracts.historical import RELATIONS
from coveo_recsys.contracts.records import Branch, CandidateResult, Explanation, SourceEvidence
from coveo_recsys.contracts.retrieval import SOURCES
from coveo_recsys.retrieval.scores import QueryScores, RetrievalIndex
from coveo_recsys.retrieval.sources import source_evidence

TEMPLATE_VERSION = "1"
_TEXT = MappingProxyType(
    {
        "detail_detail": "Граф: просмотры → просмотры.",
        "detail_add": "Граф: просмотры → добавления в корзину.",
        "detail_purchase": "Граф: просмотры → покупки.",
        "description": "Сходство векторов описания.",
        "image": "Сходство векторов изображения.",
        "search": "Сходство с вектором доступного поиска.",
        "recent": "Недавний просмотр.",
        "popularity": "Популярность в исторических данных.",
        "fill": "Заполнение по исторической популярности.",
    }
)
TEMPLATE_SHA256 = sha256_bytes(
    canonical_json_bytes({"version": TEMPLATE_VERSION, "templates": dict(_TEXT)})
)


def explain(
    candidates: CandidateResult,
    delivered_skus: tuple[str, ...],
    *,
    package_id: str,
    scores: QueryScores | None,
    index: RetrievalIndex,
    template_version: str = TEMPLATE_VERSION,
) -> tuple[Explanation, ...]:
    """Строковые опоры остаются в SourceEvidence закрытой трассы, не в публичном тексте."""
    id_bytes(package_id)
    if template_version != TEMPLATE_VERSION:
        raise InputValidationError("Неизвестная версия шаблонов объяснений")
    if len(set(delivered_skus)) != len(delivered_skus) or not set(delivered_skus) <= set(
        candidates.skus
    ):
        raise InputValidationError("Объяснения требуют уникальной выдачи выбранной ветви")
    if not delivered_skus:
        return ()
    if candidates.branch not in {Branch.PRIMARY, Branch.FALLBACK}:
        raise InputValidationError("Не выбрана ветвь рекомендаций")
    if scores is not None and (
        scores.index is not index
        or scores.context.request_id != candidates.request_id
        or scores.context.catalog_sha256 != index.catalog_sha256
    ):
        raise InputValidationError("Баллы объяснения относятся к другому запросу или каталогу")
    origins = dict(candidates.origins)
    sources = {item.source: item for item in candidates.sources}
    evidence = {(item.sku, item.source): item for item in candidates.evidence}
    if (
        len(origins) != len(candidates.origins)
        or len(sources) != len(candidates.sources)
        or len(evidence) != len(candidates.evidence)
    ):
        raise InputValidationError("Неоднозначная опора объяснения")
    explanations = []
    for sku in delivered_skus:
        names = origins.get(sku, ())
        if (
            not names
            or len(set(names)) != len(names)
            or not set(names) <= set(_TEXT)
            or sku not in index.positions
            or ("fill" in names and names != ("fill",))
            or (candidates.branch == Branch.FALLBACK and not set(names) <= {"recent", "popularity"})
        ):
            raise InputValidationError("Неизвестное или несогласованное происхождение товара")
        ordered = tuple(source for source in (*SOURCES, "fill") if source in names)
        support: list[tuple[str, float]] = []
        position = index.positions[sku]
        for source in ordered:
            item = evidence.get((sku, source))
            if item is None:
                raise InputValidationError("В трассе отсутствует опора объяснения")
            if source == "fill":
                expected = SourceEvidence(
                    sku,
                    source,
                    None,
                    float(index.package.statistics.popularity[position]),
                    None,
                    None,
                    None,
                )
            else:
                result = sources.get(source)
                if (
                    result is None
                    or not result.available
                    or result.reason is not None
                    or sku not in result.skus
                ):
                    raise InputValidationError("Объяснение ссылается на неиспользованный источник")
                value = result.scores[result.skus.index(sku)]
                if value is None:
                    raise InputValidationError("В трассе отсутствует балл источника")
                if candidates.branch == Branch.PRIMARY:
                    if (
                        scores is None
                        or not scores.eligible[position]
                        or not scores.masks[source][position]
                        or scores.reasons[source] is not None
                        or float(scores.values[source][position]) != value
                    ):
                        raise InputValidationError(
                            "Баллы источника недоступны или расходятся с трассой"
                        )
                    expected = source_evidence(
                        (replace(result, skus=(sku,), scores=(value,)),), scores
                    )[0]
                else:
                    expected = SourceEvidence(sku, source, None, value, None, None, None)
                    if source == "popularity" and value != float(
                        index.package.statistics.popularity[position]
                    ):
                        raise InputValidationError("Резерв ссылается на другой расчёт популярности")
            if item != expected:
                raise InputValidationError("Опора объяснения расходится с закреплённым расчётом")
            support.append((source + ".score", item.score))
            if source in RELATIONS:
                if scores is None or item.anchor_sku is None or item.support is None:
                    raise InputValidationError("Нет графовой опоры объяснения")
                support.extend(
                    (
                        (
                            source + ".contribution",
                            scores.contributions[source][item.anchor_sku][position],
                        ),
                        (source + ".support", float(item.support)),
                    )
                )
        explanations.append(
            Explanation(
                sku,
                candidates.branch,
                package_id,
                template_version + ":" + "+".join(ordered),
                ordered,
                tuple(support),
                " ".join(_TEXT[source] for source in ordered),
            )
        )
    return tuple(explanations)
