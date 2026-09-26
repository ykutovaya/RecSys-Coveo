"""Применение модели: порядок, глубина, привязки и численные отказы."""

from dataclasses import replace

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError, NumericError
from coveo_recsys.contracts.records import FeatureBatch

SCHEMA_SHA = "a" * 64
CATALOG_SHA = "b" * 64


def batch(skus=("a", "b", "я")):
    return FeatureBatch(
        "q",
        skus,
        ("x",),
        ("category",),
        tuple((float(index),) for index in range(len(skus))),
        ((1,),) * len(skus),
        ((True, True),) * len(skus),
        SCHEMA_SHA,
    )


class RecordedBooster:
    def __init__(self, scores):
        self.scores = np.asarray(scores)
        self.calls = []

    def predict(self, data, **kwargs):
        self.calls.append((data.copy(), kwargs))
        return self.scores


def model(scores):
    from coveo_recsys.ranking.load import RankerModel

    return RankerModel(
        RecordedBooster(scores),
        {
            "identity": {
                "catalog_sha256": CATALOG_SHA,
                "feature_schema_sha256": SCHEMA_SHA,
            },
            "feature_schema_sha256": SCHEMA_SHA,
            "numeric_columns": ["x"],
            "categorical_columns": ["category"],
            "actual_tree_count": 2,
        },
    )


def test_predict_uses_raw_scores_once_and_utf8_ties_with_top_twenty():
    from coveo_recsys.ranking.predict import predict

    skus = tuple(f"s{index:02}" for index in range(22)) + ("я",)
    scores = [float(index % 4) for index in range(len(skus))]
    loaded = model(scores)
    result = predict(loaded, batch(skus), catalog_sha256=CATALOG_SHA)
    expected = sorted(zip(skus, scores), key=lambda item: (-item[1], item[0].encode()))[:20]
    assert result == tuple(expected)
    assert len(loaded.booster.calls) == 1
    matrix, kwargs = loaded.booster.calls[0]
    np.testing.assert_array_equal(matrix[:, 0], np.arange(len(skus)))
    assert kwargs == {"raw_score": True, "num_iteration": 2, "num_threads": 1}


def test_empty_batch_is_valid_without_calling_booster():
    from coveo_recsys.ranking.predict import predict

    loaded = model([])
    assert predict(loaded, batch(()), catalog_sha256=CATALOG_SHA) == ()
    assert loaded.booster.calls == []


@pytest.mark.parametrize("change", ["catalog", "schema", "columns", "sku_order", "category"])
def test_incompatible_input_is_rejected_before_prediction(change):
    from coveo_recsys.ranking.predict import predict

    loaded, features, catalog = model([1.0, 2.0, 3.0]), batch(), CATALOG_SHA
    if change == "catalog":
        catalog = "c" * 64
    elif change == "schema":
        features = replace(features, feature_schema_sha256="c" * 64)
    elif change == "columns":
        features = replace(features, numeric_columns=("other",))
    elif change == "sku_order":
        features = replace(features, skus=("я", "a", "b"))
    else:
        features = replace(features, categorical_columns=("other_category",))
    with pytest.raises(InputValidationError):
        predict(loaded, features, catalog_sha256=catalog)
    assert loaded.booster.calls == []


@pytest.mark.parametrize("scores", [[float("nan"), 0.0, 1.0], [0.0, float("inf"), 1.0], [1.0]])
def test_invalid_output_does_not_become_a_shortened_or_zero_filled_ranking(scores):
    from coveo_recsys.ranking.predict import predict

    with pytest.raises((InputValidationError, NumericError)):
        predict(model(scores), batch(), catalog_sha256=CATALOG_SHA)
