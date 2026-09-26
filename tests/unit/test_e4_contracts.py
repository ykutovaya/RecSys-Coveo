from dataclasses import replace

import numpy as np
import pytest

from coveo_recsys.contracts import InputValidationError
from coveo_recsys.contracts.records import FeatureBatch
from coveo_recsys.contracts.serialization import from_json, record_version, to_json
from coveo_recsys.contracts.training import SupervisionRecord


@pytest.mark.parametrize(
    "change",
    [
        {"session_request_count": 2},
        {"session_weight": 0.5},
        {"pair_weights": (1.0, 1.0)},
        {"pair_weights": (0.5,)},
    ],
)
def test_observed_supervision_has_one_group_and_one_normalized_weight(change):
    value = SupervisionRecord("q", "s", (1, 0, 0), 1, 1.0, (0.5, 0.5))
    assert from_json(to_json(value), SupervisionRecord) == value
    with pytest.raises(InputValidationError):
        replace(value, **change)


def test_only_changed_supervision_contract_gets_new_version():
    assert record_version(SupervisionRecord) == "3"
    assert record_version(FeatureBatch) == "2"


def test_matrix_preserves_zero_masks_and_category_order():
    from coveo_recsys.ranking.matrix import feature_matrix

    batch = FeatureBatch(
        "q",
        ("a", "b"),
        ("x",),
        ("category",),
        ((0.0,), (2.0,)),
        ((0,), (4,)),
        ((False, False), (True, True)),
        "a" * 64,
    )
    result = feature_matrix(batch, ("x",), ("category",), "a" * 64)
    np.testing.assert_array_equal(result, [[0.0, 0.0], [2.0, 4.0]])
    assert result.dtype == np.float64 and result.flags.c_contiguous
    for changed in [
        replace(batch, categorical=((0,), (-1,))),
        replace(batch, numeric_columns=("y",)),
        replace(batch, feature_schema_sha256="b" * 64),
        replace(batch, categorical=((0,), (2**31,))),
    ]:
        with pytest.raises(InputValidationError):
            feature_matrix(changed, ("x",), ("category",), "a" * 64)
