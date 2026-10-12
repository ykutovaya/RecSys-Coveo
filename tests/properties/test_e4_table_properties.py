"""1000 случаев, seed 17: состав и порядок таблицы не зависят от порядка чтения."""

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from tests.reference.e4_table_oracle import table_oracle
from tests.unit.test_e4_table import build, example

NORMATIVE = settings(max_examples=1000, deadline=None, database=None, print_blob=True)
SKU = ("a", "z", "é", "я", "🙂")


@NORMATIVE
@seed(17)
@given(
    st.lists(
        st.tuples(
            st.sets(st.sampled_from(SKU)),
            st.sets(st.sampled_from((*SKU, "outside"))),
        ),
        min_size=0,
        max_size=5,
    ),
)
def test_training_rows_match_direct_join_and_permutation_invariance(groups):
    examples = [
        example(
            f"session-{i}",
            tuple(sorted(candidates, key=lambda x: x.encode("utf-8"))),
            tuple(sorted(targets, key=lambda x: x.encode("utf-8"))),
            time=2000 + i,
        )
        for i, (candidates, targets) in enumerate(groups)
    ]
    actual = build(reversed(examples))
    expected = table_oracle(examples)
    assert actual.labels.tolist() == expected["labels"]
    assert list(actual.group_sizes) == expected["group_sizes"]
    assert list(actual.skus) == expected["skus"]
    assert list(actual.group_keys) == expected["group_keys"]
    np.testing.assert_array_equal(actual.data, np.array(expected["data"]).reshape(-1, 3))
    assert {key: actual.statistics[key] for key in expected["stats"]} == expected["stats"]
    direct = build(examples)
    np.testing.assert_array_equal(actual.data, direct.data)
    assert actual.statistics == direct.statistics
    assert actual.supervision == direct.supervision
