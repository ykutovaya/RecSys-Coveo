"""Свойства Э2-05: Hypothesis 6.131.9, 1000 случаев, seed 17, сжатие контрпримера."""

import numpy as np
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from coveo_recsys.contracts.historical import HistoricalCatalog, MetadataRecord
from coveo_recsys.historical.content import build_content
from tests.reference.e2_content_oracle import categories, vector
from tests.unit.test_e2_content import CONFIG, source

NORMATIVE = settings(max_examples=1000, deadline=None, database=None, print_blob=True)
ATOL, RTOL = 1e-10, 1e-8
COORDINATE = st.floats(-1e300, 1e300, allow_nan=False, allow_infinity=False)


@NORMATIVE
@seed(17)
@given(
    st.one_of(st.none(), st.lists(COORDINATE, min_size=2, max_size=2)),
    st.one_of(st.none(), st.lists(COORDINATE, min_size=3, max_size=3)),
)
def test_both_modalities_match_independent_decimal_norm(description, image):
    actual = build_content(
        HistoricalCatalog(("A",)), source([MetadataRecord("A", description, image)]), CONFIG
    )
    for raw, dimension, modality in (
        (description, 2, actual.description),
        (image, 3, actual.image),
    ):
        expected, mask, reason = vector(raw, dimension)
        np.testing.assert_allclose(modality.vectors[0], expected, atol=ATOL, rtol=RTOL)
        assert modality.mask.tolist() == [mask]
        assert modality.reasons == (reason,)


@NORMATIVE
@seed(17)
@given(
    st.lists(
        st.one_of(st.none(), st.text(alphabet="ABCабéΩ😀", min_size=1, max_size=5)),
        min_size=1,
        max_size=12,
    ),
    st.integers(0, 100000),
)
def test_dictionary_uses_only_catalog_and_ignores_metadata_permutation(labels, rotation):
    records = [
        MetadataRecord(f"sku{index:02}", category=label) for index, label in enumerate(labels)
    ]
    catalog = HistoricalCatalog(tuple(record.sku for record in records))
    expected_categories, expected_codes = categories(catalog.skus, records)
    records.append(MetadataRecord("outside", category="unseen-late"))
    offset = rotation % len(records)
    shuffled = records[offset:] + records[:offset]
    actual = build_content(catalog, source(shuffled), CONFIG)
    assert actual.categories == expected_categories
    assert actual.category_codes.tolist() == expected_codes
