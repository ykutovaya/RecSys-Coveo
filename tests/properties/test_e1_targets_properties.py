from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.contracts.preparation import Coverage
from tests.unit.test_e1_cohort import cohort_case
from tests.unit.test_e1_targets import assert_oracle


@seed(17)
@given(
    t0=st.integers(-20, 720),
    offsets=st.lists(st.integers(1, 40), max_size=8),
    bought=st.booleans(),
    confirmed=st.booleans(),
    extent=st.integers(0, 60),
    gap_start=st.integers(0, 60),
    gap_length=st.integers(0, 5),
)
def test_target_window_and_maturity_against_independent_oracle(
    t0,
    offsets,
    bought,
    confirmed,
    extent,
    gap_start,
    gap_length,
):
    rows = [(t0, "detail", "A")]
    rows.extend(
        (t0 + offset, "purchase" if bought else "search", "B" if bought else None)
        for offset in offsets
    )
    gaps = ((t0 + gap_start, t0 + gap_start + gap_length),) if gap_length else ()
    # Все разрывы входят в объявленный диапазон; короткое покрытие проверяется отдельно.
    end = max(t0 + extent + 1, t0 + gap_start + gap_length)
    coverage = Coverage(
        "n", t0 - 1, end, gaps, "synthetic property proof", end if confirmed else None
    )
    assert_oracle(rows, coverage)


@seed(17)
@given(
    skus=st.lists(st.sampled_from(["A", "B", "é", "é", "商品"]), min_size=1, max_size=30),
    purchase_time=st.integers(201, 240),
    copies=st.integers(1, 5),
)
def test_purchase_multiplicity_and_row_order_do_not_change_targets(skus, purchase_time, copies):
    rows = [(200, "detail", "view")] + [(purchase_time, "purchase", sku) for sku in skus]
    base = assert_oracle(rows)
    changed = assert_oracle([(200, "detail", "view")] + list(reversed(rows[1:])) * copies)
    assert base == changed


@seed(17)
@given(
    anchor=st.integers(1, 80),
    purchase=st.integers(0, 90),
    repeat=st.integers(1, 90),
    copies=st.integers(1, 15),
    coverage_end=st.integers(30, 120),
)
def test_first_view_cohort_against_independent_formula(
    anchor, purchase, repeat, copies, coverage_end
):
    rows = [
        (0, "detail", "seed"),
        (anchor, "detail", "A"),
        (max(anchor, repeat), "detail", "A"),
    ] * copies + [(purchase, "purchase", "A")] * copies
    coverage = Coverage("n", 0, coverage_end, (), "synthetic cohort proof", coverage_end)
    cohort_case(rows, coverage=coverage)
