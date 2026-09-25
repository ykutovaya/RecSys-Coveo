from hypothesis import given, seed
from hypothesis import strategies as st

from coveo_recsys.data.periods import admit, sample_sessions
from tests.reference.e1_periods_oracle import admitted, key
from tests.unit.test_e1_periods import CONFIG, PERIODS


@seed(17)
@given(st.integers(0, 699), st.booleans(), st.booleans(), st.integers(0, 99))
def test_admission_matches_independent_formula(t0, coverage, complete, delta):
    close = t0 + delta
    period = PERIODS[t0 // 100]
    result = admit(
        t0, PERIODS, CONFIG, coverage_complete=coverage, session_complete=complete, close_ms=close
    )
    assert result.eligible
    assert result.admitted == admitted(t0, period, 20, 10, coverage, complete, close)


@seed(17)
@given(
    st.sets(st.text(alphabet="abé商品012", min_size=1, max_size=10), min_size=1, max_size=20),
    st.integers(0, 25),
)
def test_sampling_permutation_and_nesting(ids, n):
    ids = sorted(ids)
    expected = tuple(sorted(ids, key=key))
    assert sample_sessions(ids, n) == expected[:n]
    assert sample_sessions(reversed(ids), n) == expected[:n]
    assert sample_sessions(ids, n + 1)[:n] == expected[:n]
