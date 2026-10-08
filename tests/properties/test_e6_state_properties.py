"""Независимые свойства транспортного журнала: 1000 случаев, seed 17."""

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from tests.unit.test_e6_state import accept, packet, service


@st.composite
def histories(draw):
    offset = draw(st.integers(1000, 1_000_000))
    actions = draw(
        st.lists(
            st.tuples(st.sampled_from(["add", "remove"]), st.sampled_from(["A", "B", "C"])),
            max_size=5,
        )
    )
    rows = [*actions, ("detail", "D")]
    order = draw(st.permutations(tuple(range(len(rows)))))
    return offset, rows, order


@seed(17)
@settings(max_examples=1000, deadline=None, database=None)
@given(histories())
def test_late_arrival_prefix_and_exact_repeats_match_literal_history(case):
    offset, rows, order = case
    app = service()
    now = offset + 1000
    for index in order:
        kind, sku = rows[index]
        wire = packet(offset + index, block=f"b{index}", events=[(f"e{index}", kind, sku)])
        assert accept(app, wire, now=now).status == "accepted"
        snapshot = app.store.read("n", "s")
        assert accept(app, wire, now=now).status == "duplicate"
        assert app.store.read("n", "s") == snapshot
        now += 1
    expected_cart = {}
    for index, (kind, sku) in enumerate(rows[:-1]):
        expected_cart[sku] = kind, offset + index
    last = f"b{len(rows) - 1}"
    request = app.prefix("n", "s", last, a_q=now, check_deadline=lambda: None)
    assert {
        item.sku: (item.state.value, item.updated_at_ms) for item in request.prefix.cart
    } == expected_cart
    assert request.prefix.purchased_skus == ()
    assert sum(value for _, _, value in request.prefix.counters) == len(rows)
    assert request.prefix.detail_history[0].sku == "D"
    assert request.t_q == offset + len(rows) - 1 and request.a_q == now
    assert app.first_trigger("n", "s", last, a_q=now).canonical_block_id == request.trigger_block_id
