"""Малый независимый эталон канонических кратностей, без производственного кода."""


def expected_counts(rows):
    seen = {}
    counts = {}
    for row in rows:
        signature = (
            row.source_namespace,
            row.session_id,
            row.event_time_ms,
            row.event_type.value,
            row.sku,
        )
        if row.transport_id is not None:
            identity = (row.source_namespace, row.transport_id)
            if identity in seen:
                if seen[identity] != signature:
                    raise ValueError("conflict")
                continue
            seen[identity] = signature
        counts[signature] = counts.get(signature, 0) + 1
    return counts
