"""Независимый эталон: множества пар и точная рациональная популярность."""

from fractions import Fraction


def statistics(rows, skus, eta=1.0):
    details = {(sid, sku) for sid, _, action, sku in rows if action == "detail"}
    purchases = {(sid, sku) for sid, _, action, sku in rows if action == "purchase"}
    detail_counts = [sum(item_sku == sku for _, item_sku in details) for sku in skus]
    purchase_counts = [sum(item_sku == sku for _, item_sku in purchases) for sku in skus]
    masses = [Fraction(value) + Fraction.from_float(float(eta)) for value in detail_counts]
    total = sum(masses, Fraction())
    popularity = [float(value / total) for value in masses]
    return detail_counts, purchase_counts, popularity, len({row[0] for row in rows})
