"""Прямая сумма пар при фиксированных рангах; без кода ранжировщика."""

import math


def frozen_pairs(scores, labels, group_sizes, skus):
    pairs = []
    offset = 0
    for size in group_sizes:
        indices = list(range(offset, offset + size))
        order = sorted(indices, key=lambda index: (-scores[index], skus[index].encode("utf-8")))
        discount = {
            index: 1.0 / math.log2(rank + 1) if rank <= 20 else 0.0
            for rank, index in enumerate(order, start=1)
        }
        positive = [index for index in indices if labels[index] == 1]
        negative = [index for index in indices if labels[index] == 0]
        normalizer = math.fsum(
            1.0 / math.log2(rank + 1) for rank in range(1, min(20, len(positive)) + 1)
        )
        if positive and negative:
            pair_weight = 1.0 / (len(positive) * len(negative))
            for i in positive:
                for j in negative:
                    weight = pair_weight * abs(discount[i] - discount[j]) / normalizer
                    pairs.append((i, j, weight))
        offset += size
    return tuple(pairs)


def pair_loss(scores, pairs):
    terms = []
    for i, j, weight in pairs:
        argument = -(float(scores[i]) - float(scores[j]))
        softplus = max(argument, 0.0) + math.log1p(math.exp(-abs(argument)))
        terms.append(weight * softplus)
    return math.fsum(terms)


def direct_derivatives(scores, labels, group_sizes, skus):
    gradients = [[] for _ in scores]
    hessians = [[] for _ in scores]
    for i, j, weight in frozen_pairs(scores, labels, group_sizes, skus):
        difference = float(scores[i]) - float(scores[j])
        if difference >= 0:
            odds = math.exp(-difference)
            probability = odds / (1.0 + odds)
        else:
            probability = 1.0 / (1.0 + math.exp(difference))
        # За границей cosh поправка к exp(-|d|) уже меньше точности FP64.
        curvature = (
            math.exp(-abs(difference))
            if abs(difference) > 700
            else 1.0 / (2.0 + 2.0 * math.cosh(difference))
        )
        gradients[i].append(-weight * probability)
        gradients[j].append(weight * probability)
        hessians[i].append(weight * curvature)
        hessians[j].append(weight * curvature)
    return [math.fsum(values) for values in gradients], [math.fsum(values) for values in hessians]
