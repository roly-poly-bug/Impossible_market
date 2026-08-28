from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _unique_top_k(ranking: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    unique = []
    seen = set()
    for item in ranking:
        if item not in seen:
            unique.append(item)
            seen.add(item)
        if len(unique) == k:
            break
    return unique


def recall_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_items = set(relevant)
    if not relevant_items:
        raise ValueError("Recall is undefined when the relevance set is empty")
    recommended = set(_unique_top_k(ranking, k))
    return len(recommended & relevant_items) / len(relevant_items)


def precision_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_items = set(relevant)
    recommended = _unique_top_k(ranking, k)
    return sum(item in relevant_items for item in recommended) / k


def hit_rate_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_items = set(relevant)
    recommended = _unique_top_k(ranking, k)
    return float(any(item in relevant_items for item in recommended))


def ndcg_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_items = set(relevant)
    if not relevant_items:
        raise ValueError("NDCG is undefined when the relevance set is empty")
    recommended = _unique_top_k(ranking, k)
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, item in enumerate(recommended)
        if item in relevant_items
    )
    ideal_count = min(k, len(relevant_items))
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal_dcg


def evaluate_ranking(
    ranking: Sequence[str],
    relevant: Iterable[str],
    *,
    k: int,
) -> dict[str, float]:
    relevant_items = set(relevant)
    return {
        "recall": recall_at_k(ranking, relevant_items, k),
        "ndcg": ndcg_at_k(ranking, relevant_items, k),
        "hit_rate": hit_rate_at_k(ranking, relevant_items, k),
        "precision": precision_at_k(ranking, relevant_items, k),
    }
