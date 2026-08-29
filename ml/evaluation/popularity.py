from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean

from ml.baselines.popularity import recommend_from_global_ranking
from ml.evaluation.metrics import evaluate_ranking


DEFAULT_K_VALUES = (5, 10, 20)


def eligible_user_ids(
    relevance_by_user: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    return tuple(
        sorted(user_id for user_id, items in relevance_by_user.items() if items)
    )


def evaluate_popularity_ranking(
    ranking: Sequence[str],
    relevance_by_user: Mapping[str, Sequence[str]],
    seen_items_by_user: Mapping[str, Sequence[str]],
    *,
    exclude_seen: bool,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> list[dict[str, float | int]]:
    users = eligible_user_ids(relevance_by_user)
    if not users:
        raise ValueError("Popularity evaluation requires at least one eligible User")
    results = []
    for k in k_values:
        per_user = []
        for user_id in users:
            recommendations = recommend_from_global_ranking(
                ranking,
                seen_items=seen_items_by_user.get(user_id, ()),
                exclude_seen=exclude_seen,
                k=k,
            )
            per_user.append(
                evaluate_ranking(recommendations, relevance_by_user[user_id], k=k)
            )
        results.append(
            {
                "k": k,
                "eligible_users": len(users),
                "recall": fmean(values["recall"] for values in per_user),
                "ndcg": fmean(values["ndcg"] for values in per_user),
                "hit_rate": fmean(values["hit_rate"] for values in per_user),
                "precision": fmean(values["precision"] for values in per_user),
            }
        )
    return results
