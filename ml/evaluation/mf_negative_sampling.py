from __future__ import annotations

from statistics import fmean

from ml.evaluation.matrix_factorization import EvaluationData
from ml.evaluation.metrics import evaluate_ranking
from ml.training.mf_data import IndexedInteractions


def purchase_group_performance(
    indexed: IndexedInteractions,
    evaluation: EvaluationData,
    purchase_top10: dict[str, list[str]],
    group_user_indices: set[int] | frozenset[int],
) -> dict[str, dict[str, float | int | None]]:
    relevance = evaluation.relevance["test"]["purchase"]
    eligible = sorted(user for user, items in relevance.items() if items)
    group = [user for user in eligible if indexed.user_to_index[user] in group_user_indices]
    other = [user for user in eligible if indexed.user_to_index[user] not in group_user_indices]

    def summarize(users: list[str]) -> dict[str, float | int | None]:
        values = [evaluate_ranking(purchase_top10[user], relevance[user], k=10) for user in users]
        return {
            "user_count": len(users),
            "recall_at_10": fmean(value["recall"] for value in values) if values else None,
            "ndcg_at_10": fmean(value["ndcg"] for value in values) if values else None,
            "hit_rate_at_10": fmean(value["hit_rate"] for value in values) if values else None,
            "precision_at_10": fmean(value["precision"] for value in values) if values else None,
        }

    return {"backfill_users": summarize(group), "no_backfill_users": summarize(other)}
