from __future__ import annotations

from statistics import fmean

from ml.evaluation.matrix_factorization import EvaluationData
from ml.evaluation.metrics import evaluate_ranking
from ml.training.mf_data import IndexedInteractions


def fallback_group_diagnostic(
    indexed: IndexedInteractions,
    evaluation: EvaluationData,
    purchase_top10: dict[str, list[str]],
) -> list[dict[str, object]]:
    relevance = evaluation.relevance["test"]["purchase"]
    eligible = sorted(user for user, items in relevance.items() if items)
    cold = {indexed.user_ids[index] for index in indexed.cold_user_indices}
    groups = {
        "all": eligible,
        "learned": [user for user in eligible if user not in cold],
        "fallback": [user for user in eligible if user in cold],
    }
    rows = []
    for group, users in groups.items():
        values = [evaluate_ranking(purchase_top10[user], relevance[user], k=10) for user in users]
        rows.append({
            "group": group,
            "user_count": len(users),
            "eligible_share": len(users) / len(eligible),
            "recall_at_10": fmean(value["recall"] for value in values) if values else None,
            "ndcg_at_10": fmean(value["ndcg"] for value in values) if values else None,
            "hit_rate_at_10": fmean(value["hit_rate"] for value in values) if values else None,
            "precision_at_10": fmean(value["precision"] for value in values) if values else None,
        })
    return rows


def signal_alignment(indexed: IndexedInteractions, evaluation: EvaluationData) -> dict[str, object]:
    relevance = evaluation.relevance["test"]["purchase"]
    eligible = sorted(user for user, items in relevance.items() if items)
    positives: dict[str, set[str]] = {user: set() for user in eligible}
    for user_index, item_index in indexed.positive_pairs:
        user = indexed.user_ids[user_index]
        if user in positives:
            positives[user].add(indexed.item_ids[item_index])
    overlaps = [len(positives[user] & set(relevance[user])) for user in eligible]
    return {
        "test_purchase_eligible_users": len(eligible),
        "users_with_train_positive": sum(bool(positives[user]) for user in eligible),
        "users_without_train_positive": sum(not positives[user] for user in eligible),
        "users_with_exact_item_continuity": sum(value > 0 for value in overlaps),
        "exact_item_continuity_user_rate": sum(value > 0 for value in overlaps) / len(eligible),
        "mean_exact_item_overlap_count": fmean(overlaps),
    }
