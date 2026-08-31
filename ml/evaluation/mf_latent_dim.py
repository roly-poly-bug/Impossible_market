from __future__ import annotations

import math
from collections import Counter
from statistics import fmean

from ml.evaluation.matrix_factorization import EvaluationData
from ml.evaluation.metrics import evaluate_ranking
from ml.training.mf_data import IndexedInteractions


def parameter_counts(model) -> dict[str, int]:
    user = model.user_embeddings.weight.numel()
    item = model.item_embeddings.weight.numel()
    bias = model.item_bias.weight.numel()
    return {
        "total_trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "user_embedding_parameters": user,
        "item_embedding_parameters": item,
        "item_bias_parameters": bias,
    }


def _group_metrics(users, rankings, relevance) -> dict[str, object]:
    eligible = [user for user in users if relevance.get(user)]
    values = [evaluate_ranking(rankings[user], relevance[user], k=10) for user in eligible]
    return {
        "eligible_users": len(eligible),
        "recall_at_10": fmean(value["recall"] for value in values) if values else None,
        "ndcg_at_10": fmean(value["ndcg"] for value in values) if values else None,
        "hit_rate_at_10": fmean(value["hit_rate"] for value in values) if values else None,
        "precision_at_10": fmean(value["precision"] for value in values) if values else None,
    }


def history_group_metrics(indexed: IndexedInteractions, evaluation: EvaluationData, rankings: dict[str, list[str]]) -> list[dict[str, object]]:
    counts = Counter(indexed.user_ids[user] for user, _ in indexed.positive_pairs)
    eligible = sorted(user for user, items in evaluation.relevance["test"]["purchase"].items() if items)
    ordered = sorted(eligible, key=lambda user: (counts[user], user))
    groups = {
        "low": ordered[: len(ordered) // 3],
        "medium": ordered[len(ordered) // 3 : 2 * len(ordered) // 3],
        "high": ordered[2 * len(ordered) // 3 :],
    }
    rows = []
    for name, users in groups.items():
        metrics = _group_metrics(users, rankings, evaluation.relevance["test"]["purchase"])
        rows.append({
            "history_group": name,
            "min_train_positive_pairs": min(counts[user] for user in users),
            "max_train_positive_pairs": max(counts[user] for user in users),
            "mean_train_positive_pairs": fmean(counts[user] for user in users),
            **metrics,
        })
    return rows


def item_popularity_group_metrics(evaluation: EvaluationData, rankings: dict[str, list[str]]) -> list[dict[str, object]]:
    candidates = sorted(evaluation.candidates["purchase"], key=lambda item: (evaluation.cart_scores[item], item))
    groups = {
        "low": set(candidates[: len(candidates) // 3]),
        "medium": set(candidates[len(candidates) // 3 : 2 * len(candidates) // 3]),
        "high": set(candidates[2 * len(candidates) // 3 :]),
    }
    relevance = evaluation.relevance["test"]["purchase"]
    rows = []
    for name, items in groups.items():
        filtered = {user: [item for item in values if item in items] for user, values in relevance.items()}
        users = sorted(user for user, values in filtered.items() if values)
        metrics = _group_metrics(users, rankings, filtered)
        rows.append({
            "popularity_group": name,
            "item_count": len(items),
            "min_train_cart_count": min(evaluation.cart_scores[item] for item in items),
            "max_train_cart_count": max(evaluation.cart_scores[item] for item in items),
            "mean_train_cart_count": fmean(evaluation.cart_scores[item] for item in items),
            **metrics,
        })
    return rows


def normalized_norm(mean_norm: float, latent_dim: int) -> float:
    return mean_norm / math.sqrt(latent_dim)
