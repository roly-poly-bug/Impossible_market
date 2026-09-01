from __future__ import annotations

import math
from itertools import combinations
from statistics import fmean, pstdev

import torch

from ml.evaluation.matrix_factorization import EvaluationData, K_VALUES, TASKS
from ml.evaluation.metrics import evaluate_ranking
from ml.training.mf_data import IndexedInteractions


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def zscore(values: torch.Tensor) -> torch.Tensor:
    values = values.to(dtype=torch.float64)
    std = values.std(unbiased=False)
    if not torch.isfinite(std) or float(std) == 0.0:
        return torch.zeros_like(values)
    normalized = (values - values.mean()) / std
    if not torch.isfinite(normalized).all():
        raise ValueError("Non-finite z-score")
    return normalized


def hybrid_score(mf_z: torch.Tensor, cart_z: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if mf_z.shape != cart_z.shape:
        raise ValueError("MF and Cart score shapes must match")
    return alpha * mf_z + (1.0 - alpha) * cart_z


def _rank(scores: torch.Tensor, item_ids: tuple[str, ...], seen: set[str]) -> list[str]:
    return sorted(
        (item for item in item_ids if item not in seen),
        key=lambda item: (-float(scores[item_ids.index(item)]), item),
    )


class HybridRanker:
    def __init__(self, model, indexed: IndexedInteractions, evaluation: EvaluationData) -> None:
        model.eval()
        with torch.no_grad():
            self.mf_scores = model.score_all_items(torch.arange(len(indexed.user_ids))).cpu().to(torch.float64)
        self.indexed = indexed
        self.evaluation = evaluation

    def score_components(self, user_id: str, task: str) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
        candidates = self.evaluation.candidates[task]
        indices = torch.tensor([self.indexed.item_to_index[item] for item in candidates])
        user = self.indexed.user_to_index[user_id]
        mf_z = zscore(self.mf_scores[user, indices])
        cart_z = zscore(torch.tensor([self.evaluation.cart_scores[item] for item in candidates]))
        return candidates, mf_z, cart_z

    def ranking(self, user_id: str, task: str, alpha: float) -> list[str]:
        seen = set(self.evaluation.seen[task].get(user_id, ()))
        user = self.indexed.user_to_index[user_id]
        if user in self.indexed.cold_user_indices:
            return [item for item in self.evaluation.cart_rankings[task] if item not in seen]
        candidates, mf_z, cart_z = self.score_components(user_id, task)
        return _rank(hybrid_score(mf_z, cart_z, alpha), candidates, seen)


def evaluate_hybrid(
    ranker: HybridRanker,
    *,
    split: str,
    alpha: float,
    tasks: tuple[str, ...] = TASKS,
    k_values: tuple[int, ...] = K_VALUES,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    rows: list[dict[str, object]] = []
    purchase_top10: dict[str, list[str]] = {}
    for task in tasks:
        relevance = ranker.evaluation.relevance[split][task]
        users = sorted(user for user, items in relevance.items() if items)
        per_k = {k: [] for k in k_values}
        for user in users:
            ranking = ranker.ranking(user, task, alpha)
            if task == "purchase":
                purchase_top10[user] = ranking[:10]
            for k in k_values:
                per_k[k].append(evaluate_ranking(ranking[:k], relevance[user], k=k))
        for k in k_values:
            values = per_k[k]
            rows.append({
                "task": task, "split": split, "alpha": alpha, "k": k,
                "eligible_users": len(users),
                "recall": fmean(value["recall"] for value in values),
                "ndcg": fmean(value["ndcg"] for value in values),
                "hit_rate": fmean(value["hit_rate"] for value in values),
                "precision": fmean(value["precision"] for value in values),
            })
    return rows, purchase_top10


def select_alpha(rows: list[dict[str, object]]) -> float:
    candidates = [row for row in rows if row["task"] == "purchase" and row["split"] == "validation" and row["k"] == 10]
    if {float(row["alpha"]) for row in candidates} != set(ALPHAS):
        raise ValueError("Validation selection requires the fixed alpha grid")
    return float(max(candidates, key=lambda row: (float(row["ndcg"]), float(row["recall"]), float(row["alpha"])))["alpha"])


class FinalHybridTestEvaluator:
    def __init__(self) -> None:
        self._used = False

    def evaluate(self, ranker: HybridRanker, alpha: float):
        if self._used:
            raise RuntimeError("Hybrid Test evaluation is allowed only once")
        self._used = True
        return evaluate_hybrid(ranker, split="test", alpha=alpha)


def personalization_diagnostics(rankings: dict[str, list[str]], evaluation: EvaluationData) -> dict[str, object]:
    users = sorted(rankings)
    overlaps = [len(set(rankings[a]) & set(rankings[b])) / 10 for a, b in combinations(users, 2)]
    cart_overlaps, cart_counts = [], []
    for user in users:
        seen = set(evaluation.seen["purchase"].get(user, ()))
        cart = [item for item in evaluation.cart_rankings["purchase"] if item not in seen][:10]
        cart_overlaps.append(len(set(rankings[user]) & set(cart)) / 10)
        cart_counts.extend(evaluation.cart_scores[item] for item in rankings[user])
    return {
        "unique_purchase_top10_lists": len({tuple(values) for values in rankings.values()}),
        "average_pairwise_top10_overlap": fmean(overlaps),
        "average_cart_popularity_top10_overlap": fmean(cart_overlaps),
        "recommended_item_cart_score_mean": fmean(cart_counts),
    }


def contribution_diagnostics(ranker: HybridRanker, alpha: float) -> list[dict[str, object]]:
    mf_values, cart_values, total_values = [], [], []
    relevance = ranker.evaluation.relevance["test"]["purchase"]
    for user in sorted(user for user, items in relevance.items() if items):
        if ranker.indexed.user_to_index[user] in ranker.indexed.cold_user_indices:
            continue
        _, mf_z, cart_z = ranker.score_components(user, "purchase")
        mf = alpha * mf_z
        cart = (1.0 - alpha) * cart_z
        mf_values.extend(mf.tolist()); cart_values.extend(cart.tolist()); total_values.extend((mf + cart).tolist())
    rows = []
    for name, values in (("mf", mf_values), ("cart_popularity", cart_values), ("hybrid_total", total_values)):
        rows.append({"component": name, "mean": fmean(values), "std": pstdev(values), "variance": pstdev(values) ** 2, "min": min(values), "max": max(values), "finite": all(math.isfinite(value) for value in values)})
    return rows
