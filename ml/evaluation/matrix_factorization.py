from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean, median, pstdev

import torch

from ml.baselines.popularity import (
    SIGNAL_CART,
    WeightedSignalConfig,
    build_popularity_scores,
    deterministic_ranking,
    load_train_interactions,
)
from ml.evaluation.metrics import evaluate_ranking
from ml.models.matrix_factorization import MatrixFactorization
from ml.training.mf_data import IndexedInteractions


TASKS = ("purchase", "viewplus", "favoriteplus")
K_VALUES = (5, 10, 20)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class EvaluationData:
    relevance: dict[str, dict[str, dict[str, list[str]]]]
    seen: dict[str, dict[str, list[str]]]
    candidates: dict[str, tuple[str, ...]]
    cart_rankings: dict[str, tuple[str, ...]]
    cart_scores: dict[str, float]


def load_evaluation_data(
    dataset_dir: str | Path,
    indexed: IndexedInteractions,
) -> EvaluationData:
    root = Path(dataset_dir)
    relevance = {
        split: {
            task: _read_json(root / f"{split}_relevance_{task}.json")["relevant_items_by_user"]
            for task in TASKS
        }
        for split in ("validation", "test")
    }
    seen = {
        task: _read_json(root / f"train_seen_items_{task}.json")["items_by_user"]
        for task in TASKS
    }
    policies = _read_json(root / "candidate_sets.json")["policies"]
    candidates = {task: tuple(policies[task]["product_ids"]) for task in TASKS}
    train_rows = load_train_interactions(root / "train_viewplus.csv")
    scores = build_popularity_scores(
        train_rows,
        indexed.item_ids,
        weighted_config=WeightedSignalConfig(),
    )[SIGNAL_CART]
    cart_rankings = {
        task: deterministic_ranking(scores, candidates[task]) for task in TASKS
    }
    return EvaluationData(relevance, seen, candidates, cart_rankings, scores)


def _rank_for_user(
    scores: torch.Tensor,
    candidate_ids: tuple[str, ...],
    item_to_index: dict[str, int],
    seen: set[str],
) -> list[str]:
    return sorted(
        (item for item in candidate_ids if item not in seen),
        key=lambda item: (-float(scores[item_to_index[item]]), item),
    )


def evaluate_model(
    model: MatrixFactorization,
    indexed: IndexedInteractions,
    evaluation: EvaluationData,
    *,
    split: str,
    k_values: tuple[int, ...] = K_VALUES,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    model.eval()
    with torch.no_grad():
        all_scores = model.score_all_items(torch.arange(len(indexed.user_ids))).cpu()
    rows = []
    purchase_top10: dict[str, list[str]] = {}
    for task in TASKS:
        relevance = evaluation.relevance[split][task]
        users = sorted(user for user, items in relevance.items() if items)
        per_k: dict[int, list[dict[str, float]]] = {k: [] for k in k_values}
        for user_id in users:
            user_index = indexed.user_to_index[user_id]
            seen = set(evaluation.seen[task].get(user_id, ()))
            if user_index in indexed.cold_user_indices:
                ranking = [item for item in evaluation.cart_rankings[task] if item not in seen]
            else:
                ranking = _rank_for_user(
                    all_scores[user_index],
                    evaluation.candidates[task],
                    indexed.item_to_index,
                    seen,
                )
            if task == "purchase":
                purchase_top10[user_id] = ranking[:10]
            for k in k_values:
                per_k[k].append(evaluate_ranking(ranking[:k], relevance[user_id], k=k))
        for k in k_values:
            values = per_k[k]
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "k": k,
                    "eligible_users": len(users),
                    "recall": fmean(value["recall"] for value in values),
                    "ndcg": fmean(value["ndcg"] for value in values),
                    "hit_rate": fmean(value["hit_rate"] for value in values),
                    "precision": fmean(value["precision"] for value in values),
                }
            )
    return rows, purchase_top10


class FinalTestEvaluator:
    def __init__(self) -> None:
        self._used = False

    def evaluate(
        self,
        models: dict[str, MatrixFactorization],
        indexed: IndexedInteractions,
        evaluation: EvaluationData,
    ) -> dict[str, tuple[list[dict[str, object]], dict[str, list[str]]]]:
        if self._used:
            raise RuntimeError("Test evaluation is allowed only once per experiment run")
        self._used = True
        return {
            name: evaluate_model(model, indexed, evaluation, split="test")
            for name, model in models.items()
        }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def model_diagnostics(
    model: MatrixFactorization,
    indexed: IndexedInteractions,
    evaluation: EvaluationData,
    purchase_top10: dict[str, list[str]],
) -> dict[str, object]:
    with torch.no_grad():
        user_norms = model.user_embeddings.weight.norm(dim=1).cpu().tolist()
        item_norms = model.item_embeddings.weight.norm(dim=1).cpu().tolist()
        scores = model.score_all_items(torch.arange(len(indexed.user_ids))).flatten().cpu().tolist()
    if not all(math.isfinite(value) for value in (*user_norms, *item_norms, *scores)):
        raise ValueError("Non-finite embedding diagnostic detected")
    users = sorted(purchase_top10)
    pair_overlaps = [
        len(set(purchase_top10[left]) & set(purchase_top10[right])) / 10
        for left, right in combinations(users, 2)
    ]
    cart_overlap = []
    recommended_cart_scores = []
    for user_id in users:
        seen = set(evaluation.seen["purchase"].get(user_id, ()))
        cart_top10 = [item for item in evaluation.cart_rankings["purchase"] if item not in seen][:10]
        cart_overlap.append(len(set(purchase_top10[user_id]) & set(cart_top10)) / 10)
        recommended_cart_scores.extend(evaluation.cart_scores[item] for item in purchase_top10[user_id])
    return {
        "user_embedding_norm": _summary(user_norms),
        "item_embedding_norm": _summary(item_norms),
        "score_distribution": _summary(scores),
        "all_values_finite": True,
        "unique_purchase_top10_lists": len({tuple(values) for values in purchase_top10.values()}),
        "purchase_evaluation_users": len(users),
        "average_pairwise_top10_overlap": fmean(pair_overlaps),
        "average_cart_popularity_top10_overlap": fmean(cart_overlap),
        "recommended_item_cart_score": _summary(recommended_cart_scores),
    }
