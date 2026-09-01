from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import torch

from ml.evaluation.matrix_factorization import load_evaluation_data
from ml.evaluation.mf_bias import _correlations
from ml.evaluation.mf_cart_hybrid import ALPHAS, FinalHybridTestEvaluator, HybridRanker, contribution_diagnostics, evaluate_hybrid, personalization_diagnostics, select_alpha
from ml.evaluation.mf_cart_signal import fallback_group_diagnostic
from ml.evaluation.mf_latent_dim import history_group_metrics, item_popularity_group_metrics
from ml.experiments.mf_negative_sampling_experiment import _cart_comparison
from ml.models.mf_bias import BiasedMatrixFactorization
from ml.representations.mf_cart_signal import EXISTING_WEIGHTED, load_cart_signal


EXPERIMENT_VERSION = "mf_cart_hybrid_v1"


@dataclass
class HybridExperimentResult:
    data: object
    evaluation: object
    validation: list[dict[str, object]]
    best_alpha: float
    test_metrics: list[dict[str, object]]
    rankings: dict[str, list[str]]
    comparison: list[dict[str, object]]
    personalization: list[dict[str, object]]
    contributions: list[dict[str, object]]
    history_groups: list[dict[str, object]]
    popularity_groups: list[dict[str, object]]
    fallback: list[dict[str, object]]
    bias_cart_correlation: dict[str, float]
    evaluation_runtime_seconds: float
    additional_peak_memory_bytes: int


def _load_model(path: Path, users: int, items: int):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = BiasedMatrixFactorization(users, items, 8, "item_bias")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _saved_personalization(path: Path) -> dict[str, object]:
    import csv
    with path.open(encoding="utf-8", newline="") as source:
        row = next(csv.DictReader(source))
    return {key: (int(value) if key == "unique_purchase_top10_lists" else float(value)) for key, value in row.items() if key != "objective"}


def run_hybrid_experiment(dataset_dir, popularity_dir, latent_results_dir, objective_results_dir):
    data, _ = load_cart_signal(dataset_dir, EXISTING_WEIGHTED, sample_ratio=4, seed=42)
    evaluation = load_evaluation_data(dataset_dir, data.indexed)
    model = _load_model(Path(latent_results_dir) / "checkpoints" / "dim8_best.pt", len(data.indexed.user_ids), len(data.indexed.item_ids))
    ranker = HybridRanker(model, data.indexed, evaluation)

    validation = []
    for alpha in ALPHAS:
        rows, _ = evaluate_hybrid(ranker, split="validation", alpha=alpha, tasks=("purchase",), k_values=(10,))
        validation.extend(rows)
    best_alpha = select_alpha(validation)

    tracemalloc.start()
    started = time.perf_counter()
    test_metrics, rankings = FinalHybridTestEvaluator().evaluate(ranker, best_alpha)
    runtime = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    hybrid_diag = personalization_diagnostics(rankings, evaluation)
    mf_diag = _saved_personalization(Path(objective_results_dir) / "personalization.csv")
    cart_rankings = {}
    for user, items in evaluation.relevance["test"]["purchase"].items():
        if items:
            seen = set(evaluation.seen["purchase"].get(user, ()))
            cart_rankings[user] = [item for item in evaluation.cart_rankings["purchase"] if item not in seen][:10]
    personalization = [
        {"model": "cart_popularity", **personalization_diagnostics(cart_rankings, evaluation)},
        {"model": "best_mf", **mf_diag},
        {"model": "best_hybrid", **hybrid_diag},
    ]

    with torch.no_grad():
        bias = model.item_bias.weight.squeeze(-1).cpu().tolist()
    cart = [evaluation.cart_scores[item] for item in data.indexed.item_ids]
    bias_cart = _correlations(bias, cart)

    cart_metrics = _cart_comparison(Path(popularity_dir))
    comparison = []
    for k in (5, 10, 20):
        base = cart_metrics[k]
        comparison.append({"model": "cart_popularity", "k": k, **{name: int(base[name]) if name == "eligible_users" else float(base[name]) for name in ("eligible_users", "recall", "ndcg", "hit_rate", "precision")}})
        import csv
        with (Path(latent_results_dir) / "dim8" / "metrics.csv").open(encoding="utf-8", newline="") as source:
            mf = next(row for row in csv.DictReader(source) if row["task"] == "purchase" and row["split"] == "test" and int(row["k"]) == k)
        comparison.append({"model": "best_mf", "k": k, **{name: int(mf[name]) if name == "eligible_users" else float(mf[name]) for name in ("eligible_users", "recall", "ndcg", "hit_rate", "precision")}})
        hybrid = next(row for row in test_metrics if row["task"] == "purchase" and row["k"] == k)
        comparison.append({"model": "best_hybrid", **{name: hybrid[name] for name in ("k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")}})

    return HybridExperimentResult(
        data, evaluation, validation, best_alpha, test_metrics, rankings, comparison, personalization,
        contribution_diagnostics(ranker, best_alpha),
        [{"model": "best_hybrid", **row} for row in history_group_metrics(data.indexed, evaluation, rankings)],
        [{"model": "best_hybrid", **row} for row in item_popularity_group_metrics(evaluation, rankings)],
        [{"model": "best_hybrid", **row} for row in fallback_group_diagnostic(data.indexed, evaluation, rankings)],
        bias_cart, runtime, peak,
    )
