from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

from ml.evaluation.matrix_factorization import evaluate_model, load_evaluation_data, model_diagnostics
from ml.evaluation.mf_negative_sampling import purchase_group_performance
from ml.evaluation.mf_signal import SignalFinalTestEvaluator
from ml.representations.mf_signal import WEIGHTED, RepresentationData, load_representation
from ml.training.mf_negative_sampling import (
    EXPOSED_NON_CONVERSION,
    MIXED,
    RANDOM_UNKNOWN,
    STRATEGIES,
    NegativeSamplingResult,
    hardness_diagnostics,
    sample_non_positives,
    sampling_statistics,
)
from ml.training.mf_signal_trainer import SignalTrainingResult, train_signal_representation
from ml.training.mf_trainer import MFTrainingConfig


EXPERIMENT_VERSION = "mf_negative_sampling_v1"
TRAINED_STRATEGIES = (EXPOSED_NON_CONVERSION, MIXED)


@dataclass
class NegativeSamplingExperimentResult:
    config: MFTrainingConfig
    weighted_data: RepresentationData
    sampling: dict[str, NegativeSamplingResult]
    training: dict[str, SignalTrainingResult]
    histories: dict[str, list[dict[str, object]]]
    metrics: dict[str, list[dict[str, object]]]
    diagnostics: dict[str, dict[str, object]]
    comparison: list[dict[str, object]]
    sampling_stats: list[dict[str, object]]
    hardness: list[dict[str, object]]
    personalization: list[dict[str, object]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _frozen_random(weighted_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    history = [{key: float(value) for key, value in row.items()} for row in _read_csv(weighted_dir / "training_history.csv")]
    metrics: list[dict[str, object]] = []
    for row in _read_csv(weighted_dir / "metrics.csv"):
        metrics.append(
            {
                "task": row["task"], "split": row["split"], "k": int(row["k"]),
                "eligible_users": int(row["eligible_users"]), "recall": float(row["recall"]),
                "ndcg": float(row["ndcg"]), "hit_rate": float(row["hit_rate"]),
                "precision": float(row["precision"]),
            }
        )
    diagnostics = json.loads((weighted_dir / "diagnostics.json").read_text(encoding="utf-8"))
    diagnostics["source"] = "reused frozen Weighted Random Unknown result; Test not re-evaluated"
    return history, metrics, diagnostics


def _item_observed_counts(dataset_dir: Path, data: RepresentationData) -> tuple[dict[int, float], dict[int, float]]:
    cart = {index: 0.0 for index in range(len(data.indexed.item_ids))}
    exposure = {index: 0.0 for index in range(len(data.indexed.item_ids))}
    for row in _read_csv(dataset_dir / "train_viewplus.csv"):
        item = data.indexed.item_to_index[row["product_id"]]
        cart[item] += float(row["cart_count"])
        exposure[item] += float(row["impression_count"])
    return cart, exposure


def _cart_comparison(popularity_dir: Path) -> dict[int, dict[str, str]]:
    return {
        int(row["k"]): row for row in _read_csv(popularity_dir / "metrics.csv")
        if row["popularity_signal"] == "cart_count"
        and row["evaluation_task"] == "purchase" and row["split"] == "test"
    }


def run_negative_sampling_experiment(
    dataset_dir: str | Path,
    popularity_dir: str | Path,
    signal_results_dir: str | Path,
    *,
    config: MFTrainingConfig | None = None,
) -> NegativeSamplingExperimentResult:
    settings = config or MFTrainingConfig()
    root = Path(dataset_dir)
    weighted = load_representation(
        root, WEIGHTED, negative_ratio=settings.negative_ratio, seed=settings.seed
    )
    sampling = {
        name: sample_non_positives(
            weighted.indexed, name, sample_ratio=settings.negative_ratio, seed=settings.seed
        )
        for name in STRATEGIES
    }
    if sampling[RANDOM_UNKNOWN].sampled.triples != weighted.sampled.triples:
        raise AssertionError("Random Unknown control samples changed from frozen Weighted v1")
    evaluation = load_evaluation_data(root, weighted.indexed)
    training: dict[str, SignalTrainingResult] = {}
    validation: dict[str, list[dict[str, object]]] = {}
    for strategy in TRAINED_STRATEGIES:
        strategy_data = replace(weighted, sampled=sampling[strategy].sampled)
        training[strategy] = train_signal_representation(strategy_data, evaluation, settings)
        validation[strategy] = evaluate_model(
            training[strategy].result.model, weighted.indexed, evaluation, split="validation"
        )[0]
    test = SignalFinalTestEvaluator().evaluate(
        {
            strategy: (training[strategy].result.model, weighted.indexed, evaluation)
            for strategy in TRAINED_STRATEGIES
        }
    )
    random_history, random_metrics, random_diagnostics = _frozen_random(
        Path(signal_results_dir) / "weighted"
    )
    histories = {RANDOM_UNKNOWN: random_history}
    histories.update({name: training[name].result.history for name in TRAINED_STRATEGIES})
    metrics = {RANDOM_UNKNOWN: random_metrics}
    metrics.update({name: [*validation[name], *test[name][0]] for name in TRAINED_STRATEGIES})
    diagnostics = {RANDOM_UNKNOWN: random_diagnostics}
    for name in TRAINED_STRATEGIES:
        diagnostics[name] = model_diagnostics(
            training[name].result.model, weighted.indexed, evaluation, test[name][1]
        )
        diagnostics[name]["backfill_group_performance"] = purchase_group_performance(
            weighted.indexed, evaluation, test[name][1], sampling[name].backfill_user_indices
        )
    cart_rows = _cart_comparison(Path(popularity_dir))
    comparison: list[dict[str, object]] = []
    for k in (5, 10, 20):
        cart = cart_rows[k]
        comparison.append(
            {
                "strategy": "cart_popularity", "k": k,
                "eligible_users": int(cart["eligible_users"]), "recall": float(cart["recall"]),
                "ndcg": float(cart["ndcg"]), "hit_rate": float(cart["hit_rate"]),
                "precision": float(cart["precision"]),
            }
        )
        for strategy in STRATEGIES:
            row = next(
                value for value in metrics[strategy]
                if value["split"] == "test" and value["task"] == "purchase" and value["k"] == k
            )
            comparison.append({"strategy": strategy, **{key: row[key] for key in ("k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")}})
    cart_scores, exposure_counts = _item_observed_counts(root, weighted)
    stats = [
        sampling_statistics(value, user_count=len(weighted.indexed.user_ids), item_count=len(weighted.indexed.item_ids))
        for value in sampling.values()
    ]
    hardness = [
        row for value in sampling.values()
        for row in hardness_diagnostics(value, item_cart_scores=cart_scores, item_exposure_counts=exposure_counts)
    ]
    personalization = [
        {
            "strategy": name,
            "unique_purchase_top10_lists": diagnostics[name]["unique_purchase_top10_lists"],
            "purchase_evaluation_users": diagnostics[name]["purchase_evaluation_users"],
            "average_pairwise_top10_overlap": diagnostics[name]["average_pairwise_top10_overlap"],
            "average_cart_popularity_top10_overlap": diagnostics[name]["average_cart_popularity_top10_overlap"],
            "recommended_item_cart_score_mean": diagnostics[name]["recommended_item_cart_score"]["mean"],
            "recommended_item_cart_score_median": diagnostics[name]["recommended_item_cart_score"]["median"],
        }
        for name in STRATEGIES
    ]
    return NegativeSamplingExperimentResult(
        settings, weighted, sampling, training, histories, metrics, diagnostics,
        comparison, stats, hardness, personalization,
    )
