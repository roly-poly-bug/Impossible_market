from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ml.evaluation.matrix_factorization import (
    FinalTestEvaluator,
    evaluate_model,
    load_evaluation_data,
    model_diagnostics,
)
from ml.training.mf_data import (
    IndexedInteractions,
    SampledTrainingData,
    load_binary_viewplus_training,
    sample_random_unknowns,
)
from ml.training.mf_trainer import (
    MFTrainingConfig,
    TrainingResult,
    train_matrix_factorization,
)


EXPERIMENT_VERSION = "matrix_factorization_v1"


@dataclass
class MFExperimentResult:
    config: MFTrainingConfig
    indexed: IndexedInteractions
    sampled: SampledTrainingData
    training: dict[str, TrainingResult]
    metrics: dict[str, list[dict[str, object]]]
    diagnostics: dict[str, dict[str, object]]
    comparison: list[dict[str, object]]


def _cart_purchase_test_metrics(popularity_metrics_path: Path) -> dict[int, dict[str, object]]:
    with popularity_metrics_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return {
        int(row["k"]): row
        for row in rows
        if row["popularity_signal"] == "cart_count"
        and row["evaluation_task"] == "purchase"
        and row["split"] == "test"
    }


def run_mf_experiment(
    dataset_dir: str | Path,
    popularity_dir: str | Path,
    *,
    config: MFTrainingConfig | None = None,
) -> MFExperimentResult:
    settings = config or MFTrainingConfig()
    root = Path(dataset_dir)
    indexed = load_binary_viewplus_training(root / "train_viewplus.csv")
    sampled = sample_random_unknowns(
        indexed,
        negative_ratio=settings.negative_ratio,
        seed=settings.seed,
    )
    evaluation = load_evaluation_data(root, indexed)
    training = {
        model_type: train_matrix_factorization(
            model_type, indexed, sampled, evaluation, settings
        )
        for model_type in ("bce", "bpr")
    }
    validation = {
        name: evaluate_model(result.model, indexed, evaluation, split="validation")[0]
        for name, result in training.items()
    }
    final_test = FinalTestEvaluator().evaluate(
        {name: result.model for name, result in training.items()},
        indexed,
        evaluation,
    )
    metrics = {
        name: [*validation[name], *final_test[name][0]] for name in training
    }
    diagnostics = {
        name: model_diagnostics(
            result.model,
            indexed,
            evaluation,
            final_test[name][1],
        )
        for name, result in training.items()
    }
    cart = _cart_purchase_test_metrics(Path(popularity_dir) / "metrics.csv")
    comparison = []
    for k in (5, 10, 20):
        cart_row = cart[k]
        comparison.append(
            {
                "model": "cart_popularity",
                "k": k,
                "eligible_users": int(cart_row["eligible_users"]),
                "recall": float(cart_row["recall"]),
                "ndcg": float(cart_row["ndcg"]),
                "hit_rate": float(cart_row["hit_rate"]),
                "precision": float(cart_row["precision"]),
            }
        )
        for name in ("bce", "bpr"):
            row = next(
                row
                for row in metrics[name]
                if row["split"] == "test" and row["task"] == "purchase" and row["k"] == k
            )
            comparison.append({"model": f"{name}_mf", **{key: row[key] for key in ("k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")}})
    return MFExperimentResult(
        settings,
        indexed,
        sampled,
        training,
        metrics,
        diagnostics,
        comparison,
    )
