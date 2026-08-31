from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from ml.evaluation.matrix_factorization import evaluate_model, load_evaluation_data, model_diagnostics
from ml.evaluation.mf_signal import SignalFinalTestEvaluator, purchase_cold_start_diagnostic
from ml.representations.mf_signal import (
    BINARY_VIEW,
    FAVORITEPLUS,
    LOG_VIEW,
    PURCHASE_ONLY,
    REPRESENTATIONS,
    WEIGHTED,
    RepresentationData,
    load_representation,
)
from ml.training.mf_signal_trainer import SignalTrainingResult, train_signal_representation
from ml.training.mf_trainer import MFTrainingConfig


EXPERIMENT_VERSION = "mf_signal_representation_v1"
TRAINED_REPRESENTATIONS = (LOG_VIEW, FAVORITEPLUS, WEIGHTED, PURCHASE_ONLY)


@dataclass
class SignalExperimentResult:
    config: MFTrainingConfig
    data: dict[str, RepresentationData]
    training: dict[str, SignalTrainingResult]
    histories: dict[str, list[dict[str, object]]]
    metrics: dict[str, list[dict[str, object]]]
    diagnostics: dict[str, dict[str, object]]
    comparison: list[dict[str, object]]
    coverage: list[dict[str, object]]
    confidence_stats: list[dict[str, object]]
    purchase_only_cold_start: dict[str, object]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _binary_control(mf_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    history = [{key: float(value) for key, value in row.items()} for row in _read_csv(mf_dir / "bce" / "training_history.csv")]
    metrics: list[dict[str, object]] = []
    for row in _read_csv(mf_dir / "bce" / "metrics.csv"):
        metrics.append(
            {
                "task": row["task"],
                "split": row["split"],
                "k": int(row["k"]),
                "eligible_users": int(row["eligible_users"]),
                "recall": float(row["recall"]),
                "ndcg": float(row["ndcg"]),
                "hit_rate": float(row["hit_rate"]),
                "precision": float(row["precision"]),
            }
        )
    diagnostics = json.loads((mf_dir / "bce" / "diagnostics.json").read_text(encoding="utf-8"))
    diagnostics["source"] = "reused frozen matrix_factorization_v1 BCE result; Test not re-evaluated"
    return history, metrics, diagnostics


def _cart_rows(popularity_dir: Path) -> dict[int, dict[str, str]]:
    return {
        int(row["k"]): row
        for row in _read_csv(popularity_dir / "metrics.csv")
        if row["popularity_signal"] == "cart_count"
        and row["evaluation_task"] == "purchase"
        and row["split"] == "test"
    }


def run_signal_experiment(
    dataset_dir: str | Path,
    popularity_dir: str | Path,
    mf_v1_dir: str | Path,
    *,
    config: MFTrainingConfig | None = None,
) -> SignalExperimentResult:
    settings = config or MFTrainingConfig()
    root = Path(dataset_dir)
    data = {
        name: load_representation(
            root, name, negative_ratio=settings.negative_ratio, seed=settings.seed
        )
        for name in REPRESENTATIONS
    }
    training: dict[str, SignalTrainingResult] = {}
    validation: dict[str, list[dict[str, object]]] = {}
    for name in TRAINED_REPRESENTATIONS:
        evaluation = load_evaluation_data(root, data[name].indexed)
        training[name] = train_signal_representation(data[name], evaluation, settings)
        validation[name] = evaluate_model(
            training[name].result.model,
            data[name].indexed,
            evaluation,
            split="validation",
        )[0]

    # This is the one and only Test pass for the four newly trained representations.
    test = SignalFinalTestEvaluator().evaluate(
        {
            name: (
                training[name].result.model,
                data[name].indexed,
                load_evaluation_data(root, data[name].indexed),
            )
            for name in TRAINED_REPRESENTATIONS
        }
    )

    binary_history, binary_metrics, binary_diagnostics = _binary_control(Path(mf_v1_dir))
    histories = {BINARY_VIEW: binary_history}
    histories.update({name: value.result.history for name, value in training.items()})
    metrics = {BINARY_VIEW: binary_metrics}
    metrics.update({name: [*validation[name], *test[name][0]] for name in TRAINED_REPRESENTATIONS})
    diagnostics = {BINARY_VIEW: binary_diagnostics}
    for name in TRAINED_REPRESENTATIONS:
        evaluation = load_evaluation_data(root, data[name].indexed)
        diagnostics[name] = model_diagnostics(
            training[name].result.model, data[name].indexed, evaluation, test[name][1]
        )

    cart = _cart_rows(Path(popularity_dir))
    comparison: list[dict[str, object]] = []
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
        for name in REPRESENTATIONS:
            row = next(
                row for row in metrics[name]
                if row["split"] == "test" and row["task"] == "purchase" and row["k"] == k
            )
            comparison.append({"model": name, **{key: row[key] for key in ("k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")}})

    purchase_evaluation = load_evaluation_data(root, data[PURCHASE_ONLY].indexed)
    cold_start = purchase_cold_start_diagnostic(
        data[PURCHASE_ONLY].indexed, purchase_evaluation, test[PURCHASE_ONLY][1]
    )
    return SignalExperimentResult(
        settings,
        data,
        training,
        histories,
        metrics,
        diagnostics,
        comparison,
        [data[name].coverage for name in REPRESENTATIONS],
        [data[name].confidence_stats for name in REPRESENTATIONS],
        cold_start,
    )
