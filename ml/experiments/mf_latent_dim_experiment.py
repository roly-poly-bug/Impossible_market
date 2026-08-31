from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ml.evaluation.matrix_factorization import evaluate_model, load_evaluation_data, model_diagnostics
from ml.evaluation.mf_bias import bias_diagnostics
from ml.evaluation.mf_latent_dim import history_group_metrics, item_popularity_group_metrics, normalized_norm, parameter_counts
from ml.evaluation.mf_signal import SignalFinalTestEvaluator
from ml.experiments.mf_bias_experiment import _frozen_control
from ml.experiments.mf_negative_sampling_experiment import _cart_comparison
from ml.models.mf_bias import BiasedMatrixFactorization
from ml.representations.mf_cart_signal import EXISTING_WEIGHTED, load_cart_signal
from ml.training.mf_bias_trainer import BiasTrainingResult, train_bias_model
from ml.training.mf_trainer import MFTrainingConfig


DIMENSIONS = (8, 16, 32, 64)
TRAINED_DIMENSIONS = (8, 32, 64)
EXPERIMENT_VERSION = "mf_latent_dim_v1"


@dataclass
class LatentDimExperimentResult:
    base_config: MFTrainingConfig
    data: object
    training: dict[int, BiasTrainingResult]
    models: dict[int, BiasedMatrixFactorization]
    histories: dict[int, list[dict[str, object]]]
    metrics: dict[int, list[dict[str, object]]]
    diagnostics: dict[int, dict[str, object]]
    runtimes: dict[int, float | None]
    comparison: list[dict[str, object]]
    capacity: list[dict[str, object]]
    embeddings: list[dict[str, object]]
    bias_rows: list[dict[str, object]]
    decomposition: list[dict[str, object]]
    personalization: list[dict[str, object]]
    history_groups: list[dict[str, object]]
    popularity_groups: list[dict[str, object]]


def _config_for(base: MFTrainingConfig, latent_dim: int) -> MFTrainingConfig:
    return MFTrainingConfig(
        latent_dim=latent_dim, learning_rate=base.learning_rate, weight_decay=base.weight_decay,
        batch_size=base.batch_size, negative_ratio=base.negative_ratio, seed=base.seed,
        max_epochs=base.max_epochs, patience=base.patience,
        deterministic_algorithms=base.deterministic_algorithms, torch_num_threads=base.torch_num_threads,
    )


def _load_frozen_dim16(path: Path, user_count: int, item_count: int) -> BiasedMatrixFactorization:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = BiasedMatrixFactorization(user_count, item_count, 16, "item_bias")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def run_latent_dim_experiment(dataset_dir, popularity_dir, bias_results_dir, *, config=None) -> LatentDimExperimentResult:
    base = config or MFTrainingConfig()
    data, sampling = load_cart_signal(dataset_dir, EXISTING_WEIGHTED, sample_ratio=4, seed=42)
    evaluation = load_evaluation_data(dataset_dir, data.indexed)
    training: dict[int, BiasTrainingResult] = {}
    runtimes: dict[int, float | None] = {16: None}
    for dimension in TRAINED_DIMENSIONS:
        started = time.perf_counter()
        training[dimension] = train_bias_model("item_bias", data, evaluation, _config_for(base, dimension))
        runtimes[dimension] = time.perf_counter() - started
    models = {dimension: training[dimension].model for dimension in TRAINED_DIMENSIONS}
    models[16] = _load_frozen_dim16(Path(bias_results_dir) / "checkpoints" / "item_bias_best.pt", len(data.indexed.user_ids), len(data.indexed.item_ids))
    validation = {dimension: evaluate_model(model, data.indexed, evaluation, split="validation")[0] for dimension, model in models.items()}
    test = SignalFinalTestEvaluator().evaluate({f"dim{dimension}": (model, data.indexed, evaluation) for dimension, model in models.items()})
    frozen_history, frozen_metrics, _ = _frozen_control(Path(bias_results_dir) / "item_bias")
    histories = {16: frozen_history, **{dimension: result.history for dimension, result in training.items()}}
    metrics = {
        dimension: [*validation[dimension], *test[f"dim{dimension}"][0]]
        for dimension in DIMENSIONS
    }
    diagnostics = {}
    capacity = []
    embeddings = []
    bias_rows = []
    decomposition = []
    personalization = []
    history_groups = []
    popularity_groups = []
    for dimension in DIMENSIONS:
        model = models[dimension]
        rankings = test[f"dim{dimension}"][1]
        diag = model_diagnostics(model, data.indexed, evaluation, rankings)
        diag["bias_analysis"] = bias_diagnostics(model, data.indexed, evaluation, Path(dataset_dir) / "train_viewplus.csv")
        diagnostics[dimension] = diag
        count = parameter_counts(model)
        capacity.append({"latent_dim": dimension, **count, "training_runtime_seconds": runtimes[dimension], "epoch_runtime_seconds": (runtimes[dimension] / len(histories[dimension])) if runtimes[dimension] is not None else None})
        user_norm, item_norm = diag["user_embedding_norm"], diag["item_embedding_norm"]
        score = diag["score_distribution"]
        embeddings.append({
            "latent_dim": dimension,
            **{f"user_norm_{key}": user_norm[key] for key in ("mean", "std", "min", "median", "max")},
            "user_norm_mean_per_sqrt_dim": normalized_norm(user_norm["mean"], dimension),
            **{f"item_norm_{key}": item_norm[key] for key in ("mean", "std", "min", "median", "max")},
            "item_norm_mean_per_sqrt_dim": normalized_norm(item_norm["mean"], dimension),
            **{f"score_{key}": score[key] for key in ("mean", "std", "min", "median", "max")},
        })
        bias = diag["bias_analysis"]
        corr = bias["item_bias_correlations"]
        bias_rows.append({"latent_dim": dimension, **bias["item_bias"], "cart_pearson": corr["cart"]["pearson"], "cart_spearman": corr["cart"]["spearman"], "purchase_pearson": corr["purchase"]["pearson"], "purchase_spearman": corr["purchase"]["spearman"]})
        dec = bias["score_decomposition"]
        decomposition.append({"latent_dim": dimension, "personal_variance": dec["personal_variance"], "item_bias_variance": dec["item_bias_variance"], "bias_to_personal_variance_ratio": dec["item_bias_to_personal_variance_ratio"]})
        personalization.append({"latent_dim": dimension, "unique_purchase_top10_lists": diag["unique_purchase_top10_lists"], "average_pairwise_top10_overlap": diag["average_pairwise_top10_overlap"], "average_cart_popularity_top10_overlap": diag["average_cart_popularity_top10_overlap"], "recommended_item_cart_score_mean": diag["recommended_item_cart_score"]["mean"]})
        history_groups.extend({"latent_dim": dimension, **row} for row in history_group_metrics(data.indexed, evaluation, rankings))
        popularity_groups.extend({"latent_dim": dimension, **row} for row in item_popularity_group_metrics(evaluation, rankings))
    cart = _cart_comparison(Path(popularity_dir))
    comparison = []
    for k in (5, 10, 20):
        row = cart[k]
        comparison.append({"model": "cart_popularity", "k": k, **{key: int(row[key]) if key == "eligible_users" else float(row[key]) for key in ("eligible_users", "recall", "ndcg", "hit_rate", "precision")}})
        for dimension in DIMENSIONS:
            metric = next(row for row in metrics[dimension] if row["split"] == "test" and row["task"] == "purchase" and row["k"] == k)
            comparison.append({"model": f"dim{dimension}", **{key: metric[key] for key in ("k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")}})
    return LatentDimExperimentResult(base, data, training, models, histories, metrics, diagnostics, runtimes, comparison, capacity, embeddings, bias_rows, decomposition, personalization, history_groups, popularity_groups)
