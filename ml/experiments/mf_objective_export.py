from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ml.experiments.mf_latent_dim_export import _csv, _meta
from ml.experiments.mf_objective_experiment import EXPERIMENT_VERSION, ObjectiveExperimentResult


HISTORY = ("epoch", "train_loss", "validation_purchase_recall_at_10", "validation_purchase_ndcg_at_10", "validation_viewplus_ndcg_at_10", "validation_favoriteplus_ndcg_at_10")
METRICS = ("model", "task", "split", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")
COMPARISON = ("model", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")
MARGIN = ("objective", "count", "mean", "std", "min", "median", "max", "positive_share")
PERSONALIZATION = ("objective", "unique_purchase_top10_lists", "average_pairwise_top10_overlap", "average_cart_popularity_top10_overlap", "recommended_item_cart_score_mean")
BIAS = ("objective", "mean", "std", "min", "median", "max", "cart_pearson", "cart_spearman", "purchase_pearson", "purchase_spearman")
DECOMPOSITION = ("objective", "personal_variance", "item_bias_variance", "bias_to_personal_variance_ratio")
EMBEDDING = ("objective", "user_norm_mean", "user_norm_std", "user_norm_min", "user_norm_median", "user_norm_max", "item_norm_mean", "item_norm_std", "item_norm_min", "item_norm_median", "item_norm_max", "score_mean", "score_std", "score_min", "score_median", "score_max")
HISTORY_GROUP = ("objective", "history_group", "min_train_positive_pairs", "max_train_positive_pairs", "mean_train_positive_pairs", "eligible_users", "recall_at_10", "ndcg_at_10", "hit_rate_at_10", "precision_at_10")
POPULARITY_GROUP = ("objective", "popularity_group", "item_count", "min_train_cart_count", "max_train_cart_count", "mean_train_cart_count", "eligible_users", "recall_at_10", "ndcg_at_10", "hit_rate_at_10", "precision_at_10")
CAPACITY = ("objective", "total_trainable_parameters", "user_embedding_parameters", "item_embedding_parameters", "item_bias_parameters", "training_runtime_seconds", "epoch_runtime_seconds", "best_checkpoint_bytes")


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_objective_results(result: ObjectiveExperimentResult, output_dir, *, dataset_dir, latent_results_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    bpr_checkpoint = checkpoint_dir / "bpr_best.pt"
    torch.save(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "objective": "confidence_weighted_bpr",
            "state_dict": result.bpr_training.model.state_dict(),
            "best_epoch": result.bpr_training.best_epoch,
            "best_validation_purchase_ndcg_at_10": result.bpr_training.best_validation_purchase_ndcg_at_10,
            "config": result.config.as_dict(),
        },
        bpr_checkpoint,
    )
    bce_checkpoint = Path(latent_results_dir) / "checkpoints" / "dim8_best.pt"

    artifacts = {}
    for objective in ("bce", "bpr"):
        directory = root / objective
        directory.mkdir(exist_ok=True)
        history = directory / "training_history.csv"
        metrics = directory / "metrics.csv"
        artifacts[f"{objective}/training_history.csv"] = _meta(history, _csv(history, result.histories[objective], HISTORY))
        artifacts[f"{objective}/metrics.csv"] = _meta(metrics, _csv(metrics, ({"model": objective, **row} for row in result.metrics[objective]), METRICS))
        diagnostics = directory / "diagnostics.json"
        _json(diagnostics, result.diagnostics[objective])
        artifacts[f"{objective}/diagnostics.json"] = _meta(diagnostics)

    result.capacity[0]["best_checkpoint_bytes"] = bce_checkpoint.stat().st_size
    result.capacity[1]["best_checkpoint_bytes"] = bpr_checkpoint.stat().st_size
    tables = {
        "comparison.csv": (result.comparison, COMPARISON),
        "margin_diagnostics.csv": (result.margins, MARGIN),
        "personalization.csv": (result.personalization, PERSONALIZATION),
        "bias_diagnostics.csv": (result.bias_rows, BIAS),
        "score_decomposition.csv": (result.decomposition, DECOMPOSITION),
        "embedding_diagnostics.csv": (result.embeddings, EMBEDDING),
        "history_group_metrics.csv": (result.history_groups, HISTORY_GROUP),
        "item_popularity_metrics.csv": (result.popularity_groups, POPULARITY_GROUP),
        "capacity.csv": (result.capacity, CAPACITY),
    }
    for filename, (rows, columns) in tables.items():
        path = root / filename
        artifacts[filename] = _meta(path, _csv(path, rows, columns))

    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "objectives": ["BCE", "confidence-weighted BPR"],
        "fixed_parameters": result.config.as_dict(),
        "representation": "Existing Weighted v1: log1p(view)+3*favorite+5*cart+8*purchase",
        "confidence": "1+log1p(strength)",
        "sampling": "shared Train-only Exposed Non-conversion with Random Unknown backfill",
        "comparison_ratio": 4,
        "model": "MF latent_dim=8 with Item Bias",
        "selection_metric": "validation_purchase_ndcg_at_10",
        "test_policy": "single batched final Test evaluation; no post-Test tuning",
        "bpr_weighting": "positive confidence",
        "synthetic_hidden_dimension_used": False,
    }
    _json(root / "config.json", config)
    artifacts["config.json"] = _meta(root / "config.json")
    manifest = {
        **config,
        "shared_positive_pair_count": len(result.data.indexed.positive_pairs),
        "shared_comparison_triple_count": len(result.data.sampled.triples),
        "bce_source_checkpoint": {"path": "results/mf_latent_dim_v1/checkpoints/dim8_best.pt", "bytes": bce_checkpoint.stat().st_size, "sha256": _sha(bce_checkpoint)},
        "bpr_checkpoint": _meta(bpr_checkpoint),
        "dataset_manifest_sha256": _sha(Path(dataset_dir) / "manifest.json"),
        "latent_results_manifest_sha256": _sha(Path(latent_results_dir) / "manifest.json"),
        "artifacts": artifacts,
    }
    _json(root / "manifest.json", manifest)
    return manifest
