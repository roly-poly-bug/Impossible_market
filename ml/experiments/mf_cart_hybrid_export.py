from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml.experiments.mf_cart_signal_export import _meta, _write_csv, _write_json
from ml.experiments.mf_cart_hybrid_experiment import EXPERIMENT_VERSION, HybridExperimentResult


METRICS = ("model", "task", "split", "alpha", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")
COMPARISON = ("model", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")
PERSONALIZATION = ("model", "unique_purchase_top10_lists", "average_pairwise_top10_overlap", "average_cart_popularity_top10_overlap", "recommended_item_cart_score_mean")
CONTRIBUTION = ("component", "mean", "std", "variance", "min", "max", "finite")
HISTORY = ("model", "history_group", "min_train_positive_pairs", "max_train_positive_pairs", "mean_train_positive_pairs", "eligible_users", "recall_at_10", "ndcg_at_10", "hit_rate_at_10", "precision_at_10")
POPULARITY = ("model", "popularity_group", "item_count", "min_train_cart_count", "max_train_cart_count", "mean_train_cart_count", "eligible_users", "recall_at_10", "ndcg_at_10", "hit_rate_at_10", "precision_at_10")
FALLBACK = ("model", "group", "user_count", "eligible_share", "recall_at_10", "ndcg_at_10", "hit_rate_at_10", "precision_at_10")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_hybrid_results(result: HybridExperimentResult, output_dir, *, dataset_dir, latent_results_dir):
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_alpha_metrics.csv": (({"model": "hybrid", **row} for row in result.validation), METRICS),
        "best_hybrid_metrics.csv": (({"model": "best_hybrid", **row} for row in result.test_metrics), METRICS),
        "comparison.csv": (result.comparison, COMPARISON),
        "personalization.csv": (result.personalization, PERSONALIZATION),
        "score_contribution.csv": (result.contributions, CONTRIBUTION),
        "history_group_metrics.csv": (result.history_groups, HISTORY),
        "item_popularity_metrics.csv": (result.popularity_groups, POPULARITY),
        "fallback_diagnostics.csv": (result.fallback, FALLBACK),
    }
    artifacts = {}
    for filename, (rows, columns) in tables.items():
        path = root / filename; artifacts[filename] = _meta(path, _write_csv(path, rows, columns))
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "model_training": False,
        "mf_checkpoint": "results/mf_latent_dim_v1/checkpoints/dim8_best.pt",
        "mf_score": "user_embedding dot item_embedding + item_bias",
        "cart_score": "Train-only add_to_cart count",
        "normalization": "per-user candidate MF z-score plus task-candidate Cart z-score; population std; zero std -> zeros",
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "selection": "Validation Purchase NDCG@10; tie: Recall@10, then higher alpha",
        "best_alpha": result.best_alpha,
        "exclude_seen": True,
        "fallback": "unchanged Train-only Cart Popularity for users without Train positives",
        "test_policy": "best Validation alpha only, one final Test evaluation; no post-Test tuning",
        "item_bias_cart_pearson": result.bias_cart_correlation["pearson"],
        "item_bias_cart_spearman": result.bias_cart_correlation["spearman"],
        "evaluation_runtime_seconds": result.evaluation_runtime_seconds,
        "additional_peak_memory_bytes": result.additional_peak_memory_bytes,
    }
    _write_json(root / "config.json", config); artifacts["config.json"] = _meta(root / "config.json")
    manifest = {
        **config,
        "dataset_manifest_sha256": _sha(Path(dataset_dir) / "manifest.json"),
        "mf_results_manifest_sha256": _sha(Path(latent_results_dir) / "manifest.json"),
        "mf_checkpoint_sha256": _sha(Path(latent_results_dir) / "checkpoints" / "dim8_best.pt"),
        "artifacts": artifacts,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest
