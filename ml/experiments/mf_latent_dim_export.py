from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ml.experiments.mf_latent_dim_experiment import DIMENSIONS, TRAINED_DIMENSIONS, EXPERIMENT_VERSION, LatentDimExperimentResult
from ml.training.mf_trainer import save_checkpoint


H = ("epoch", "train_loss", "validation_purchase_recall_at_10", "validation_purchase_ndcg_at_10", "validation_viewplus_ndcg_at_10", "validation_favoriteplus_ndcg_at_10")
M = ("model", "task", "split", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")
CMP = ("model", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")
CAP = ("latent_dim", "total_trainable_parameters", "user_embedding_parameters", "item_embedding_parameters", "item_bias_parameters", "training_runtime_seconds", "epoch_runtime_seconds", "best_checkpoint_bytes")
EMB = ("latent_dim", "user_norm_mean", "user_norm_std", "user_norm_min", "user_norm_median", "user_norm_max", "user_norm_mean_per_sqrt_dim", "item_norm_mean", "item_norm_std", "item_norm_min", "item_norm_median", "item_norm_max", "item_norm_mean_per_sqrt_dim", "score_mean", "score_std", "score_min", "score_median", "score_max")
BIAS = ("latent_dim", "mean", "std", "min", "median", "max", "cart_pearson", "cart_spearman", "purchase_pearson", "purchase_spearman")
DEC = ("latent_dim", "personal_variance", "item_bias_variance", "bias_to_personal_variance_ratio")
PER = ("latent_dim", "unique_purchase_top10_lists", "average_pairwise_top10_overlap", "average_cart_popularity_top10_overlap", "recommended_item_cart_score_mean")
HIST = ("latent_dim", "history_group", "min_train_positive_pairs", "max_train_positive_pairs", "mean_train_positive_pairs", "eligible_users", "recall_at_10", "ndcg_at_10", "hit_rate_at_10", "precision_at_10")
POP = ("latent_dim", "popularity_group", "item_count", "min_train_cart_count", "max_train_cart_count", "mean_train_cart_count", "eligible_users", "recall_at_10", "ndcg_at_10", "hit_rate_at_10", "precision_at_10")


def _csv(path: Path, rows, columns) -> int:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); count = 0
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns}); count += 1
    return count


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta(path: Path, rows=None):
    value = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    if rows is not None: value["data_rows"] = rows
    return value


def export_latent_dim_results(result: LatentDimExperimentResult, output_dir, *, dataset_dir, bias_results_dir):
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    checkpoints = root / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    checkpoint_sizes = {16: (Path(bias_results_dir) / "checkpoints" / "item_bias_best.pt").stat().st_size}
    for dimension in TRAINED_DIMENSIONS:
        checkpoint = checkpoints / f"dim{dimension}_best.pt"
        save_checkpoint(result.training[dimension], checkpoint)
        checkpoint_sizes[dimension] = checkpoint.stat().st_size
    model_manifests = {}
    for dimension in DIMENSIONS:
        name = f"dim{dimension}"; directory = root / name; directory.mkdir(exist_ok=True)
        history = directory / "training_history.csv"; metrics = directory / "metrics.csv"
        hn = _csv(history, result.histories[dimension], H)
        mn = _csv(metrics, ({"model": name, **row} for row in result.metrics[dimension]), M)
        diagnostics = directory / "diagnostics.json"; _json(diagnostics, result.diagnostics[dimension])
        config = directory / "config.json"; fixed = result.base_config.as_dict(); fixed["latent_dim"] = dimension
        _json(config, {"experiment_version": EXPERIMENT_VERSION, "latent_dim": dimension, "fixed_parameters": fixed, "only_changed_parameter": "latent_dim", "representation": "Existing Weighted v1", "sampling": "shared Train-only Exposed Non-conversion with Random Unknown backfill", "model": "BCE MF with Item Bias", "selection_metric": "validation_purchase_ndcg_at_10", "test_policy": "single batched final Test; no tuning", "synthetic_hidden_dimension_used": False})
        artifacts = {history.name: _meta(history, hn), metrics.name: _meta(metrics, mn), diagnostics.name: _meta(diagnostics), config.name: _meta(config)}
        if dimension in TRAINED_DIMENSIONS:
            artifacts[f"../checkpoints/dim{dimension}_best.pt"] = _meta(checkpoints / f"dim{dimension}_best.pt")
            best_epoch = result.training[dimension].best_epoch; best_ndcg = result.training[dimension].best_validation_purchase_ndcg_at_10
        else:
            artifacts["source_checkpoint"] = {"path": "results/mf_bias_v1/checkpoints/item_bias_best.pt", "bytes": checkpoint_sizes[16], "sha256": _sha(Path(bias_results_dir) / "checkpoints" / "item_bias_best.pt")}
            best_epoch = 9; best_ndcg = max(float(row["validation_purchase_ndcg_at_10"]) for row in result.histories[16])
        manifest = directory / "manifest.json"; _json(manifest, {"latent_dim": dimension, "best_epoch": best_epoch, "best_validation_purchase_ndcg_at_10": best_ndcg, "artifacts": artifacts})
        model_manifests[name] = _meta(manifest)
    capacity = [{**row, "best_checkpoint_bytes": checkpoint_sizes[int(row["latent_dim"])]} for row in result.capacity]
    tables = {
        "comparison.csv": (result.comparison, CMP), "capacity.csv": (capacity, CAP),
        "embedding_diagnostics.csv": (result.embeddings, EMB), "bias_diagnostics.csv": (result.bias_rows, BIAS),
        "score_decomposition.csv": (result.decomposition, DEC), "personalization.csv": (result.personalization, PER),
        "history_group_metrics.csv": (result.history_groups, HIST), "item_popularity_metrics.csv": (result.popularity_groups, POP),
    }
    artifacts = {}
    for filename, (rows, columns) in tables.items():
        path = root / filename; artifacts[filename] = _meta(path, _csv(path, rows, columns))
    manifest = {
        "experiment_version": EXPERIMENT_VERSION, "dimensions": list(DIMENSIONS),
        "newly_trained_dimensions": list(TRAINED_DIMENSIONS), "reused_dimension": 16,
        "fixed_config_except_latent_dim": {key: value for key, value in result.base_config.as_dict().items() if key != "latent_dim"},
        "shared_positive_pair_count": len(result.data.indexed.positive_pairs), "shared_sample_count": len(result.data.sampled.triples),
        "synthetic_hidden_dimension_used": False,
        "test_policy": "four fixed dimensions in one batched final Test evaluation; no post-Test tuning",
        "dataset_manifest_sha256": _sha(Path(dataset_dir) / "manifest.json"),
        "bias_results_manifest_sha256": _sha(Path(bias_results_dir) / "manifest.json"),
        "model_manifests": model_manifests, "artifacts": artifacts,
    }
    _json(root / "manifest.json", manifest)
    return manifest
