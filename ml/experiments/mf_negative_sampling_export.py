from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ml.experiments.mf_negative_sampling_experiment import (
    EXPERIMENT_VERSION,
    NegativeSamplingExperimentResult,
    TRAINED_STRATEGIES,
)
from ml.training.mf_negative_sampling import RANDOM_UNKNOWN, STRATEGIES
from ml.training.mf_trainer import save_checkpoint


HISTORY_COLUMNS = (
    "epoch", "train_loss", "validation_purchase_recall_at_10",
    "validation_purchase_ndcg_at_10", "validation_viewplus_ndcg_at_10",
    "validation_favoriteplus_ndcg_at_10",
)
METRIC_COLUMNS = (
    "strategy", "task", "split", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision",
)
COMPARISON_COLUMNS = ("strategy", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")
SAMPLING_COLUMNS = (
    "strategy", "total_sampled_pairs", "unique_sampled_user_item_pairs",
    "random_unknown_count", "exposed_non_conversion_count", "exposed_share",
    "backfill_count", "backfill_rate", "training_user_count",
    "users_with_enough_exposed_pool", "users_requiring_backfill",
    "sampled_item_unique_count", "items_never_sampled", "sampled_item_gini", "top10_sampled_share",
)
HARDNESS_COLUMNS = (
    "strategy", "sample_source", "sample_count", "mean_item_cart_popularity",
    "median_item_cart_popularity", "std_item_cart_popularity",
    "mean_item_train_exposure", "median_item_train_exposure", "std_item_train_exposure",
    "all_values_finite",
)
PERSONALIZATION_COLUMNS = (
    "strategy", "unique_purchase_top10_lists", "purchase_evaluation_users",
    "average_pairwise_top10_overlap", "average_cart_popularity_top10_overlap",
    "recommended_item_cart_score_mean", "recommended_item_cart_score_median",
)


def _write_csv(path: Path, rows, columns: tuple[str, ...]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
            count += 1
    return count


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(path: Path, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if rows is not None:
        value["data_rows"] = rows
    return value


def export_negative_sampling_results(
    result: NegativeSamplingExperimentResult,
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    popularity_dir: str | Path,
    signal_results_dir: str | Path,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    strategy_manifests = {}
    for strategy in STRATEGIES:
        strategy_dir = destination / strategy
        strategy_dir.mkdir(exist_ok=True)
        history_path = strategy_dir / "training_history.csv"
        history_rows = _write_csv(history_path, result.histories[strategy], HISTORY_COLUMNS)
        metrics_path = strategy_dir / "metrics.csv"
        metric_rows = _write_csv(
            metrics_path,
            ({"strategy": strategy, **row} for row in result.metrics[strategy]),
            METRIC_COLUMNS,
        )
        diagnostics_path = strategy_dir / "diagnostics.json"
        _write_json(diagnostics_path, result.diagnostics[strategy])
        config_path = strategy_dir / "config.json"
        _write_json(
            config_path,
            {
                "experiment_version": EXPERIMENT_VERSION,
                "strategy": strategy,
                "architecture": "bias-free Weighted BCE MF dot product",
                "positive_definition": "Train View+",
                "positive_confidence": "1 + log1p(log1p(view_count) + 3*favorite + 5*cart + 8*purchase)",
                "sample_ratio": 4,
                "mixed_ratio": "2 exposed + 2 unknown; exposed shortage backfilled by unknown",
                "exposed_policy": "impression observed and no View; training contrast label, not true dislike",
                "selection_metric": "validation_purchase_ndcg_at_10",
                "test_policy": "one final Test after checkpoint selection; random control reused",
                "candidate_policy": "Recommendation Dataset v1 task-specific full ranking",
                "exclude_seen": True,
                "fixed_parameters": result.config.as_dict(),
            },
        )
        artifacts = {
            history_path.name: _metadata(history_path, history_rows),
            metrics_path.name: _metadata(metrics_path, metric_rows),
            diagnostics_path.name: _metadata(diagnostics_path),
            config_path.name: _metadata(config_path),
        }
        if strategy in TRAINED_STRATEGIES:
            checkpoint_path = checkpoint_dir / f"{strategy}_best.pt"
            save_checkpoint(result.training[strategy].result, checkpoint_path)
            artifacts[f"../checkpoints/{checkpoint_path.name}"] = _metadata(checkpoint_path)
            best_epoch = result.training[strategy].result.best_epoch
            best_ndcg = result.training[strategy].result.best_validation_purchase_ndcg_at_10
        else:
            best_epoch = 7
            best_ndcg = max(float(row["validation_purchase_ndcg_at_10"]) for row in result.histories[strategy])
        manifest = {
            "experiment_version": EXPERIMENT_VERSION,
            "strategy": strategy,
            "source": "frozen Weighted Random Unknown control" if strategy == RANDOM_UNKNOWN else "new fixed-config training",
            "best_epoch": best_epoch,
            "best_validation_purchase_ndcg_at_10": best_ndcg,
            "sampling_stats": next(row for row in result.sampling_stats if row["strategy"] == strategy),
            "artifacts": artifacts,
        }
        manifest_path = strategy_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        strategy_manifests[strategy] = _metadata(manifest_path)
    root_tables = {
        "comparison.csv": (result.comparison, COMPARISON_COLUMNS),
        "sampling_stats.csv": (result.sampling_stats, SAMPLING_COLUMNS),
        "hardness_diagnostics.csv": (result.hardness, HARDNESS_COLUMNS),
        "personalization.csv": (result.personalization, PERSONALIZATION_COLUMNS),
    }
    root_artifacts = {}
    for filename, (rows, columns) in root_tables.items():
        path = destination / filename
        row_count = _write_csv(path, rows, columns)
        root_artifacts[filename] = _metadata(path, row_count)
    root_manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_version": "recommendation_dataset_v1",
        "fixed_mf_config": result.config.as_dict(),
        "strategies": list(STRATEGIES),
        "test_policy": "frozen Random control reused; two new best checkpoints tested once; no post-Test tuning",
        "dataset_manifest_sha256": _sha256(Path(dataset_dir) / "manifest.json"),
        "popularity_manifest_sha256": _sha256(Path(popularity_dir) / "manifest.json"),
        "signal_results_manifest_sha256": _sha256(Path(signal_results_dir) / "manifest.json"),
        "strategy_manifests": strategy_manifests,
        "artifacts": root_artifacts,
    }
    _write_json(destination / "manifest.json", root_manifest)
    return root_manifest
