from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ml.experiments.mf_experiment import EXPERIMENT_VERSION, MFExperimentResult
from ml.training.mf_trainer import save_checkpoint


HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_purchase_recall_at_10",
    "validation_purchase_ndcg_at_10",
    "validation_viewplus_ndcg_at_10",
    "validation_favoriteplus_ndcg_at_10",
)
METRIC_COLUMNS = (
    "model_type",
    "task",
    "split",
    "k",
    "eligible_users",
    "recall",
    "ndcg",
    "hit_rate",
    "precision",
)
COMPARISON_COLUMNS = (
    "model",
    "k",
    "eligible_users",
    "recall",
    "ndcg",
    "hit_rate",
    "precision",
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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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


def export_mf_results(
    result: MFExperimentResult,
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    popularity_dir: str | Path,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    model_manifests = {}
    for model_type in ("bce", "bpr"):
        model_dir = destination / model_type
        model_dir.mkdir(exist_ok=True)
        training = result.training[model_type]
        history_path = model_dir / "training_history.csv"
        history_rows = _write_csv(history_path, training.history, HISTORY_COLUMNS)
        metrics_path = model_dir / "metrics.csv"
        metric_rows = _write_csv(
            metrics_path,
            ({"model_type": model_type, **row} for row in result.metrics[model_type]),
            METRIC_COLUMNS,
        )
        diagnostics_path = model_dir / "diagnostics.json"
        _write_json(diagnostics_path, result.diagnostics[model_type])
        config_path = model_dir / "config.json"
        _write_json(
            config_path,
            {
                "experiment_version": EXPERIMENT_VERSION,
                "dataset_version": "recommendation_dataset_v1",
                "model_type": model_type,
                "architecture": "bias-free user/item embedding dot product",
                "positive_signal": "Train Binary View+",
                "sampling_strategy": "random_unknown_excluding_observed_non_conversion",
                "training_label_zero_meaning": "sampled unknown training non-positive; not true dislike",
                "objective": "BCEWithLogitsLoss" if model_type == "bce" else "-logsigmoid(score_positive-score_unknown)",
                "optimizer": "Adam",
                "selection_metric": "validation_purchase_ndcg_at_10",
                "candidate_policy": "recommendation_dataset_v1 task-specific full candidate set",
                "exclude_seen": True,
                "cold_user_policy": "Train-only Cart popularity fallback",
                "test_policy": "one final evaluation after both checkpoints are selected",
                **result.config.as_dict(),
            },
        )
        checkpoint_path = checkpoints / f"{model_type}_best.pt"
        save_checkpoint(training, checkpoint_path)
        artifacts = {
            history_path.name: _metadata(history_path, history_rows),
            metrics_path.name: _metadata(metrics_path, metric_rows),
            diagnostics_path.name: _metadata(diagnostics_path),
            config_path.name: _metadata(config_path),
            f"../checkpoints/{checkpoint_path.name}": _metadata(checkpoint_path),
        }
        manifest = {
            "experiment_version": EXPERIMENT_VERSION,
            "model_type": model_type,
            "best_epoch": training.best_epoch,
            "best_validation_purchase_ndcg_at_10": training.best_validation_purchase_ndcg_at_10,
            "artifacts": artifacts,
        }
        manifest_path = model_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        model_manifests[model_type] = _metadata(manifest_path)

    comparison_path = destination / "comparison.csv"
    comparison_rows = _write_csv(
        comparison_path, result.comparison, COMPARISON_COLUMNS
    )
    dataset_manifest = Path(dataset_dir) / "manifest.json"
    popularity_manifest = Path(popularity_dir) / "manifest.json"
    root_manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_version": "recommendation_dataset_v1",
        "seed": result.config.seed,
        "sampled_unknown_policy": "4 distinct per positive from View+ Unknown only; not true negatives",
        "positive_pair_count": len(result.indexed.positive_pairs),
        "sampled_unknown_triple_count": len(result.sampled.triples),
        "cold_user_count": len(result.indexed.cold_user_indices),
        "cold_item_count": len(result.indexed.cold_item_indices),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "popularity_manifest_sha256": _sha256(popularity_manifest),
        "model_manifests": model_manifests,
        "comparison": _metadata(comparison_path, comparison_rows),
    }
    root_path = destination / "manifest.json"
    _write_json(root_path, root_manifest)
    return root_manifest
