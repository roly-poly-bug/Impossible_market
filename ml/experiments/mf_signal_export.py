from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ml.experiments.mf_signal_experiment import EXPERIMENT_VERSION, SignalExperimentResult
from ml.representations.mf_signal import BINARY_VIEW, REPRESENTATIONS
from ml.training.mf_trainer import save_checkpoint


HISTORY_COLUMNS = (
    "epoch", "train_loss", "validation_purchase_recall_at_10",
    "validation_purchase_ndcg_at_10", "validation_viewplus_ndcg_at_10",
    "validation_favoriteplus_ndcg_at_10",
)
METRIC_COLUMNS = (
    "representation", "task", "split", "k", "eligible_users",
    "recall", "ndcg", "hit_rate", "precision",
)
COMPARISON_COLUMNS = (
    "model", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision",
)
COVERAGE_COLUMNS = (
    "representation", "positive_pair_count", "training_user_count", "training_item_count",
    "cold_user_count", "cold_item_count", "density",
)
CONFIDENCE_COLUMNS = (
    "representation", "count", "mean", "std", "min", "p25", "median", "p75", "max", "normalized",
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


def export_signal_results(
    result: SignalExperimentResult,
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    popularity_dir: str | Path,
    mf_v1_dir: str | Path,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    manifests: dict[str, dict[str, object]] = {}
    for name in REPRESENTATIONS:
        model_dir = destination / name
        model_dir.mkdir(exist_ok=True)
        history_path = model_dir / "training_history.csv"
        history_count = _write_csv(history_path, result.histories[name], HISTORY_COLUMNS)
        metrics_path = model_dir / "metrics.csv"
        metrics_count = _write_csv(
            metrics_path,
            ({"representation": name, **row} for row in result.metrics[name]),
            METRIC_COLUMNS,
        )
        diagnostics_path = model_dir / "diagnostics.json"
        _write_json(diagnostics_path, result.diagnostics[name])
        data = result.data[name]
        config_path = model_dir / "config.json"
        _write_json(
            config_path,
            {
                "experiment_version": EXPERIMENT_VERSION,
                "dataset_version": "recommendation_dataset_v1",
                "representation": name,
                "architecture": "bias-free BCE user/item embedding dot product",
                "positive_task": data.spec.task,
                "confidence_formula": data.spec.confidence_formula,
                "confidence_normalization": "none",
                "sampled_unknown_policy": "representation-specific Unknown; excludes Observed Non-conversion",
                "training_label_zero_meaning": "sampled unknown; not true dislike",
                "selection_metric": "validation_purchase_ndcg_at_10",
                "candidate_policy": "recommendation_dataset_v1 task-specific full candidate set",
                "exclude_seen": True,
                "cold_user_policy": "Train-only Cart popularity fallback",
                "test_policy": "one final evaluation after checkpoint selection; binary control reused",
                "fixed_parameters": result.config.as_dict(),
            },
        )
        artifacts = {
            history_path.name: _metadata(history_path, history_count),
            metrics_path.name: _metadata(metrics_path, metrics_count),
            diagnostics_path.name: _metadata(diagnostics_path),
            config_path.name: _metadata(config_path),
        }
        if name != BINARY_VIEW:
            checkpoint_path = checkpoint_dir / f"{name}_best.pt"
            save_checkpoint(result.training[name].result, checkpoint_path)
            artifacts[f"../checkpoints/{checkpoint_path.name}"] = _metadata(checkpoint_path)
            best_epoch = result.training[name].result.best_epoch
            best_metric = result.training[name].result.best_validation_purchase_ndcg_at_10
        else:
            best_epoch = int(result.histories[name][-1]["epoch"])
            best_metric = max(float(row["validation_purchase_ndcg_at_10"]) for row in result.histories[name])
        manifest = {
            "experiment_version": EXPERIMENT_VERSION,
            "representation": name,
            "source": "frozen matrix_factorization_v1" if name == BINARY_VIEW else "new fixed-config training",
            "best_epoch": best_epoch if name != BINARY_VIEW else 6,
            "best_validation_purchase_ndcg_at_10": best_metric,
            "coverage": data.coverage,
            "confidence_stats": data.confidence_stats,
            "artifacts": artifacts,
        }
        manifest_path = model_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        manifests[name] = _metadata(manifest_path)

    root_csvs = {
        "comparison.csv": (result.comparison, COMPARISON_COLUMNS),
        "coverage.csv": (result.coverage, COVERAGE_COLUMNS),
        "confidence_stats.csv": (result.confidence_stats, CONFIDENCE_COLUMNS),
    }
    root_artifacts: dict[str, dict[str, object]] = {}
    for filename, (rows, columns) in root_csvs.items():
        path = destination / filename
        count = _write_csv(path, rows, columns)
        root_artifacts[filename] = _metadata(path, count)
    cold_path = destination / "purchase_only_cold_start.json"
    _write_json(cold_path, result.purchase_only_cold_start)
    root_artifacts[cold_path.name] = _metadata(cold_path)
    root_manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_version": "recommendation_dataset_v1",
        "fixed_mf_config": result.config.as_dict(),
        "representations": list(REPRESENTATIONS),
        "test_policy": "frozen binary control reused; four new best checkpoints each tested once; no post-Test tuning",
        "dataset_manifest_sha256": _sha256(Path(dataset_dir) / "manifest.json"),
        "popularity_manifest_sha256": _sha256(Path(popularity_dir) / "manifest.json"),
        "mf_v1_manifest_sha256": _sha256(Path(mf_v1_dir) / "manifest.json"),
        "representation_manifests": manifests,
        "artifacts": root_artifacts,
    }
    _write_json(destination / "manifest.json", root_manifest)
    return root_manifest
