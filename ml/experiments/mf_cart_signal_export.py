from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ml.experiments.mf_cart_signal_experiment import EXPERIMENT_VERSION, SIGNALS, TRAINED_SIGNALS, CartSignalExperimentResult
from ml.training.mf_trainer import save_checkpoint


HISTORY_COLUMNS = ("epoch", "train_loss", "validation_purchase_recall_at_10", "validation_purchase_ndcg_at_10", "validation_viewplus_ndcg_at_10", "validation_favoriteplus_ndcg_at_10")
METRIC_COLUMNS = ("model", "task", "split", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")
COMPARISON_COLUMNS = ("model", "k", "eligible_users", "recall", "ndcg", "hit_rate", "precision")
COVERAGE_COLUMNS = ("signal", "positive_pair_count", "positive_user_count", "positive_item_count", "density", "train_positive_zero_user_count", "test_purchase_eligible_users", "test_eligible_with_train_positive", "test_eligible_without_train_positive", "test_fallback_rate")
FALLBACK_COLUMNS = ("signal", "group", "user_count", "eligible_share", "recall_at_10", "ndcg_at_10", "hit_rate_at_10", "precision_at_10")
PERSONALIZATION_COLUMNS = ("signal", "unique_purchase_top10_lists", "average_pairwise_top10_overlap", "average_cart_popularity_top10_overlap", "recommended_item_cart_score_mean")
BIAS_COLUMNS = ("signal", "mean", "std", "min", "median", "max", "cart_pearson", "cart_spearman", "purchase_pearson", "purchase_spearman")
ALIGNMENT_COLUMNS = ("signal", "test_purchase_eligible_users", "users_with_train_positive", "users_without_train_positive", "users_with_exact_item_continuity", "exact_item_continuity_user_rate", "mean_exact_item_overlap_count")


def _write_csv(path: Path, rows, columns) -> int:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        count = 0
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
            count += 1
    return count


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta(path: Path, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
    if rows is not None:
        value["data_rows"] = rows
    return value


def export_cart_signal_results(result: CartSignalExperimentResult, output_dir: str | Path, *, dataset_dir: str | Path, bias_results_dir: str | Path) -> dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    model_manifests = {}
    for signal in SIGNALS:
        directory = root / signal
        directory.mkdir(exist_ok=True)
        history = directory / "training_history.csv"
        metrics = directory / "metrics.csv"
        history_rows = _write_csv(history, result.histories[signal], HISTORY_COLUMNS)
        metric_rows = _write_csv(metrics, ({"model": signal, **row} for row in result.metrics[signal]), METRIC_COLUMNS)
        diagnostics = directory / "diagnostics.json"
        _write_json(diagnostics, result.diagnostics[signal])
        config = directory / "config.json"
        _write_json(config, {
            "experiment_version": EXPERIMENT_VERSION,
            "signal": signal,
            "confidence_formula": result.data[signal].spec.confidence_formula,
            "fixed_mf_config": result.config.as_dict(),
            "model": "BCE MF with item bias",
            "sampling": "Train-only Exposed Non-conversion with Random Unknown backfill; non-conversion is not a true negative",
            "selection_metric": "validation_purchase_ndcg_at_10",
            "test_policy": "single batched Test evaluation including frozen control checkpoint; no tuning",
        })
        artifacts = {path.name: _meta(path, rows) for path, rows in ((history, history_rows), (metrics, metric_rows))}
        artifacts.update({diagnostics.name: _meta(diagnostics), config.name: _meta(config)})
        if signal in TRAINED_SIGNALS:
            checkpoint = checkpoints / f"{signal}_best.pt"
            save_checkpoint(result.training[signal], checkpoint)
            artifacts[f"../checkpoints/{checkpoint.name}"] = _meta(checkpoint)
            best_epoch = result.training[signal].best_epoch
            best_ndcg = result.training[signal].best_validation_purchase_ndcg_at_10
        else:
            best_epoch = 9
            best_ndcg = max(float(row["validation_purchase_ndcg_at_10"]) for row in result.histories[signal])
        manifest = directory / "manifest.json"
        _write_json(manifest, {"signal": signal, "best_epoch": best_epoch, "best_validation_purchase_ndcg_at_10": best_ndcg, "artifacts": artifacts})
        model_manifests[signal] = _meta(manifest)
    tables = {
        "comparison.csv": (result.comparison, COMPARISON_COLUMNS),
        "coverage.csv": (result.coverage, COVERAGE_COLUMNS),
        "fallback_diagnostics.csv": (result.fallback, FALLBACK_COLUMNS),
        "personalization.csv": (result.personalization, PERSONALIZATION_COLUMNS),
        "bias_diagnostics.csv": (result.bias_rows, BIAS_COLUMNS),
        "signal_alignment.csv": (result.alignment, ALIGNMENT_COLUMNS),
    }
    artifacts = {}
    for filename, (rows, columns) in tables.items():
        path = root / filename
        artifacts[filename] = _meta(path, _write_csv(path, rows, columns))
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "signals": list(SIGNALS),
        "fixed_mf_config": result.config.as_dict(),
        "cart_centered_weights": {"log1p_view": 0.5, "favorite": 2, "cart": 6, "purchase": 10},
        "sampling": "Exposed Non-conversion with Random Unknown backfill, ratio 4, Train only",
        "true_negative_claim": False,
        "test_policy": "frozen control checkpoint plus three new models in one batched Test evaluation; no post-Test tuning",
        "dataset_manifest_sha256": _sha(Path(dataset_dir) / "manifest.json"),
        "bias_results_manifest_sha256": _sha(Path(bias_results_dir) / "manifest.json"),
        "model_manifests": model_manifests,
        "artifacts": artifacts,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest
