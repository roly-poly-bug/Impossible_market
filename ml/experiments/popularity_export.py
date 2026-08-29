from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from ml.baselines.popularity import POPULARITY_SIGNALS
from ml.evaluation.popularity import DEFAULT_K_VALUES
from ml.experiments.popularity_experiment import (
    DATASET_VERSION,
    EXPERIMENT_VERSION,
    TASK_MATCHED_SIGNALS,
    PopularityExperimentResult,
)


METRIC_COLUMNS = (
    "popularity_signal",
    "evaluation_task",
    "split",
    "exclude_seen",
    "k",
    "eligible_users",
    "recall",
    "ndcg",
    "hit_rate",
    "precision",
)
REPRESENTATION_COLUMNS = (
    "representation",
    "nonzero_pair_count",
    "density",
    "mean_nonzero",
    "std_nonzero",
    "min_nonzero",
    "p25_nonzero",
    "median_nonzero",
    "p75_nonzero",
    "max_nonzero",
)
RICHNESS_COLUMNS = (
    "signal",
    "popularity_signal",
    "nonzero_pair_count",
    "user_coverage",
    "item_coverage",
    "density",
    "test_purchase_recall_at_10",
    "test_purchase_ndcg_at_10",
)
STABILITY_COLUMNS = (
    "popularity_signal",
    "evaluation_task",
    "validation_recall_at_10",
    "test_recall_at_10",
    "recall_difference_test_minus_validation",
    "validation_ndcg_at_10",
    "test_ndcg_at_10",
    "ndcg_difference_test_minus_validation",
)


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    columns: tuple[str, ...],
) -> int:
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
    value: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        value["data_rows"] = rows
    return value


def export_popularity_results(
    result: PopularityExperimentResult,
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    exclude_seen: bool,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {}

    csv_outputs = (
        ("metrics.csv", result.metrics, METRIC_COLUMNS),
        (
            "purchase_cross_signal_metrics.csv",
            result.purchase_cross_signal_metrics,
            METRIC_COLUMNS,
        ),
        (
            "interaction_representation_stats.csv",
            result.representation_stats,
            REPRESENTATION_COLUMNS,
        ),
        ("signal_richness.csv", result.signal_richness, RICHNESS_COLUMNS),
        ("validation_test_stability.csv", result.stability, STABILITY_COLUMNS),
    )
    for filename, rows, columns in csv_outputs:
        path = destination / filename
        row_count = _write_csv(path, rows, columns)
        artifacts[filename] = _metadata(path, row_count)

    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_version": DATASET_VERSION,
        "score_source_split": "train_only",
        "personalization": False,
        "ranking_scope": "same global ranking before task-specific seen exclusion",
        "exclude_seen": exclude_seen,
        "k_values": list(DEFAULT_K_VALUES),
        "popularity_signals": list(POPULARITY_SIGNALS),
        "task_matched_signals": TASK_MATCHED_SIGNALS,
        "weighted_implicit_v1": result.weighted_config.as_dict(),
        "favoriteplus_definition": "one point per unique Train user-item pair with favorite, cart, or purchase",
        "tie_break": "score descending, product_id ascending",
        "cold_item_policy": "retain candidates with score 0 and order them by product_id",
        "candidate_policy_source": "recommendation_dataset_v1/candidate_sets.json",
        "seen_policy_source": "task-specific recommendation_dataset_v1 Train positive items",
        "zero_representation_policy": "0 means no observed positive signal for this representation; it is not a true negative",
        "ground_truth_policy": "hidden user preferences, product attributes, archetypes, preference_match, and future events are not loaded",
    }
    config_path = destination / "config.json"
    _write_json(config_path, config)
    artifacts[config_path.name] = _metadata(config_path)

    analysis_path = destination / "analysis.json"
    _write_json(
        analysis_path,
        {
            "heavy_user": result.heavy_user,
            "signal_overlap": result.signal_overlap,
        },
    )
    artifacts[analysis_path.name] = _metadata(analysis_path)

    dataset_manifest_path = Path(dataset_dir) / "manifest.json"
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "dataset_version": DATASET_VERSION,
        "dataset_manifest": {
            "path": dataset_manifest_path.as_posix(),
            "sha256": _sha256(dataset_manifest_path),
        },
        "score_source_split": "train_only",
        "exclude_seen": exclude_seen,
        "deterministic_tie_break": "score_desc_product_id_asc",
        "weighted_config": result.weighted_config.as_dict(),
        "artifacts": dict(sorted(artifacts.items())),
    }
    manifest_path = destination / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest
