from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ml.baselines.popularity import (
    POPULARITY_SIGNALS,
    SIGNAL_CART,
    SIGNAL_FAVORITEPLUS,
    SIGNAL_PURCHASE,
    SIGNAL_TOTAL_VIEW,
    SIGNAL_WEIGHTED,
    TrainInteraction,
    WeightedSignalConfig,
    build_popularity_scores,
    deterministic_ranking,
    load_train_interactions,
)
from ml.evaluation.popularity import DEFAULT_K_VALUES, evaluate_popularity_ranking
from ml.representations.interaction import (
    REPRESENTATION_WEIGHTED,
    event_signal_overlap,
    heavy_user_view_analysis,
    interaction_representation_stats,
    representation_value,
)


EXPERIMENT_VERSION = "popularity_baseline_v1"
DATASET_VERSION = "recommendation_dataset_v1"
TASKS = ("viewplus", "favoriteplus", "purchase")
SPLITS = ("validation", "test")
TASK_MATCHED_SIGNALS = {
    "viewplus": SIGNAL_TOTAL_VIEW,
    "favoriteplus": SIGNAL_FAVORITEPLUS,
    "purchase": SIGNAL_PURCHASE,
}


@dataclass(frozen=True)
class PopularityExperimentResult:
    dataset_manifest: dict[str, object]
    weighted_config: WeightedSignalConfig
    metrics: tuple[dict[str, object], ...]
    purchase_cross_signal_metrics: tuple[dict[str, object], ...]
    representation_stats: tuple[dict[str, object], ...]
    signal_richness: tuple[dict[str, object], ...]
    stability: tuple[dict[str, object], ...]
    heavy_user: dict[str, object]
    signal_overlap: dict[str, int]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_relevance(
    dataset_dir: Path,
    split: str,
    task: str,
) -> dict[str, list[str]]:
    value = _read_json(dataset_dir / f"{split}_relevance_{task}.json")
    return value["relevant_items_by_user"]


def _load_seen(dataset_dir: Path, task: str) -> dict[str, list[str]]:
    value = _read_json(dataset_dir / f"train_seen_items_{task}.json")
    return value["items_by_user"]


def _signal_richness_rows(
    interactions: tuple[TrainInteraction, ...],
    metrics: list[dict[str, object]],
    weighted_config: WeightedSignalConfig,
) -> list[dict[str, object]]:
    definitions = {
        "view": (lambda row: row.was_viewed, SIGNAL_TOTAL_VIEW),
        "favoriteplus": (lambda row: row.was_favoriteplus, SIGNAL_FAVORITEPLUS),
        "cart": (lambda row: row.was_carted, SIGNAL_CART),
        "purchase": (lambda row: row.was_purchased, SIGNAL_PURCHASE),
        "weighted": (
            lambda row: representation_value(
                REPRESENTATION_WEIGHTED,
                row,
                weighted_config=weighted_config,
            )
            > 0,
            SIGNAL_WEIGHTED,
        ),
    }
    output = []
    for label, (predicate, signal) in definitions.items():
        positive_rows = [row for row in interactions if predicate(row)]
        purchase_metric = next(
            row
            for row in metrics
            if row["popularity_signal"] == signal
            and row["evaluation_task"] == "purchase"
            and row["split"] == "test"
            and row["k"] == 10
        )
        output.append(
            {
                "signal": label,
                "popularity_signal": signal,
                "nonzero_pair_count": len(positive_rows),
                "user_coverage": len({row.user_id for row in positive_rows}),
                "item_coverage": len({row.product_id for row in positive_rows}),
                "density": len(positive_rows) / len(interactions),
                "test_purchase_recall_at_10": purchase_metric["recall"],
                "test_purchase_ndcg_at_10": purchase_metric["ndcg"],
            }
        )
    return output


def _stability_rows(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (row["popularity_signal"], row["evaluation_task"], row["split"]): row
        for row in metrics
        if row["k"] == 10
    }
    output = []
    for signal in POPULARITY_SIGNALS:
        for task in TASKS:
            validation = by_key[(signal, task, "validation")]
            test = by_key[(signal, task, "test")]
            output.append(
                {
                    "popularity_signal": signal,
                    "evaluation_task": task,
                    "validation_recall_at_10": validation["recall"],
                    "test_recall_at_10": test["recall"],
                    "recall_difference_test_minus_validation": test["recall"]
                    - validation["recall"],
                    "validation_ndcg_at_10": validation["ndcg"],
                    "test_ndcg_at_10": test["ndcg"],
                    "ndcg_difference_test_minus_validation": test["ndcg"]
                    - validation["ndcg"],
                }
            )
    return output


def run_popularity_experiment(
    dataset_dir: str | Path,
    *,
    exclude_seen: bool = True,
    weighted_config: WeightedSignalConfig | None = None,
) -> PopularityExperimentResult:
    dataset_path = Path(dataset_dir)
    config = weighted_config or WeightedSignalConfig()
    dataset_manifest = _read_json(dataset_path / "manifest.json")
    if dataset_manifest["dataset_version"] != DATASET_VERSION:
        raise ValueError(f"Expected {DATASET_VERSION}")

    interactions = load_train_interactions(dataset_path / "train_viewplus.csv")
    if len(interactions) != 200_000:
        raise ValueError("Expected exactly 200,000 Train user-item pairs")
    product_ids = tuple(sorted({row.product_id for row in interactions}))
    scores = build_popularity_scores(
        interactions,
        product_ids,
        weighted_config=config,
    )
    candidates = _read_json(dataset_path / "candidate_sets.json")["policies"]
    rankings = {
        task: {
            signal: deterministic_ranking(
                scores[signal],
                candidates[task]["product_ids"],
            )
            for signal in POPULARITY_SIGNALS
        }
        for task in TASKS
    }

    metrics: list[dict[str, object]] = []
    for split in SPLITS:
        for task in TASKS:
            relevance = _load_relevance(dataset_path, split, task)
            seen = _load_seen(dataset_path, task)
            for signal in POPULARITY_SIGNALS:
                evaluated = evaluate_popularity_ranking(
                    rankings[task][signal],
                    relevance,
                    seen,
                    exclude_seen=exclude_seen,
                    k_values=DEFAULT_K_VALUES,
                )
                for values in evaluated:
                    metrics.append(
                        {
                            "popularity_signal": signal,
                            "evaluation_task": task,
                            "split": split,
                            "exclude_seen": exclude_seen,
                            **values,
                        }
                    )

    purchase_cross = [
        row for row in metrics if row["evaluation_task"] == "purchase"
    ]
    representation_stats = interaction_representation_stats(
        interactions,
        weighted_config=config,
    )
    richness = _signal_richness_rows(interactions, metrics, config)
    return PopularityExperimentResult(
        dataset_manifest=dataset_manifest,
        weighted_config=config,
        metrics=tuple(metrics),
        purchase_cross_signal_metrics=tuple(purchase_cross),
        representation_stats=tuple(representation_stats),
        signal_richness=tuple(richness),
        stability=tuple(_stability_rows(metrics)),
        heavy_user=heavy_user_view_analysis(interactions),
        signal_overlap=event_signal_overlap(interactions),
    )
