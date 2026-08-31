from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, pstdev

from ml.training.mf_data import IndexedInteractions, SampledTrainingData, sample_random_unknowns


BINARY_VIEW = "binary_view"
LOG_VIEW = "log_view"
FAVORITEPLUS = "favoriteplus"
WEIGHTED = "weighted"
PURCHASE_ONLY = "purchase_only"
REPRESENTATIONS = (BINARY_VIEW, LOG_VIEW, FAVORITEPLUS, WEIGHTED, PURCHASE_ONLY)


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    task: str
    source_file: str
    confidence_formula: str


SPECS = {
    BINARY_VIEW: RepresentationSpec(BINARY_VIEW, "viewplus", "train_viewplus.csv", "1"),
    LOG_VIEW: RepresentationSpec(LOG_VIEW, "viewplus", "train_viewplus.csv", "1 + 1.0 * log1p(view_count)"),
    FAVORITEPLUS: RepresentationSpec(FAVORITEPLUS, "favoriteplus", "train_favoriteplus.csv", "1"),
    WEIGHTED: RepresentationSpec(WEIGHTED, "viewplus", "train_viewplus.csv", "1 + log1p(weighted_strength)"),
    PURCHASE_ONLY: RepresentationSpec(PURCHASE_ONLY, "purchase", "train_purchase.csv", "1"),
}


@dataclass(frozen=True)
class RepresentationData:
    spec: RepresentationSpec
    indexed: IndexedInteractions
    sampled: SampledTrainingData
    positive_confidence: dict[tuple[int, int], float]
    coverage: dict[str, int | float | str]
    confidence_stats: dict[str, int | float | str]

    @property
    def weighted_bce_examples(self) -> tuple[tuple[int, int, float, float], ...]:
        positives = tuple(
            (user, item, 1.0, self.positive_confidence[(user, item)])
            for user, item in self.indexed.positive_pairs
        )
        unknowns = tuple(
            (user, unknown, 0.0, 1.0)
            for user, _, unknown in self.sampled.triples
        )
        return (*positives, *unknowns)


def weighted_strength(row: dict[str, str]) -> float:
    return (
        math.log1p(int(row["view_count"]))
        + 3.0 * float(int(row["favorite_count"]) > 0)
        + 5.0 * float(int(row["cart_count"]) > 0)
        + 8.0 * float(int(row["purchase_count"]) > 0)
    )


def positive_confidence(name: str, row: dict[str, str]) -> float:
    if name == LOG_VIEW:
        return 1.0 + math.log1p(int(row["view_count"]))
    if name == WEIGHTED:
        return 1.0 + math.log1p(weighted_strength(row))
    return 1.0


def _percentile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def summarize_confidence(name: str, values: list[float]) -> dict[str, int | float | str]:
    ordered = sorted(values)
    if not ordered or not all(math.isfinite(value) and value > 0 for value in ordered):
        raise ValueError("Positive confidence must be finite and greater than zero")
    return {
        "representation": name,
        "count": len(ordered),
        "mean": fmean(ordered),
        "std": pstdev(ordered),
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "median": median(ordered),
        "p75": _percentile(ordered, 0.75),
        "max": ordered[-1],
        "normalized": False,
    }


def load_representation(
    dataset_dir: str | Path,
    name: str,
    *,
    negative_ratio: int = 4,
    seed: int = 42,
) -> RepresentationData:
    spec = SPECS[name]
    path = Path(dataset_dir) / spec.source_file
    raw_rows: list[tuple[str, str, str, float]] = []
    users: set[str] = set()
    items: set[str] = set()
    with path.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            users.add(row["user_id"])
            items.add(row["product_id"])
            raw_rows.append(
                (
                    row["user_id"],
                    row["product_id"],
                    row["state"],
                    positive_confidence(name, row) if row["state"] == "positive" else 1.0,
                )
            )
    user_ids = tuple(sorted(users))
    item_ids = tuple(sorted(items))
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}
    positives: list[tuple[int, int]] = []
    confidence: dict[tuple[int, int], float] = {}
    unknowns: dict[int, list[int]] = {index: [] for index in range(len(user_ids))}
    non_conversions: set[tuple[int, int]] = set()
    for user_id, item_id, state, weight in raw_rows:
        pair = (user_to_index[user_id], item_to_index[item_id])
        if state == "positive":
            positives.append(pair)
            confidence[pair] = weight
        elif state == "unknown":
            unknowns[pair[0]].append(pair[1])
        elif state == "observed_non_conversion":
            non_conversions.add(pair)
        else:
            raise ValueError(f"Unexpected {spec.task} state: {state}")
    positive_users = {user for user, _ in positives}
    positive_items = {item for _, item in positives}
    indexed = IndexedInteractions(
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        positive_pairs=tuple(sorted(positives)),
        unknown_items_by_user={user: tuple(sorted(values)) for user, values in unknowns.items()},
        observed_non_conversion_pairs=frozenset(non_conversions),
        cold_user_indices=frozenset(set(range(len(user_ids))) - positive_users),
        cold_item_indices=frozenset(set(range(len(item_ids))) - positive_items),
    )
    sampled = sample_random_unknowns(indexed, negative_ratio=negative_ratio, seed=seed)
    values = [confidence[pair] for pair in indexed.positive_pairs]
    return RepresentationData(
        spec,
        indexed,
        sampled,
        confidence,
        {
            "representation": name,
            "positive_pair_count": len(positives),
            "training_user_count": len(positive_users),
            "training_item_count": len(positive_items),
            "cold_user_count": len(user_ids) - len(positive_users),
            "cold_item_count": len(item_ids) - len(positive_items),
            "density": len(positives) / (len(user_ids) * len(item_ids)),
        },
        summarize_confidence(name, values),
    )
