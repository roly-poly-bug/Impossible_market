from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

from ml.representations.mf_signal import RepresentationData, RepresentationSpec, summarize_confidence
from ml.training.mf_data import IndexedInteractions, SampledTrainingData
from ml.training.mf_negative_sampling import EXPOSED_NON_CONVERSION, NegativeSamplingResult, sample_non_positives


EXISTING_WEIGHTED = "existing_weighted"
CARTPLUS = "cartplus"
FAVORITE_CARTPLUS = "favorite_cartplus"
CART_CENTERED_WEIGHTED = "cart_centered_weighted"
SIGNALS = (EXISTING_WEIGHTED, CARTPLUS, FAVORITE_CARTPLUS, CART_CENTERED_WEIGHTED)


SPECS = {
    EXISTING_WEIGHTED: RepresentationSpec(EXISTING_WEIGHTED, "viewplus", "train_viewplus.csv", "1 + log1p(log1p(view_count) + 3F + 5C + 8P)"),
    CARTPLUS: RepresentationSpec(CARTPLUS, "cartplus", "train_viewplus.csv", "1 + log1p(5C + 8P)"),
    FAVORITE_CARTPLUS: RepresentationSpec(FAVORITE_CARTPLUS, "favorite_cartplus", "train_viewplus.csv", "1 + log1p(3F + 5C + 8P)"),
    CART_CENTERED_WEIGHTED: RepresentationSpec(CART_CENTERED_WEIGHTED, "viewplus", "train_viewplus.csv", "1 + log1p(0.5log1p(view_count) + 2F + 6C + 10P)"),
}


def cart_signal_strength(name: str, row: dict[str, str]) -> float:
    view = int(row["view_count"])
    favorite = float(int(row["favorite_count"]) > 0)
    cart = float(int(row["cart_count"]) > 0)
    purchase = float(int(row["purchase_count"]) > 0)
    if name == CARTPLUS:
        return 5.0 * cart + 8.0 * purchase
    if name == FAVORITE_CARTPLUS:
        return 3.0 * favorite + 5.0 * cart + 8.0 * purchase
    if name == CART_CENTERED_WEIGHTED:
        return 0.5 * math.log1p(view) + 2.0 * favorite + 6.0 * cart + 10.0 * purchase
    return math.log1p(view) + 3.0 * favorite + 5.0 * cart + 8.0 * purchase


def is_positive(name: str, row: dict[str, str]) -> bool:
    if name == CARTPLUS:
        return int(row["cart_count"]) > 0 or int(row["purchase_count"]) > 0
    if name == FAVORITE_CARTPLUS:
        return any(int(row[key]) > 0 for key in ("favorite_count", "cart_count", "purchase_count"))
    return int(row["view_count"]) > 0


def _state(name: str, row: dict[str, str]) -> str:
    if is_positive(name, row):
        return "positive"
    # Exposed means an observed Train impression/action opportunity without this
    # signal's target action. It remains a contrast label, never a true negative.
    if int(row["view_count"]) > 0 or int(row["favorite_count"]) > 0:
        return "observed_non_conversion"
    return "unknown"


def load_cart_signal(
    dataset_dir: str | Path,
    name: str,
    *,
    sample_ratio: int = 4,
    seed: int = 42,
) -> tuple[RepresentationData, NegativeSamplingResult]:
    if name not in SIGNALS:
        raise ValueError(f"Unknown Cart signal: {name}")
    path = Path(dataset_dir) / "train_viewplus.csv"
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    user_ids = tuple(sorted({row["user_id"] for row in rows}))
    item_ids = tuple(sorted({row["product_id"] for row in rows}))
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}
    positives: list[tuple[int, int]] = []
    confidence: dict[tuple[int, int], float] = {}
    unknowns: dict[int, list[int]] = {index: [] for index in range(len(user_ids))}
    exposed: set[tuple[int, int]] = set()
    for row in rows:
        pair = (user_to_index[row["user_id"]], item_to_index[row["product_id"]])
        state = _state(name, row)
        if state == "positive":
            positives.append(pair)
            confidence[pair] = 1.0 + math.log1p(cart_signal_strength(name, row))
        elif state == "observed_non_conversion":
            exposed.add(pair)
        else:
            unknowns[pair[0]].append(pair[1])
    positive_users = {user for user, _ in positives}
    positive_items = {item for _, item in positives}
    indexed = IndexedInteractions(
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        positive_pairs=tuple(sorted(positives)),
        unknown_items_by_user={user: tuple(sorted(values)) for user, values in unknowns.items()},
        observed_non_conversion_pairs=frozenset(exposed),
        cold_user_indices=frozenset(set(range(len(user_ids))) - positive_users),
        cold_item_indices=frozenset(set(range(len(item_ids))) - positive_items),
    )
    sampled = sample_non_positives(indexed, EXPOSED_NON_CONVERSION, sample_ratio=sample_ratio, seed=seed)
    values = [confidence[pair] for pair in indexed.positive_pairs]
    data = RepresentationData(
        SPECS[name], indexed, sampled.sampled, confidence,
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
    return data, sampled
