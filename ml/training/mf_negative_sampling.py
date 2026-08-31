from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean, median, pstdev

from ml.training.mf_data import IndexedInteractions, SampledTrainingData


RANDOM_UNKNOWN = "random_unknown"
EXPOSED_NON_CONVERSION = "exposed_non_conversion"
MIXED = "mixed"
STRATEGIES = (RANDOM_UNKNOWN, EXPOSED_NON_CONVERSION, MIXED)


@dataclass(frozen=True)
class SamplingRecord:
    user: int
    positive_item: int
    sampled_item: int
    source: str
    is_backfill: bool = False


@dataclass(frozen=True)
class NegativeSamplingResult:
    strategy: str
    sampled: SampledTrainingData
    records: tuple[SamplingRecord, ...]
    enough_exposed_user_indices: frozenset[int]
    backfill_user_indices: frozenset[int]


def _exposed_by_user(data: IndexedInteractions) -> dict[int, tuple[int, ...]]:
    values: defaultdict[int, list[int]] = defaultdict(list)
    for user, item in data.observed_non_conversion_pairs:
        values[user].append(item)
    return {user: tuple(sorted(items)) for user, items in values.items()}


def sample_non_positives(
    data: IndexedInteractions,
    strategy: str,
    *,
    sample_ratio: int = 4,
    seed: int = 42,
) -> NegativeSamplingResult:
    if sample_ratio != 4:
        raise ValueError("mf_negative_sampling_v1 fixes sample_ratio at 4")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    generator = random.Random(seed)
    exposed = _exposed_by_user(data)
    exposed_target = 0 if strategy == RANDOM_UNKNOWN else (4 if strategy == EXPOSED_NON_CONVERSION else 2)
    enough_users = {
        user for user, _ in data.positive_pairs
        if len(exposed.get(user, ())) >= exposed_target
    }
    backfill_users: set[int] = set()
    records: list[SamplingRecord] = []
    positives = set(data.positive_pairs)
    for user, positive_item in data.positive_pairs:
        exposed_pool = exposed.get(user, ())
        selected_exposed = (
            generator.sample(exposed_pool, min(exposed_target, len(exposed_pool)))
            if exposed_target else []
        )
        for item in selected_exposed:
            records.append(SamplingRecord(user, positive_item, item, "exposed_non_conversion"))
        backfill = exposed_target - len(selected_exposed)
        ordinary_unknown = sample_ratio - exposed_target
        unknown_count = ordinary_unknown + backfill
        if backfill:
            backfill_users.add(user)
        unknown_pool = data.unknown_items_by_user[user]
        if len(unknown_pool) < unknown_count:
            raise ValueError(f"User {user} has too few Unknown candidates")
        selected_unknown = generator.sample(unknown_pool, unknown_count)
        for index, item in enumerate(selected_unknown):
            records.append(
                SamplingRecord(
                    user,
                    positive_item,
                    item,
                    "random_unknown",
                    is_backfill=index >= ordinary_unknown,
                )
            )
    for record in records:
        pair = (record.user, record.sampled_item)
        if pair in positives:
            raise AssertionError("A positive pair was sampled as non-positive")
        if record.source == "exposed_non_conversion" and pair not in data.observed_non_conversion_pairs:
            raise AssertionError("Exposed sampling crossed the Observed Non-conversion boundary")
        if record.source == "random_unknown" and record.sampled_item not in data.unknown_items_by_user[record.user]:
            raise AssertionError("Unknown sampling crossed the Unknown boundary")
    triples = tuple((row.user, row.positive_item, row.sampled_item) for row in records)
    return NegativeSamplingResult(
        strategy,
        SampledTrainingData(sample_ratio, seed, triples),
        tuple(records),
        frozenset(enough_users),
        frozenset(backfill_users),
    )


def _gini(counts: list[int]) -> float:
    if not counts or sum(counts) == 0:
        return 0.0
    ordered = sorted(counts)
    total = sum(ordered)
    n = len(ordered)
    return sum((2 * index - n - 1) * value for index, value in enumerate(ordered, 1)) / (n * total)


def sampling_statistics(
    result: NegativeSamplingResult,
    *,
    user_count: int,
    item_count: int,
) -> dict[str, int | float | str]:
    pairs = {(row.user, row.sampled_item) for row in result.records}
    item_counts = Counter(row.sampled_item for row in result.records)
    exposed_count = sum(row.source == "exposed_non_conversion" for row in result.records)
    unknown_count = len(result.records) - exposed_count
    backfill_count = sum(row.is_backfill for row in result.records)
    top10_count = sum(value for _, value in item_counts.most_common(10))
    positive_users = {row.user for row in result.records}
    return {
        "strategy": result.strategy,
        "total_sampled_pairs": len(result.records),
        "unique_sampled_user_item_pairs": len(pairs),
        "random_unknown_count": unknown_count,
        "exposed_non_conversion_count": exposed_count,
        "exposed_share": exposed_count / len(result.records),
        "backfill_count": backfill_count,
        "backfill_rate": backfill_count / len(result.records),
        "training_user_count": len(positive_users),
        "users_with_enough_exposed_pool": len(result.enough_exposed_user_indices),
        "users_requiring_backfill": len(result.backfill_user_indices),
        "sampled_item_unique_count": len(item_counts),
        "items_never_sampled": item_count - len(item_counts),
        "sampled_item_gini": _gini(list(item_counts.values())),
        "top10_sampled_share": top10_count / len(result.records),
    }


def hardness_diagnostics(
    result: NegativeSamplingResult,
    *,
    item_cart_scores: dict[int, float],
    item_exposure_counts: dict[int, float],
) -> list[dict[str, int | float | str]]:
    rows = []
    for source in ("all", "random_unknown", "exposed_non_conversion"):
        selected = [
            row for row in result.records
            if source == "all" or row.source == source
        ]
        if not selected:
            continue
        carts = [item_cart_scores[row.sampled_item] for row in selected]
        exposures = [item_exposure_counts[row.sampled_item] for row in selected]
        rows.append(
            {
                "strategy": result.strategy,
                "sample_source": source,
                "sample_count": len(selected),
                "mean_item_cart_popularity": fmean(carts),
                "median_item_cart_popularity": median(carts),
                "std_item_cart_popularity": pstdev(carts),
                "mean_item_train_exposure": fmean(exposures),
                "median_item_train_exposure": median(exposures),
                "std_item_train_exposure": pstdev(exposures),
                "all_values_finite": all(math.isfinite(value) for value in (*carts, *exposures)),
            }
        )
    return rows
