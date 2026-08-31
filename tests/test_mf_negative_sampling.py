from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from ml.experiments.mf_negative_sampling_export import _write_csv
from ml.representations.mf_signal import WEIGHTED, load_representation
from ml.training.mf_data import IndexedInteractions
from ml.training.mf_negative_sampling import (
    EXPOSED_NON_CONVERSION,
    MIXED,
    RANDOM_UNKNOWN,
    sample_non_positives,
    sampling_statistics,
)


def _indexed() -> IndexedInteractions:
    users = ("u0", "u1")
    items = tuple("abcdefghi")
    exposed = frozenset(
        {(0, item) for item in (1, 2, 3, 4)} | {(1, 1)}
    )
    unknowns = {0: (5, 6, 7, 8), 1: (2, 3, 4, 5, 6, 7, 8)}
    return IndexedInteractions(
        users, items, {value: index for index, value in enumerate(users)},
        {value: index for index, value in enumerate(items)},
        ((0, 0), (1, 0)), unknowns, exposed, frozenset(), frozenset({1, 2, 3, 4, 5, 6, 7, 8}),
    )


def test_random_unknown_sampling_correctness() -> None:
    result = sample_non_positives(_indexed(), RANDOM_UNKNOWN)
    assert len(result.records) == 8
    assert {row.source for row in result.records} == {"random_unknown"}
    assert all(row.sampled_item in _indexed().unknown_items_by_user[row.user] for row in result.records)


def test_exposed_sampling_and_backfill_correctness() -> None:
    result = sample_non_positives(_indexed(), EXPOSED_NON_CONVERSION)
    u0 = [row for row in result.records if row.user == 0]
    u1 = [row for row in result.records if row.user == 1]
    assert sum(row.source == "exposed_non_conversion" for row in u0) == 4
    assert sum(row.source == "exposed_non_conversion" for row in u1) == 1
    assert sum(row.is_backfill for row in u1) == 3
    assert result.backfill_user_indices == frozenset({1})


def test_mixed_uses_fixed_two_two_and_backfills_shortage() -> None:
    result = sample_non_positives(_indexed(), MIXED)
    u0 = [row for row in result.records if row.user == 0]
    u1 = [row for row in result.records if row.user == 1]
    assert sum(row.source == "exposed_non_conversion" for row in u0) == 2
    assert sum(row.source == "random_unknown" for row in u0) == 2
    assert sum(row.source == "exposed_non_conversion" for row in u1) == 1
    assert sum(row.source == "random_unknown" for row in u1) == 3
    assert sum(row.is_backfill for row in u1) == 1


def test_positive_never_sampled_and_no_duplicate_per_positive() -> None:
    positives = set(_indexed().positive_pairs)
    for strategy in (RANDOM_UNKNOWN, EXPOSED_NON_CONVERSION, MIXED):
        result = sample_non_positives(_indexed(), strategy)
        assert all((row.user, row.sampled_item) not in positives for row in result.records)
        for user, positive in positives:
            values = [row.sampled_item for row in result.records if (row.user, row.positive_item) == (user, positive)]
            assert len(values) == len(set(values)) == 4


def test_sampling_is_deterministic() -> None:
    for strategy in (RANDOM_UNKNOWN, EXPOSED_NON_CONVERSION, MIXED):
        assert sample_non_positives(_indexed(), strategy, seed=42) == sample_non_positives(_indexed(), strategy, seed=42)


def test_sampling_stats_are_correct() -> None:
    result = sample_non_positives(_indexed(), EXPOSED_NON_CONVERSION)
    stats = sampling_statistics(result, user_count=2, item_count=9)
    assert stats["total_sampled_pairs"] == 8
    assert stats["exposed_non_conversion_count"] == 5
    assert stats["random_unknown_count"] == 3
    assert stats["backfill_count"] == 3
    assert stats["users_requiring_backfill"] == 1
    assert 0 <= stats["sampled_item_gini"] <= 1


def _write_viewplus(path: Path) -> None:
    fields = (
        "user_id", "product_id", "task", "state", "is_positive",
        "is_observed_non_conversion", "is_unknown", "impression_count", "view_count",
        "favorite_count", "cart_count", "purchase_count", "first_interaction_at", "last_interaction_at",
        "validation_purchase", "test_purchase",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, product in enumerate("abcdef"):
            state = "positive" if index == 0 else ("observed_non_conversion" if index == 1 else "unknown")
            writer.writerow(
                {
                    "user_id": "u", "product_id": product, "task": "viewplus", "state": state,
                    "is_positive": int(state == "positive"),
                    "is_observed_non_conversion": int(state == "observed_non_conversion"),
                    "is_unknown": int(state == "unknown"), "impression_count": int(state != "unknown"),
                    "view_count": int(state == "positive"), "favorite_count": 0, "cart_count": 0,
                    "purchase_count": 0, "first_interaction_at": "", "last_interaction_at": "",
                    "validation_purchase": 1, "test_purchase": 1,
                }
            )


def test_exposed_pair_has_impression_and_no_view_and_no_future_leakage(tmp_path: Path) -> None:
    _write_viewplus(tmp_path / "train_viewplus.csv")
    data = load_representation(tmp_path, WEIGHTED)
    pair = next(iter(data.indexed.observed_non_conversion_pairs))
    assert data.indexed.item_ids[pair[1]] == "b"
    result = sample_non_positives(data.indexed, EXPOSED_NON_CONVERSION)
    assert any(row.source == "exposed_non_conversion" and row.sampled_item == pair[1] for row in result.records)


def test_positive_confidence_is_unchanged_by_sampling(tmp_path: Path) -> None:
    _write_viewplus(tmp_path / "train_viewplus.csv")
    data = load_representation(tmp_path, WEIGHTED)
    sampled = sample_non_positives(data.indexed, MIXED)
    changed = replace(data, sampled=sampled.sampled)
    assert changed.positive_confidence == data.positive_confidence


def test_observed_non_conversion_is_not_named_true_negative() -> None:
    result = sample_non_positives(_indexed(), EXPOSED_NON_CONVERSION)
    assert all("true_negative" not in row.source for row in result.records)


def test_result_csv_artifact_is_reproducible(tmp_path: Path) -> None:
    rows = [{"strategy": "mixed", "count": 8}]
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    _write_csv(first, rows, ("strategy", "count"))
    _write_csv(second, rows, ("strategy", "count"))
    assert first.read_bytes() == second.read_bytes()
