from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexedInteractions:
    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    positive_pairs: tuple[tuple[int, int], ...]
    unknown_items_by_user: dict[int, tuple[int, ...]]
    observed_non_conversion_pairs: frozenset[tuple[int, int]]
    cold_user_indices: frozenset[int]
    cold_item_indices: frozenset[int]


@dataclass(frozen=True)
class SampledTrainingData:
    negative_ratio: int
    seed: int
    triples: tuple[tuple[int, int, int], ...]

    @property
    def bce_examples(self) -> tuple[tuple[int, int, float], ...]:
        positives = {(user, item) for user, item, _ in self.triples}
        positive_rows = tuple((user, item, 1.0) for user, item in sorted(positives))
        unknown_rows = tuple((user, unknown, 0.0) for user, _, unknown in self.triples)
        return (*positive_rows, *unknown_rows)


def load_binary_viewplus_training(path: str | Path) -> IndexedInteractions:
    raw_rows = []
    users = set()
    items = set()
    with Path(path).open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            users.add(row["user_id"])
            items.add(row["product_id"])
            raw_rows.append((row["user_id"], row["product_id"], row["state"]))
    user_ids = tuple(sorted(users))
    item_ids = tuple(sorted(items))
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index for index, value in enumerate(item_ids)}
    positives = []
    unknowns: dict[int, list[int]] = {index: [] for index in range(len(user_ids))}
    non_conversions = set()
    for user_id, item_id, state in raw_rows:
        pair = (user_to_index[user_id], item_to_index[item_id])
        if state == "positive":
            positives.append(pair)
        elif state == "unknown":
            unknowns[pair[0]].append(pair[1])
        elif state == "observed_non_conversion":
            non_conversions.add(pair)
        else:
            raise ValueError(f"Unexpected View+ state: {state}")
    positive_users = {user for user, _ in positives}
    positive_items = {item for _, item in positives}
    return IndexedInteractions(
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


def sample_random_unknowns(
    data: IndexedInteractions,
    *,
    negative_ratio: int,
    seed: int,
) -> SampledTrainingData:
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")
    generator = random.Random(seed)
    triples = []
    positives = set(data.positive_pairs)
    for user, positive_item in data.positive_pairs:
        pool = data.unknown_items_by_user[user]
        if len(pool) < negative_ratio:
            raise ValueError(f"User {user} has too few Unknown candidates")
        for unknown_item in generator.sample(pool, negative_ratio):
            pair = (user, unknown_item)
            if pair in positives or pair in data.observed_non_conversion_pairs:
                raise AssertionError("Sampler crossed the View+ Unknown boundary")
            triples.append((user, positive_item, unknown_item))
    return SampledTrainingData(negative_ratio, seed, tuple(triples))
