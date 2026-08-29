from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SIGNAL_TOTAL_VIEW = "total_view_count"
SIGNAL_UNIQUE_VIEW_USERS = "unique_view_users"
SIGNAL_LOG_VIEW = "log_view_strength"
SIGNAL_FAVORITEPLUS = "favoriteplus_unique_pairs"
SIGNAL_CART = "cart_count"
SIGNAL_PURCHASE = "purchase_count"
SIGNAL_WEIGHTED = "weighted_popularity_v1"
POPULARITY_SIGNALS = (
    SIGNAL_TOTAL_VIEW,
    SIGNAL_UNIQUE_VIEW_USERS,
    SIGNAL_LOG_VIEW,
    SIGNAL_FAVORITEPLUS,
    SIGNAL_CART,
    SIGNAL_PURCHASE,
    SIGNAL_WEIGHTED,
)


@dataclass(frozen=True)
class WeightedSignalConfig:
    view_weight: float = 1.0
    favorite_weight: float = 3.0
    cart_weight: float = 5.0
    purchase_weight: float = 8.0
    view_transform: str = "log1p"

    def __post_init__(self) -> None:
        if self.view_transform != "log1p":
            raise ValueError("Weighted Baseline v1 supports only log1p view transform")
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("Popularity weights must be non-negative")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "view": self.view_weight,
            "favorite": self.favorite_weight,
            "cart": self.cart_weight,
            "purchase": self.purchase_weight,
        }

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "formula": "1*sum_user(log1p(view_count_ui)) + 3*favorite_pairs + 5*cart_pairs + 8*purchase_pairs"}


@dataclass(frozen=True)
class TrainInteraction:
    user_id: str
    product_id: str
    impression_count: int
    view_count: int
    favorite_count: int
    cart_count: int
    purchase_count: int

    @property
    def was_viewed(self) -> bool:
        return self.view_count > 0

    @property
    def was_favorited(self) -> bool:
        return self.favorite_count > 0

    @property
    def was_carted(self) -> bool:
        return self.cart_count > 0

    @property
    def was_purchased(self) -> bool:
        return self.purchase_count > 0

    @property
    def was_favoriteplus(self) -> bool:
        return self.was_favorited or self.was_carted or self.was_purchased


def load_train_interactions(path: str | Path) -> tuple[TrainInteraction, ...]:
    interactions = []
    with Path(path).open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            interactions.append(
                TrainInteraction(
                    user_id=row["user_id"],
                    product_id=row["product_id"],
                    impression_count=int(row["impression_count"]),
                    view_count=int(row["view_count"]),
                    favorite_count=int(row["favorite_count"]),
                    cart_count=int(row["cart_count"]),
                    purchase_count=int(row["purchase_count"]),
                )
            )
    return tuple(interactions)


def _empty_scores(product_ids: Sequence[str]) -> dict[str, float]:
    return {product_id: 0.0 for product_id in product_ids}


def build_popularity_scores(
    interactions: Iterable[TrainInteraction],
    product_ids: Sequence[str],
    *,
    weighted_config: WeightedSignalConfig,
) -> dict[str, dict[str, float]]:
    scores = {signal: _empty_scores(product_ids) for signal in POPULARITY_SIGNALS}
    known_products = set(product_ids)
    for row in interactions:
        if row.product_id not in known_products:
            raise ValueError(f"Unknown Product in Train interactions: {row.product_id}")
        log_view = math.log1p(row.view_count)
        scores[SIGNAL_TOTAL_VIEW][row.product_id] += row.view_count
        scores[SIGNAL_UNIQUE_VIEW_USERS][row.product_id] += int(row.was_viewed)
        scores[SIGNAL_LOG_VIEW][row.product_id] += log_view
        scores[SIGNAL_FAVORITEPLUS][row.product_id] += int(row.was_favoriteplus)
        scores[SIGNAL_CART][row.product_id] += row.cart_count
        scores[SIGNAL_PURCHASE][row.product_id] += row.purchase_count
        scores[SIGNAL_WEIGHTED][row.product_id] += (
            weighted_config.view_weight * log_view
            + weighted_config.favorite_weight * int(row.was_favorited)
            + weighted_config.cart_weight * int(row.was_carted)
            + weighted_config.purchase_weight * int(row.was_purchased)
        )
    return scores


def deterministic_ranking(
    scores: Mapping[str, float],
    candidate_ids: Iterable[str],
) -> tuple[str, ...]:
    candidates = tuple(candidate_ids)
    missing = set(candidates) - set(scores)
    if missing:
        raise ValueError(f"Candidate Products are missing scores: {sorted(missing)[:5]}")
    return tuple(sorted(candidates, key=lambda product_id: (-scores[product_id], product_id)))


def recommend_from_global_ranking(
    ranking: Sequence[str],
    *,
    seen_items: Iterable[str] = (),
    exclude_seen: bool = True,
    k: int,
) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    seen = set(seen_items) if exclude_seen else set()
    return [item for item in ranking if item not in seen][:k]
