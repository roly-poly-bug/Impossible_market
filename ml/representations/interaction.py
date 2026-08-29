from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import fmean, median, pstdev
from typing import Iterable

from ml.baselines.popularity import TrainInteraction, WeightedSignalConfig


REPRESENTATION_BINARY_VIEWPLUS = "binary_viewplus"
REPRESENTATION_VIEW_COUNT = "view_count"
REPRESENTATION_LOG_VIEW = "log_view_count"
REPRESENTATION_FAVORITEPLUS = "binary_favoriteplus"
REPRESENTATION_PURCHASE = "binary_purchase"
REPRESENTATION_WEIGHTED = "weighted_implicit_v1"
REPRESENTATIONS = (
    REPRESENTATION_BINARY_VIEWPLUS,
    REPRESENTATION_VIEW_COUNT,
    REPRESENTATION_LOG_VIEW,
    REPRESENTATION_FAVORITEPLUS,
    REPRESENTATION_PURCHASE,
    REPRESENTATION_WEIGHTED,
)


def representation_value(
    name: str,
    row: TrainInteraction,
    *,
    weighted_config: WeightedSignalConfig,
) -> float:
    if name == REPRESENTATION_BINARY_VIEWPLUS:
        return float(row.was_viewed)
    if name == REPRESENTATION_VIEW_COUNT:
        return float(row.view_count)
    if name == REPRESENTATION_LOG_VIEW:
        return math.log1p(row.view_count)
    if name == REPRESENTATION_FAVORITEPLUS:
        return float(row.was_favoriteplus)
    if name == REPRESENTATION_PURCHASE:
        return float(row.was_purchased)
    if name == REPRESENTATION_WEIGHTED:
        return (
            weighted_config.view_weight * math.log1p(row.view_count)
            + weighted_config.favorite_weight * float(row.was_favorited)
            + weighted_config.cart_weight * float(row.was_carted)
            + weighted_config.purchase_weight * float(row.was_purchased)
        )
    raise ValueError(f"Unknown interaction representation: {name}")


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile for no values")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def interaction_representation_stats(
    interactions: Iterable[TrainInteraction],
    *,
    weighted_config: WeightedSignalConfig,
) -> list[dict[str, float | int | str | None]]:
    rows = tuple(interactions)
    total_pairs = len(rows)
    results = []
    for representation in REPRESENTATIONS:
        values = sorted(
            value
            for row in rows
            if (
                value := representation_value(
                    representation,
                    row,
                    weighted_config=weighted_config,
                )
            )
            > 0
        )
        summary = {
            "mean_nonzero": fmean(values) if values else None,
            "std_nonzero": pstdev(values) if values else None,
            "min_nonzero": min(values) if values else None,
            "p25_nonzero": _percentile(values, 0.25) if values else None,
            "median_nonzero": median(values) if values else None,
            "p75_nonzero": _percentile(values, 0.75) if values else None,
            "max_nonzero": max(values) if values else None,
        }
        results.append(
            {
                "representation": representation,
                "nonzero_pair_count": len(values),
                "density": len(values) / total_pairs,
                **summary,
            }
        )
    return results


def heavy_user_view_analysis(
    interactions: Iterable[TrainInteraction],
) -> dict[str, object]:
    raw_by_user: defaultdict[str, float] = defaultdict(float)
    log_by_user: defaultdict[str, float] = defaultdict(float)
    for row in interactions:
        raw_by_user[row.user_id] += row.view_count
        log_by_user[row.user_id] += math.log1p(row.view_count)
    users = sorted(set(raw_by_user) | set(log_by_user))
    raw_values = [raw_by_user[user] for user in users]
    log_values = [log_by_user[user] for user in users]
    top_count = max(1, math.ceil(len(users) * 0.01))

    ranked_users = sorted(users, key=lambda user: (-raw_by_user[user], user))
    top_users = ranked_users[:top_count]
    decile_rows = []
    for index in range(10):
        start = index * len(ranked_users) // 10
        end = (index + 1) * len(ranked_users) // 10
        members = ranked_users[start:end]
        decile_rows.append(
            {
                "observed_activity_decile": index + 1,
                "user_count": len(members),
                "mean_raw_view_strength": fmean(raw_by_user[user] for user in members),
                "mean_log_view_strength": fmean(log_by_user[user] for user in members),
            }
        )
    return {
        "user_count": len(users),
        "top_1_percent_user_count": top_count,
        "top_1_percent_raw_share": sum(raw_by_user[user] for user in top_users)
        / sum(raw_values),
        "top_1_percent_log_share": sum(log_by_user[user] for user in top_users)
        / sum(log_values),
        "max_raw_strength": max(raw_values),
        "max_log_strength": max(log_values),
        "activity_deciles": decile_rows,
        "activity_tier_note": "Deciles use observed Train view strength only; hidden synthetic activity tier is not loaded.",
    }


def event_signal_overlap(
    interactions: Iterable[TrainInteraction],
) -> dict[str, int]:
    view_pairs = set()
    favorite_pairs = set()
    purchase_pairs = set()
    max_counts = Counter()
    for row in interactions:
        pair = (row.user_id, row.product_id)
        if row.was_viewed:
            view_pairs.add(pair)
        if row.was_favoriteplus:
            favorite_pairs.add(pair)
        if row.was_purchased:
            purchase_pairs.add(pair)
        max_counts["favorite"] = max(max_counts["favorite"], row.favorite_count)
        max_counts["cart"] = max(max_counts["cart"], row.cart_count)
        max_counts["purchase"] = max(max_counts["purchase"], row.purchase_count)
    return {
        "view_pairs": len(view_pairs),
        "favoriteplus_pairs": len(favorite_pairs),
        "purchase_pairs": len(purchase_pairs),
        "favoriteplus_outside_view": len(favorite_pairs - view_pairs),
        "purchase_outside_view": len(purchase_pairs - view_pairs),
        "purchase_outside_favoriteplus": len(purchase_pairs - favorite_pairs),
        "max_favorite_count_per_pair": max_counts["favorite"],
        "max_cart_count_per_pair": max_counts["cart"],
        "max_purchase_count_per_pair": max_counts["purchase"],
    }
