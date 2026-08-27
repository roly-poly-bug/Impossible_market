from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean, pstdev

from backend.app.db.models import EventType
from synthetic_data.config import ATTRIBUTE_NAMES
from synthetic_data.interaction_config import (
    ARCHETYPE_PRIMARY_CATEGORIES,
    DEFAULT_INTERACTION_SEED,
    EXPOSURE_SOURCE_WEIGHTS,
    INTERACTION_GENERATION_VERSION,
    VIEW_EXPLORATION_WEIGHT,
    VIEW_INTERCEPT,
    VIEW_NOISE_STDDEV,
    VIEW_NOVELTY_WEIGHT,
    VIEW_POPULARITY_WEIGHT,
    VIEW_PREFERENCE_WEIGHT,
    VIEW_PRICE_PENALTY,
)
from synthetic_data.product_generator import SyntheticProductRecord
from synthetic_data.session_generator import SyntheticSessionRecord
from synthetic_data.user_config import PREFERENCE_PRODUCT_MAPPING
from synthetic_data.user_generator import SyntheticUserRecord


@dataclass(frozen=True)
class SyntheticEventRecord:
    event_id: str
    session_id: str
    user_id: str
    product_id: str
    product_category: str
    event_type: EventType
    timestamp: datetime
    exposure_source: str
    preference_match: float
    initial_popularity: float
    product_novelty: float
    price_log10: float
    over_budget: bool
    view_probability: float
    user_archetype: str
    activity_tier: str
    generation_version: str
    generation_seed: int


def _stable_noise(product_name: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{product_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def initial_popularity(
    products: list[SyntheticProductRecord],
    seed: int,
) -> dict[str, float]:
    raw = {
        product.name: (
            0.20 * product.rarity
            + 0.18 * product.attributes["luxury"]
            + 0.17 * product.attributes["novelty"]
            + 0.45 * _stable_noise(product.name, seed)
        )
        for product in products
    }
    minimum = min(raw.values())
    maximum = max(raw.values())
    return {
        name: round((value - minimum) / (maximum - minimum), 6)
        for name, value in raw.items()
    }


def _attribute_standardization(
    products: list[SyntheticProductRecord],
) -> tuple[dict[str, float], dict[str, float]]:
    means = {
        name: fmean(product.attributes[name] for product in products)
        for name in ATTRIBUTE_NAMES
    }
    standard_deviations = {
        name: pstdev(product.attributes[name] for product in products)
        for name in ATTRIBUTE_NAMES
    }
    return means, standard_deviations


def preference_match(
    user: SyntheticUserRecord,
    product: SyntheticProductRecord,
    means: dict[str, float],
    standard_deviations: dict[str, float],
) -> float:
    """Centered, catalog-standardized match avoiding positive raw-cosine compression."""
    numerator = 0.0
    magnitude = 1.0
    for preference_name, attribute_name in PREFERENCE_PRODUCT_MAPPING.items():
        preference_weight = user.preferences[preference_name] - 0.5
        standardized_attribute = (
            (product.attributes[attribute_name] - means[attribute_name])
            / standard_deviations[attribute_name]
        )
        standardized_attribute = min(2.5, max(-2.5, standardized_attribute))
        numerator += preference_weight * standardized_attribute
        magnitude += abs(preference_weight)
    return 1.0 / (1.0 + math.exp(-2.2 * numerator / magnitude))


def _allocate_sources(total: int) -> list[str]:
    source_names = list(EXPOSURE_SOURCE_WEIGHTS)
    weight_total = sum(EXPOSURE_SOURCE_WEIGHTS.values())
    raw_counts = [total * EXPOSURE_SOURCE_WEIGHTS[name] / weight_total for name in source_names]
    counts = [math.floor(value) for value in raw_counts]
    for index in sorted(
        range(len(source_names)),
        key=lambda item: raw_counts[item] - counts[item],
        reverse=True,
    )[: total - sum(counts)]:
        counts[index] += 1
    return [
        source
        for source, count in zip(source_names, counts, strict=True)
        for _ in range(count)
    ]


def _continuity(
    previous: SyntheticProductRecord,
    candidate: SyntheticProductRecord,
) -> float:
    attribute_closeness = 1.0 - fmean(
        abs(previous.attributes[name] - candidate.attributes[name])
        for name in ATTRIBUTE_NAMES
    )
    return 0.65 * (previous.category_parent == candidate.category_parent) + 0.35 * attribute_closeness


def _weighted_choice(
    candidates: list[SyntheticProductRecord],
    weights: list[float],
    rng: random.Random,
) -> SyntheticProductRecord:
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights, strict=True):
        cumulative += weight
        if cumulative >= threshold:
            return candidate
    return candidates[-1]


def _candidate_weight(
    source: str,
    user: SyntheticUserRecord,
    product: SyntheticProductRecord,
    match: float,
    popularity: float,
    continuity_score: float | None,
) -> float:
    if source == "preference":
        weight = 0.05 + match**2.5
    elif source == "popular":
        weight = 0.05 + popularity**2.5
    elif source == "exploration":
        weight = (
            0.10
            + user.exploration_tendency
            * (0.55 * product.attributes["novelty"] + 0.45 * (1.0 - match))
            + (1.0 - user.exploration_tendency) * 0.35 * match
        )
        primary_categories = ARCHETYPE_PRIMARY_CATEGORIES.get(user.archetype)
        if primary_categories:
            outside_primary = product.category_parent not in primary_categories
            weight *= (
                1.0 + 1.20 * user.exploration_tendency
                if outside_primary
                else 1.0 + 0.80 * (1.0 - user.exploration_tendency)
            )
    else:
        weight = 1.0
    if continuity_score is not None:
        weight *= 1.0 + 0.30 * continuity_score
    return weight


def _view_probability(
    user: SyntheticUserRecord,
    product: SyntheticProductRecord,
    match: float,
    popularity: float,
    price_log10: float,
    rng: random.Random,
) -> float:
    price_gap = max(0.0, price_log10 - user.budget_log10)
    primary_categories = ARCHETYPE_PRIMARY_CATEGORIES.get(user.archetype)
    outside_primary = bool(
        primary_categories and product.category_parent not in primary_categories
    )
    exploration_value = (
        product.attributes["novelty"]
        + (1.0 - match)
        - 1.0
        + (0.35 if outside_primary else -0.10)
    )
    utility = (
        VIEW_INTERCEPT
        + VIEW_PREFERENCE_WEIGHT * (match - 0.5)
        + VIEW_POPULARITY_WEIGHT
        * user.popularity_preference
        * (popularity - 0.5)
        + VIEW_NOVELTY_WEIGHT * (product.attributes["novelty"] - 0.5)
        + VIEW_EXPLORATION_WEIGHT * user.exploration_tendency * exploration_value
        - VIEW_PRICE_PENALTY * user.price_sensitivity * min(price_gap, 8.0)
        + rng.gauss(0.0, VIEW_NOISE_STDDEV)
    )
    return 1.0 / (1.0 + math.exp(-utility))


def generate_events(
    sessions: list[SyntheticSessionRecord],
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
    *,
    seed: int = DEFAULT_INTERACTION_SEED,
) -> list[SyntheticEventRecord]:
    """Generate Impression then probabilistic View events for existing Sessions."""
    rng = random.Random(seed)
    users_by_id = {user.user_id: user for user in users}
    means, standard_deviations = _attribute_standardization(products)
    popularity = initial_popularity(products, seed)
    preference_matches = {
        user.user_id: {
            product.name: preference_match(user, product, means, standard_deviations)
            for product in products
        }
        for user in users
    }
    log_prices = {product.name: math.log10(float(product.price)) for product in products}
    continuity_scores = {
        (left.name, right.name): _continuity(left, right)
        for left in products
        for right in products
        if left.name != right.name
    }
    events = []
    event_index = 1

    for session in sessions:
        user = users_by_id[session.user_id]
        remaining = list(products)
        sources = _allocate_sources(session.impression_target)
        rng.shuffle(sources)
        previous_view = None
        timestamp = session.started_at + timedelta(seconds=1)
        user_matches = preference_matches[user.user_id]

        for source in sources:
            weights = [
                _candidate_weight(
                    source,
                    user,
                    product,
                    user_matches[product.name],
                    popularity[product.name],
                    (
                        continuity_scores[(previous_view.name, product.name)]
                        if previous_view is not None
                        else None
                    ),
                )
                for product in remaining
            ]
            product = _weighted_choice(remaining, weights, rng)
            remaining.remove(product)
            match = user_matches[product.name]
            price_log10 = log_prices[product.name]
            probability = _view_probability(
                user,
                product,
                match,
                popularity[product.name],
                price_log10,
                rng,
            )
            common = {
                "session_id": session.session_id,
                "user_id": user.user_id,
                "product_id": product.name,
                "product_category": product.category_parent,
                "exposure_source": source,
                "preference_match": round(match, 6),
                "initial_popularity": popularity[product.name],
                "product_novelty": product.attributes["novelty"],
                "price_log10": round(price_log10, 6),
                "over_budget": price_log10 > user.budget_log10,
                "view_probability": round(probability, 6),
                "user_archetype": user.archetype,
                "activity_tier": user.activity_tier.value,
                "generation_version": INTERACTION_GENERATION_VERSION,
                "generation_seed": seed,
            }
            events.append(
                SyntheticEventRecord(
                    event_id=f"synthetic-event-v1-{event_index:09d}",
                    event_type=EventType.IMPRESSION,
                    timestamp=timestamp,
                    **common,
                )
            )
            event_index += 1

            if rng.random() < probability:
                timestamp += timedelta(seconds=rng.randint(3, 18))
                events.append(
                    SyntheticEventRecord(
                        event_id=f"synthetic-event-v1-{event_index:09d}",
                        event_type=EventType.VIEW,
                        timestamp=timestamp,
                        **common,
                    )
                )
                event_index += 1
                previous_view = product
                timestamp += timedelta(seconds=rng.randint(2, 8))
            else:
                timestamp += timedelta(seconds=rng.randint(1, 5))

            if timestamp >= session.ended_at:
                raise RuntimeError(
                    f"Generated event sequence exceeded duration for {session.session_id}"
                )
    return events
