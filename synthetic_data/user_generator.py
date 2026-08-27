from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean, median, pstdev

from backend.app.db.models import ActivityTier, BudgetTier
from synthetic_data.user_archetypes import ARCHETYPE_BY_NAME, USER_ARCHETYPES, UserArchetype
from synthetic_data.user_config import (
    DEFAULT_USER_COUNT,
    DEFAULT_USER_SEED,
    MAX_USER_COUNT,
    MIXED_PREFERENCE_PROBABILITY,
    PREFERENCE_NAMES,
    PRODUCT_CATALOG_VERSION,
    SYNTHETIC_USER_CREATED_AT,
    USER_GENERATION_VERSION,
    USER_PREFERENCE_NOISE_STDDEV,
)


class UserValidationError(ValueError):
    """Raised before persistence when generated users violate the v1 contract."""


@dataclass(frozen=True)
class SyntheticUserRecord:
    user_id: str
    archetype: str
    secondary_archetype: str | None
    secondary_archetype_weight: float
    preferences: dict[str, float]
    budget_log10: float
    budget_tier: BudgetTier
    price_sensitivity: float
    popularity_preference: float
    exploration_tendency: float
    impulsiveness: float
    activity_level: float
    activity_tier: ActivityTier
    catalog_version: str
    user_generation_version: str
    generation_seed: int
    created_at: datetime


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _allocate_counts(total: int, weights: list[int]) -> list[int]:
    weight_total = sum(weights)
    raw_counts = [total * weight / weight_total for weight in weights]
    allocated = [math.floor(value) for value in raw_counts]
    remaining = total - sum(allocated)
    priorities = sorted(
        range(len(raw_counts)),
        key=lambda index: (raw_counts[index] - allocated[index], weights[index]),
        reverse=True,
    )
    for index in priorities[:remaining]:
        allocated[index] += 1
    return allocated


def _budget_tier(value: float) -> BudgetTier:
    if value < 11.0:
        return BudgetTier.LOW
    if value < 15.0:
        return BudgetTier.MEDIUM
    if value < 20.0:
        return BudgetTier.HIGH
    if value < 24.0:
        return BudgetTier.ULTRA_HIGH
    return BudgetTier.ABSURD


def _activity_tier(value: float) -> ActivityTier:
    if value < 0.35:
        return ActivityTier.CASUAL
    if value < 0.80:
        return ActivityTier.REGULAR
    return ActivityTier.HEAVY


def _mixed_preferences(
    primary: UserArchetype,
    rng: random.Random,
) -> tuple[dict[str, float], str | None, float]:
    secondary = None
    secondary_weight = 0.0
    if rng.random() < MIXED_PREFERENCE_PROBABILITY:
        secondary = rng.choice([candidate for candidate in USER_ARCHETYPES if candidate != primary])
        secondary_weight = rng.uniform(0.20, 0.40)

    generated = {}
    for name in PREFERENCE_NAMES:
        prototype = primary.preference_prototype[name]
        if secondary is not None:
            prototype = (
                (1.0 - secondary_weight) * prototype
                + secondary_weight * secondary.preference_prototype[name]
            )
        generated[name] = round(
            _clip(prototype + rng.gauss(0.0, USER_PREFERENCE_NOISE_STDDEV)),
            4,
        )
    return generated, secondary.name if secondary else None, round(secondary_weight, 4)


def generate_users(
    count: int = DEFAULT_USER_COUNT,
    seed: int = DEFAULT_USER_SEED,
) -> list[SyntheticUserRecord]:
    """Generate deterministic hidden-ground-truth user profiles without DB access."""
    if not 1 <= count <= MAX_USER_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_USER_COUNT}")

    rng = random.Random(seed)
    allocations = _allocate_counts(count, [archetype.weight for archetype in USER_ARCHETYPES])
    assigned_archetypes = [
        archetype
        for archetype, allocation in zip(USER_ARCHETYPES, allocations, strict=True)
        for _ in range(allocation)
    ]
    rng.shuffle(assigned_archetypes)

    records = []
    for index, archetype in enumerate(assigned_archetypes, start=1):
        preference_values, secondary_name, secondary_weight = _mixed_preferences(archetype, rng)
        budget_log10 = round(_clip(rng.gauss(archetype.budget_log10, 3.0), 7.2, 27.2), 4)
        activity_level = round(_clip(rng.gauss(archetype.activity_level, 0.18)), 4)
        record = SyntheticUserRecord(
            user_id=f"synthetic-user-v1-{index:06d}",
            archetype=archetype.name,
            secondary_archetype=secondary_name,
            secondary_archetype_weight=secondary_weight,
            preferences=preference_values,
            budget_log10=budget_log10,
            budget_tier=_budget_tier(budget_log10),
            price_sensitivity=round(_clip(rng.gauss(archetype.price_sensitivity, 0.19)), 4),
            popularity_preference=round(
                _clip(rng.gauss(archetype.popularity_preference, 0.18)), 4
            ),
            exploration_tendency=round(
                _clip(rng.gauss(archetype.exploration_tendency, 0.18)), 4
            ),
            impulsiveness=round(_clip(rng.gauss(archetype.impulsiveness, 0.20)), 4),
            activity_level=activity_level,
            activity_tier=_activity_tier(activity_level),
            catalog_version=PRODUCT_CATALOG_VERSION,
            user_generation_version=USER_GENERATION_VERSION,
            generation_seed=seed,
            created_at=SYNTHETIC_USER_CREATED_AT + timedelta(seconds=index - 1),
        )
        records.append(record)

    validate_users(records, expected_count=count, expected_seed=seed)
    return records


def validate_users(
    records: list[SyntheticUserRecord],
    expected_count: int,
    expected_seed: int | None = None,
) -> None:
    errors = []
    identities = [record.user_id for record in records]
    valid_archetypes = set(ARCHETYPE_BY_NAME)

    if len(records) != expected_count:
        errors.append(f"expected {expected_count} users, found {len(records)}")
    if len(identities) != len(set(identities)):
        errors.append("user identities must be unique")

    for record in records:
        prefix = record.user_id or "<missing user identity>"
        if not record.user_id.strip():
            errors.append("user identity must not be empty")
        if record.archetype not in valid_archetypes:
            errors.append(f"{prefix}: undefined archetype {record.archetype}")
        if record.secondary_archetype is not None:
            if record.secondary_archetype not in valid_archetypes:
                errors.append(f"{prefix}: undefined secondary archetype")
            if record.secondary_archetype == record.archetype:
                errors.append(f"{prefix}: secondary archetype must differ from primary")
            if not 0.20 <= record.secondary_archetype_weight <= 0.40:
                errors.append(f"{prefix}: invalid secondary archetype weight")
        elif record.secondary_archetype_weight != 0.0:
            errors.append(f"{prefix}: secondary weight requires a secondary archetype")
        if set(record.preferences) != set(PREFERENCE_NAMES):
            errors.append(f"{prefix}: preferences must exactly match the common nine-axis schema")
        if any(not 0.0 <= value <= 1.0 for value in record.preferences.values()):
            errors.append(f"{prefix}: preference values must be in [0, 1]")
        if not 5.0 <= record.budget_log10 <= 30.0:
            errors.append(f"{prefix}: budget_log10 must be in [5, 30]")
        if not isinstance(record.budget_tier, BudgetTier):
            errors.append(f"{prefix}: invalid budget tier")
        for field_name in (
            "price_sensitivity",
            "popularity_preference",
            "exploration_tendency",
            "impulsiveness",
            "activity_level",
        ):
            if not 0.0 <= getattr(record, field_name) <= 1.0:
                errors.append(f"{prefix}: {field_name} must be in [0, 1]")
        if not isinstance(record.activity_tier, ActivityTier):
            errors.append(f"{prefix}: invalid activity tier")
        if record.catalog_version != PRODUCT_CATALOG_VERSION:
            errors.append(f"{prefix}: invalid product catalog version")
        if record.user_generation_version != USER_GENERATION_VERSION:
            errors.append(f"{prefix}: invalid user generation version")
        if expected_seed is not None and record.generation_seed != expected_seed:
            errors.append(f"{prefix}: generation seed does not match requested seed")
        if record.created_at.tzinfo is None:
            errors.append(f"{prefix}: created_at must be timezone-aware")

    if errors:
        raise UserValidationError("Synthetic user validation failed:\n- " + "\n- ".join(errors))


def _summary_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": fmean(values),
        "std": pstdev(values),
        "min": ordered[0],
        "25%": _percentile(ordered, 0.25),
        "median": median(ordered),
        "75%": _percentile(ordered, 0.75),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_users(records: list[SyntheticUserRecord]) -> dict[str, object]:
    behavior_names = (
        "price_sensitivity",
        "popularity_preference",
        "exploration_tendency",
        "impulsiveness",
        "activity_level",
    )
    return {
        "user_count": len(records),
        "archetype_distribution": dict(Counter(record.archetype for record in records)),
        "mixed_user_count": sum(record.secondary_archetype is not None for record in records),
        "preference_statistics": {
            name: _summary_stats([record.preferences[name] for record in records])
            for name in PREFERENCE_NAMES
        },
        "budget": _summary_stats([record.budget_log10 for record in records]),
        "budget_tier_distribution": dict(Counter(record.budget_tier.value for record in records)),
        "behavior_statistics": {
            name: _summary_stats([getattr(record, name) for record in records])
            for name in behavior_names
        },
        "activity_tier_distribution": dict(
            Counter(record.activity_tier.value for record in records)
        ),
    }


def format_user_summary(summary: dict[str, object]) -> str:
    lines = [f"Users: {summary['user_count']}", "", "Archetype distribution:"]
    lines.extend(f"  {name}: {count}" for name, count in summary["archetype_distribution"].items())
    lines.extend(("", f"Mixed-preference users: {summary['mixed_user_count']}", "", "Preference mean/std:"))
    for name, values in summary["preference_statistics"].items():
        lines.append(f"  {name}: mean={values['mean']:.4f}, std={values['std']:.4f}")
    budget = summary["budget"]
    lines.extend(
        (
            "",
            "Budget log10:",
            f"  min={budget['min']:.4f}, q25={budget['25%']:.4f}, "
            f"median={budget['median']:.4f}, q75={budget['75%']:.4f}, max={budget['max']:.4f}",
            "",
            "Budget tiers:",
        )
    )
    lines.extend(f"  {name}: {count}" for name, count in summary["budget_tier_distribution"].items())
    lines.extend(("", "Behavior mean/std:"))
    for name, values in summary["behavior_statistics"].items():
        lines.append(f"  {name}: mean={values['mean']:.4f}, std={values['std']:.4f}")
    lines.extend(("", "Activity tiers:"))
    lines.extend(f"  {name}: {count}" for name, count in summary["activity_tier_distribution"].items())
    return "\n".join(lines)
