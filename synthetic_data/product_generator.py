from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, localcontext
from statistics import fmean, median

from backend.app.db.models import ProductStatus, RealityType
from synthetic_data.config import (
    ATTRIBUTE_NAMES,
    ATTRIBUTE_NOISE_STDDEV,
    CATALOG_VERSION,
    DEFAULT_PRODUCT_COUNT,
    MAX_PRODUCT_COUNT,
    STATUS_PROPORTIONS,
    TAG_VOCABULARY,
)
from synthetic_data.vocabularies import (
    CATEGORY_SPECS,
    FEATURED_DESCRIPTIONS,
    NAME_ADJECTIVES,
    CategorySpec,
)


class CatalogValidationError(ValueError):
    """Raised before persistence when a generated catalog violates its contract."""


@dataclass(frozen=True)
class SyntheticProductRecord:
    name: str
    description: str
    category_parent: str
    category_name: str
    category_slug: str
    price: Decimal
    rarity: float
    image_url: str | None
    status: ProductStatus
    reality_type: RealityType
    tags: tuple[str, ...]
    attributes: dict[str, float]
    catalog_version: str
    generation_seed: int


REALITY_OVERRIDES = {
    "Moon": RealityType.REAL,
    "Mars": RealityType.REAL,
    "Pacific Ocean": RealityType.REAL,
    "Tyrannosaurus Rex": RealityType.HISTORICAL,
    "Time Machine": RealityType.SPECULATIVE,
    "Roman Empire": RealityType.HISTORICAL,
    "International Space Station": RealityType.REAL,
    "Dragon": RealityType.FICTIONAL,
    "Dragon Egg": RealityType.FICTIONAL,
    "Luck": RealityType.ABSTRACT,
    "One Extra Hour Per Day": RealityType.ABSTRACT,
}


DESCRIPTION_TEMPLATES = (
    "{name} is a {reality} {category} selected for its {tag1}, {tag2}, and {tag3} qualities.",
    "A singular {category} known for being {tag1}, {tag2}, and unusually {tag3}.",
    "This {reality} {category} combines {tag1} character with {tag2} value and {tag3} appeal.",
)


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _allocate_counts(total: int, weighted_counts: list[int]) -> list[int]:
    weight_total = sum(weighted_counts)
    raw_counts = [total * weight / weight_total for weight in weighted_counts]
    allocated = [math.floor(value) for value in raw_counts]
    remaining = total - sum(allocated)
    priorities = sorted(
        range(len(raw_counts)),
        key=lambda index: (raw_counts[index] - allocated[index], weighted_counts[index]),
        reverse=True,
    )
    for index in priorities[:remaining]:
        allocated[index] += 1
    return allocated


def _allocate_statuses(count: int, rng: random.Random) -> list[ProductStatus]:
    status_names = list(STATUS_PROPORTIONS)
    weights = [int(STATUS_PROPORTIONS[name] * 100) for name in status_names]
    counts = _allocate_counts(count, weights)
    statuses = [
        ProductStatus(status_name)
        for status_name, status_count in zip(status_names, counts, strict=True)
        for _ in range(status_count)
    ]
    rng.shuffle(statuses)
    return statuses


def _weighted_reality(spec: CategorySpec, rng: random.Random) -> RealityType:
    threshold = rng.random()
    cumulative = 0.0
    for reality_name, weight in spec.reality_weights.items():
        cumulative += weight
        if threshold <= cumulative:
            return RealityType(reality_name)
    return RealityType(next(reversed(spec.reality_weights)))


def _attributes_from_prototype(spec: CategorySpec, rng: random.Random) -> dict[str, float]:
    return {
        attribute_name: round(
            _clip(prototype_value + rng.gauss(0.0, ATTRIBUTE_NOISE_STDDEV)),
            4,
        )
        for attribute_name, prototype_value in spec.prototype.items()
    }


def _tags_for_product(
    spec: CategorySpec,
    attributes: dict[str, float],
    rng: random.Random,
) -> tuple[str, ...]:
    tags = list(spec.base_tags)
    candidates = []
    if attributes["danger"] >= 0.78:
        candidates.extend(("dangerous", "destructive"))
    if attributes["luxury"] >= 0.80:
        candidates.extend(("luxury", "valuable"))
    if attributes["novelty"] >= 0.82:
        candidates.extend(("rare", "mysterious"))
    if attributes["historical_value"] >= 0.80:
        candidates.extend(("historic", "cultural"))
    if attributes["technology_level"] >= 0.80:
        candidates.extend(("technology", "artificial", "scientific"))
    if attributes["natural_significance"] >= 0.80:
        candidates.append("natural")
    if attributes["fantasy_level"] >= 0.80:
        candidates.extend(("fantasy", "legendary"))
    if attributes["space_affinity"] >= 0.80:
        candidates.extend(("space", "unexplored"))
    if attributes["power"] >= 0.85:
        candidates.extend(("powerful", "exclusive"))

    rng.shuffle(candidates)
    target_count = rng.randint(len(spec.base_tags), 6)
    for candidate in candidates:
        if candidate not in tags:
            tags.append(candidate)
        if len(tags) == target_count:
            break
    return tuple(tags[:6])


def _price_from_metadata(
    spec: CategorySpec,
    rarity: float,
    attributes: dict[str, float],
    rng: random.Random,
) -> Decimal:
    log10_price = (
        spec.base_log10_price
        + 2.0 * (rarity - 0.5)
        + 1.5 * (attributes["luxury"] - 0.5)
        + 1.7 * (attributes["power"] - 0.5)
        + 1.0 * (attributes["historical_value"] - 0.5)
        + rng.gauss(0.0, 0.42)
    )
    log10_price = min(30.0, max(5.0, log10_price))
    exponent = math.floor(log10_price)
    mantissa = round(10 ** (log10_price - exponent), 3)
    with localcontext() as context:
        context.prec = 50
        return (Decimal(str(mantissa)) * (Decimal(10) ** exponent)).quantize(Decimal("1"))


def _name_candidates(spec: CategorySpec) -> tuple[str, ...]:
    generated = tuple(f"{adjective} {spec.generated_subject}" for adjective in NAME_ADJECTIVES)
    return spec.featured_names + generated


def _description(
    name: str,
    spec: CategorySpec,
    reality_type: RealityType,
    tags: tuple[str, ...],
    rng: random.Random,
) -> str:
    if name in FEATURED_DESCRIPTIONS:
        return FEATURED_DESCRIPTIONS[name]
    template = rng.choice(DESCRIPTION_TEMPLATES)
    return template.format(
        name=name,
        reality=reality_type.value,
        category=spec.name.lower(),
        tag1=tags[0],
        tag2=tags[1],
        tag3=tags[2],
    )


def generate_catalog(
    count: int = DEFAULT_PRODUCT_COUNT,
    seed: int = 42,
) -> list[SyntheticProductRecord]:
    """Generate validated structured records without touching a database."""
    if not 1 <= count <= MAX_PRODUCT_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_PRODUCT_COUNT}")

    rng = random.Random(seed)
    allocations = _allocate_counts(count, [spec.target_count for spec in CATEGORY_SPECS])
    statuses = iter(_allocate_statuses(count, rng))
    records = []

    for spec, allocated_count in zip(CATEGORY_SPECS, allocations, strict=True):
        names = _name_candidates(spec)[:allocated_count]
        for name in names:
            attributes = _attributes_from_prototype(spec, rng)
            rarity = round(
                _clip(
                    0.18
                    + 0.48 * attributes["novelty"]
                    + 0.20 * attributes["luxury"]
                    + rng.gauss(0.0, 0.075)
                ),
                4,
            )
            tags = _tags_for_product(spec, attributes, rng)
            reality_type = REALITY_OVERRIDES.get(name, _weighted_reality(spec, rng))
            record = SyntheticProductRecord(
                name=name,
                description=_description(name, spec, reality_type, tags, rng),
                category_parent=spec.parent,
                category_name=spec.name,
                category_slug=spec.slug,
                price=_price_from_metadata(spec, rarity, attributes, rng),
                rarity=rarity,
                image_url=None,
                status=next(statuses),
                reality_type=reality_type,
                tags=tags,
                attributes=attributes,
                catalog_version=CATALOG_VERSION,
                generation_seed=seed,
            )
            records.append(record)

    validate_catalog(records, expected_count=count)
    return records


def validate_catalog(
    records: list[SyntheticProductRecord],
    expected_count: int,
) -> None:
    errors = []
    valid_categories = {spec.slug for spec in CATEGORY_SPECS}
    names = [record.name for record in records]

    if len(records) != expected_count:
        errors.append(f"expected {expected_count} products, found {len(records)}")
    if len(names) != len(set(names)):
        errors.append("product names must be unique")

    for record in records:
        prefix = record.name or "<unnamed product>"
        if not record.name.strip():
            errors.append("product name must not be empty")
        if not record.description.strip():
            errors.append(f"{prefix}: description must not be empty")
        if record.category_slug not in valid_categories:
            errors.append(f"{prefix}: unknown category {record.category_slug}")
        if not 3 <= len(record.tags) <= 6:
            errors.append(f"{prefix}: expected 3-6 tags")
        unknown_tags = set(record.tags) - TAG_VOCABULARY
        if unknown_tags:
            errors.append(f"{prefix}: unknown tags {sorted(unknown_tags)}")
        if set(record.attributes) != set(ATTRIBUTE_NAMES):
            errors.append(f"{prefix}: attributes must exactly match the common nine-axis schema")
        if any(not 0.0 <= value <= 1.0 for value in record.attributes.values()):
            errors.append(f"{prefix}: attribute values must be in [0, 1]")
        if not 0.0 <= record.rarity <= 1.0:
            errors.append(f"{prefix}: rarity must be in [0, 1]")
        if not isinstance(record.reality_type, RealityType):
            errors.append(f"{prefix}: invalid reality type")
        if not isinstance(record.status, ProductStatus):
            errors.append(f"{prefix}: invalid status")
        if record.price <= 0:
            errors.append(f"{prefix}: price must be positive")
        if record.catalog_version != CATALOG_VERSION:
            errors.append(f"{prefix}: invalid catalog version")

    if errors:
        raise CatalogValidationError("Catalog validation failed:\n- " + "\n- ".join(errors))


def summarize_catalog(records: list[SyntheticProductRecord]) -> dict[str, object]:
    prices = sorted(record.price for record in records)
    return {
        "product_count": len(records),
        "category_distribution": dict(Counter(record.category_parent for record in records)),
        "reality_distribution": dict(Counter(record.reality_type.value for record in records)),
        "status_distribution": dict(Counter(record.status.value for record in records)),
        "price": {
            "min": prices[0],
            "median": median(prices),
            "max": prices[-1],
        },
        "attribute_means": {
            name: fmean(record.attributes[name] for record in records)
            for name in ATTRIBUTE_NAMES
        },
    }


def format_summary(summary: dict[str, object]) -> str:
    price = summary["price"]
    lines = [f"Products: {summary['product_count']}", "", "Category distribution:"]
    lines.extend(f"  {name}: {count}" for name, count in summary["category_distribution"].items())
    lines.extend(("", "Reality type distribution:"))
    lines.extend(f"  {name}: {count}" for name, count in summary["reality_distribution"].items())
    lines.extend(("", "Status distribution:"))
    lines.extend(f"  {name}: {count}" for name, count in summary["status_distribution"].items())
    lines.extend(
        (
            "",
            "Price:",
            f"  min: {price['min']}",
            f"  median: {price['median']}",
            f"  max: {price['max']}",
            "",
            "Attribute means:",
        )
    )
    lines.extend(f"  {name}: {value:.4f}" for name, value in summary["attribute_means"].items())
    return "\n".join(lines)
