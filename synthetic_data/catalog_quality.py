from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Sequence

from synthetic_data.config import ATTRIBUTE_NAMES, DEFAULT_PRODUCT_COUNT, DEFAULT_SEED, TAG_VOCABULARY
from synthetic_data.product_generator import SyntheticProductRecord, generate_catalog


REPRESENTATIVE_PRODUCTS = (
    "Moon",
    "Time Machine",
    "Tyrannosaurus Rex",
    "Roman Empire",
    "Luck",
)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower]) * (1 - fraction) + float(ordered[upper]) * fraction


def descriptive_statistics(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "25%": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "75%": percentile(values, 0.75),
        "max": max(values),
    }


def decimal_percentile(values: Sequence[int], probability: float) -> Decimal:
    ordered = sorted(Decimal(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = Decimal(len(ordered) - 1) * Decimal(str(probability))
    lower = int(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - Decimal(lower)
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Pearson inputs must have equal non-zero length")
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(x * y for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return numerator / denominator if denominator else 0.0


def _attribute_means(records: Iterable[SyntheticProductRecord]) -> dict[str, float]:
    group = list(records)
    return {
        attribute: fmean(record.attributes[attribute] for record in group)
        for attribute in ATTRIBUTE_NAMES
    }


def _group_attribute_means(
    records: Sequence[SyntheticProductRecord], field: str
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[SyntheticProductRecord]] = defaultdict(list)
    for record in records:
        groups[getattr(record, field)].append(record)
    return {name: _attribute_means(group) for name, group in sorted(groups.items())}


def analyze_catalog(records: Sequence[SyntheticProductRecord]) -> dict[str, object]:
    attribute_values = {
        attribute: [record.attributes[attribute] for record in records]
        for attribute in ATTRIBUTE_NAMES
    }
    correlation_matrix = {
        left: {
            right: pearson(attribute_values[left], attribute_values[right])
            for right in ATTRIBUTE_NAMES
        }
        for left in ATTRIBUTE_NAMES
    }
    strong_correlations = [
        (left, right, correlation_matrix[left][right])
        for left_index, left in enumerate(ATTRIBUTE_NAMES)
        for right in ATTRIBUTE_NAMES[left_index + 1 :]
        if abs(correlation_matrix[left][right]) >= 0.8
    ]

    rarity_values = [record.rarity for record in records]
    novelty_values = attribute_values["novelty"]
    log_prices = [math.log10(float(record.price)) for record in records]

    prices_by_parent: dict[str, list[int]] = defaultdict(list)
    for record in records:
        prices_by_parent[record.category_parent].append(int(record.price))

    tag_counts = Counter(tag for record in records for tag in record.tags)
    unused_tags = sorted(TAG_VOCABULARY - tag_counts.keys())

    vectors = {
        record.name: [record.attributes[attribute] for attribute in ATTRIBUTE_NAMES]
        for record in records
    }
    nearest_neighbors: dict[str, list[tuple[str, float]]] = {}
    for name in REPRESENTATIVE_PRODUCTS:
        neighbors = sorted(
            (
                (other_name, cosine_similarity(vectors[name], other_vector))
                for other_name, other_vector in vectors.items()
                if other_name != name
            ),
            key=lambda item: (-item[1], item[0]),
        )
        nearest_neighbors[name] = neighbors[:5]

    return {
        "metadata": {
            "catalog_version": records[0].catalog_version,
            "generation_seed": records[0].generation_seed,
            "count": len(records),
        },
        "attribute_statistics": {
            attribute: descriptive_statistics(values)
            for attribute, values in attribute_values.items()
        },
        "attribute_boundary_share": {
            attribute: {
                "at_or_below_0.05": sum(value <= 0.05 for value in values) / len(values),
                "at_or_above_0.95": sum(value >= 0.95 for value in values) / len(values),
            }
            for attribute, values in attribute_values.items()
        },
        "parent_category_means": _group_attribute_means(records, "category_parent"),
        "child_category_means": _group_attribute_means(records, "category_name"),
        "attribute_correlation_matrix": correlation_matrix,
        "strong_attribute_correlations": strong_correlations,
        "rarity_statistics": descriptive_statistics(rarity_values),
        "rarity_novelty_correlation": pearson(rarity_values, novelty_values),
        "price": {
            "statistics": {
                "min": min(int(record.price) for record in records),
                "25%": decimal_percentile([int(record.price) for record in records], 0.25),
                "median": decimal_percentile([int(record.price) for record in records], 0.5),
                "75%": decimal_percentile([int(record.price) for record in records], 0.75),
                "max": max(int(record.price) for record in records),
            },
            "log10_statistics": descriptive_statistics(log_prices),
            "digit_count_distribution": dict(
                sorted(Counter(len(str(int(record.price))) for record in records).items())
            ),
            "parent_category_median": {
                category: decimal_percentile(values, 0.5)
                for category, values in sorted(prices_by_parent.items())
            },
            "factor_correlations": {
                factor: pearson(
                    log_prices,
                    rarity_values if factor == "rarity" else attribute_values[factor],
                )
                for factor in ("rarity", "luxury", "power", "historical_value")
            },
        },
        "reality_distribution": dict(sorted(Counter(record.reality_type.value for record in records).items())),
        "status_distribution": dict(sorted(Counter(record.status.value for record in records).items())),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))),
        "unused_tags": unused_tags,
        "nearest_neighbors": nearest_neighbors,
    }


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _format_number(value: Decimal | float | int) -> str:
    if isinstance(value, Decimal):
        return f"{value:,.0f}" if value == value.to_integral() else f"{value:,}"
    return f"{value:,}" if isinstance(value, int) else f"{value:.4f}"


def render_markdown_report(analysis: dict[str, object]) -> str:
    metadata = analysis["metadata"]
    statistics = analysis["attribute_statistics"]
    boundaries = analysis["attribute_boundary_share"]
    correlation_matrix = analysis["attribute_correlation_matrix"]
    price = analysis["price"]
    lines = [
        "# Synthetic Product v1 Catalog Quality Report",
        "",
        "This report audits the frozen candidate catalog without changing category prototypes, noise, rarity, or price generation rules.",
        "Population standard deviation (`ddof=0`) and linearly interpolated quartiles are used.",
        "",
        "## Catalog",
        "",
        f"- Version: `{metadata['catalog_version']}`",
        f"- Seed: `{metadata['generation_seed']}`",
        f"- Product count: `{metadata['count']}`",
        "",
        "## Attribute descriptive statistics",
        "",
    ]
    lines.extend(
        _markdown_table(
            ("attribute", "count", "mean", "std", "min", "25%", "median", "75%", "max", "<=0.05", ">=0.95"),
            (
                (
                    attribute,
                    str(values["count"]),
                    *(f"{values[key]:.4f}" for key in ("mean", "std", "min", "25%", "median", "75%", "max")),
                    f"{boundaries[attribute]['at_or_below_0.05']:.1%}",
                    f"{boundaries[attribute]['at_or_above_0.95']:.1%}",
                )
                for attribute, values in statistics.items()
            ),
        )
    )
    lines.extend(("", "## Parent-category attribute means", ""))
    lines.extend(
        _markdown_table(
            ("category", *ATTRIBUTE_NAMES),
            (
                (category, *(f"{means[attribute]:.3f}" for attribute in ATTRIBUTE_NAMES))
                for category, means in analysis["parent_category_means"].items()
            ),
        )
    )
    lines.extend(("", "## Child-category attribute means", ""))
    lines.extend(
        _markdown_table(
            ("category", *ATTRIBUTE_NAMES),
            (
                (category, *(f"{means[attribute]:.3f}" for attribute in ATTRIBUTE_NAMES))
                for category, means in analysis["child_category_means"].items()
            ),
        )
    )
    lines.extend(("", "## Pearson attribute correlation matrix", ""))
    lines.extend(
        _markdown_table(
            ("attribute", *ATTRIBUTE_NAMES),
            (
                (left, *(f"{correlation_matrix[left][right]:.3f}" for right in ATTRIBUTE_NAMES))
                for left in ATTRIBUTE_NAMES
            ),
        )
    )
    lines.extend(("", "Absolute correlations >= 0.8:", ""))
    strong = analysis["strong_attribute_correlations"]
    if strong:
        lines.extend(f"- `{left}` / `{right}`: {value:.4f}" for left, right, value in strong)
    else:
        lines.append("- None")

    rarity = analysis["rarity_statistics"]
    lines.extend(
        (
            "",
            "## Rarity and novelty",
            "",
            f"Rarity: mean {rarity['mean']:.4f}, std {rarity['std']:.4f}, min {rarity['min']:.4f}, "
            f"25% {rarity['25%']:.4f}, median {rarity['median']:.4f}, 75% {rarity['75%']:.4f}, max {rarity['max']:.4f}.",
            f"Pearson correlation between rarity and novelty: **{analysis['rarity_novelty_correlation']:.4f}**.",
            "",
            "## Price",
            "",
        )
    )
    price_stats = price["statistics"]
    log_stats = price["log10_statistics"]
    lines.extend(
        _markdown_table(
            ("metric", "min", "25%", "median", "75%", "max"),
            (
                ("price", *(_format_number(price_stats[key]) for key in ("min", "25%", "median", "75%", "max"))),
                ("log10(price)", *(f"{log_stats[key]:.4f}" for key in ("min", "25%", "median", "75%", "max"))),
            ),
        )
    )
    lines.extend(("", "Parent-category median price:", ""))
    lines.extend(f"- {category}: {_format_number(value)}" for category, value in price["parent_category_median"].items())
    lines.extend(("", "Price digit-count distribution:", ""))
    lines.extend(f"- {digits} digits: {count}" for digits, count in price["digit_count_distribution"].items())
    lines.extend(("", "Correlation with `log10(price)`:", ""))
    lines.extend(f"- `{factor}`: {value:.4f}" for factor, value in price["factor_correlations"].items())

    lines.extend(("", "## Reality type and status", "", "Reality type:", ""))
    lines.extend(f"- `{name}`: {count}" for name, count in analysis["reality_distribution"].items())
    lines.extend(("", "Status:", ""))
    lines.extend(f"- `{name}`: {count}" for name, count in analysis["status_distribution"].items())

    lines.extend(("", "## Tag usage", ""))
    lines.extend(f"- `{tag}`: {count}" for tag, count in analysis["tag_counts"].items())
    unused = analysis["unused_tags"]
    lines.extend(("", f"Unused vocabulary tags: {', '.join(unused) if unused else 'none'}."))

    lines.extend(("", "## Representative cosine nearest neighbors", ""))
    for product, neighbors in analysis["nearest_neighbors"].items():
        lines.append(f"### {product}")
        lines.append("")
        lines.extend(f"{index}. {name} — {similarity:.4f}" for index, (name, similarity) in enumerate(neighbors, 1))
        lines.append("")

    lines.extend(
        (
            "## Interpretation and detected concerns",
            "",
            "- Category prototypes remain clearly visible: Space leads `space_affinity` (0.972), Technology leads `technology_level` (0.921), History leads `historical_value` (0.961), Fantasy leads `fantasy_level` (0.963), and Geography/Creatures are high on `natural_significance` (0.952/0.796). Child-category means retain additional local structure.",
            "- The feature space is not collapsed, but luxury, novelty, and power have high centers and compressed upper tails. This reduces headroom for synthetic-user preference discrimination among premium items.",
            "- No attribute pair reaches the |r| >= 0.8 redundancy threshold. Rarity and novelty are related but not interchangeable.",
            "- Raw positive-valued vectors produce very high cosine scores, so neighbor ranking is mostly locally sensible but weakly separated. Future model experiments should compare centered/scaled features; this report does not implement a recommender.",
            "- Price spans many orders of magnitude by design. Log-price correlations and category medians show that category base price remains a major driver alongside generated factors; extreme values fit the marketplace premise and should not be removed solely as outliers.",
            "- 23 of 25 configured tags are used; `habitable` and `portable` are unused. The most common tag (`valuable`) appears on 48.5% of products, so no tag is close to universal. The used vocabulary has meaningful content-based variation.",
            "",
            "## Freeze recommendation",
            "",
            "Freeze this exact seed-42 snapshot as the tracked **v1 catalog** and use it for the first Synthetic User/Event experiments. Category separation, non-redundant attributes, reproducibility, and tag variation are sufficient for that milestone. Treat the high luxury/novelty/power centers, compressed raw cosine scores, and two unused tags as documented limitations. A v1.1 can broaden the catalog later if initial simulator diagnostics show weak preference separation. The current generator has not been modified by this audit.",
            "",
            "Suggested v1.1 proposal (not implemented):",
            "",
            "| Item | Proposal |",
            "| --- | --- |",
            "| Current issue | Luxury, novelty, and power are high across much of the catalog, raw cosine distances are compressed, and two vocabulary tags are unused. |",
            "| Why it matters | Synthetic users with different premium/power preferences may receive less differentiated relevance signals. |",
            "| Suggested v1.1 change | Add a controlled ordinary/lower-intensity slice or minimally lower only the affected category prototype axes; keep nine-axis schema and category-specific peaks. |",
            "| Expected effect | Wider usable range and neighbor separation without removing the intentionally extravagant long-tail catalog. |",
            "",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit synthetic_product_v1 feature-space quality.")
    parser.add_argument("--count", type=int, default=DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("docs/synthetic_product_v1_quality.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analysis = analyze_catalog(generate_catalog(count=args.count, seed=args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown_report(analysis), encoding="utf-8")
    print(f"Quality report complete: {args.output}")


if __name__ == "__main__":
    main()
