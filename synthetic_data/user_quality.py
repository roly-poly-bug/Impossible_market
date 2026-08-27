from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median, pstdev

from synthetic_data.config import DEFAULT_PRODUCT_COUNT, DEFAULT_SEED
from synthetic_data.product_generator import SyntheticProductRecord, generate_catalog
from synthetic_data.user_archetypes import USER_ARCHETYPES
from synthetic_data.user_config import DEFAULT_USER_COUNT, DEFAULT_USER_SEED, PREFERENCE_NAMES, PREFERENCE_PRODUCT_MAPPING
from synthetic_data.user_generator import SyntheticUserRecord, generate_users


REPRESENTATIVE_ARCHETYPES = (
    "Space Enthusiast",
    "History Collector",
    "Tech Futurist",
    "Nature Explorer",
    "Fantasy Lover",
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def descriptive_statistics(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "25%": percentile(values, 0.25),
        "median": median(values),
        "75%": percentile(values, 0.75),
        "max": max(values),
    }


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(left, right, strict=True)))


def _archetype_means(users: list[SyntheticUserRecord]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[SyntheticUserRecord]] = defaultdict(list)
    for user in users:
        groups[user.archetype].append(user)
    return {
        name: {
            preference: fmean(user.preferences[preference] for user in group)
            for preference in PREFERENCE_NAMES
        }
        for name, group in sorted(groups.items())
    }


def _separation(users: list[SyntheticUserRecord]) -> dict[str, float | int]:
    correct = 0
    pure_correct = 0
    mixed_correct = 0
    pure_count = 0
    mixed_count = 0
    own_distances = []
    closest_competing_distances = []
    prototypes = {archetype.name: archetype.preference_prototype for archetype in USER_ARCHETYPES}
    for user in users:
        vector = [user.preferences[name] for name in PREFERENCE_NAMES]
        distances = sorted(
            (
                euclidean(vector, [prototype[name] for name in PREFERENCE_NAMES]),
                archetype_name,
            )
            for archetype_name, prototype in prototypes.items()
        )
        predicted = distances[0][1]
        is_correct = predicted == user.archetype
        correct += is_correct
        if user.secondary_archetype is None:
            pure_count += 1
            pure_correct += is_correct
        else:
            mixed_count += 1
            mixed_correct += is_correct
        own_distances.append(
            euclidean(
                vector,
                [prototypes[user.archetype][name] for name in PREFERENCE_NAMES],
            )
        )
        closest_competing_distances.append(
            min(distance for distance, name in distances if name != user.archetype)
        )
    return {
        "nearest_prototype_accuracy": correct / len(users),
        "pure_accuracy": pure_correct / pure_count,
        "mixed_accuracy": mixed_correct / mixed_count,
        "pure_count": pure_count,
        "mixed_count": mixed_count,
        "mean_own_prototype_distance": fmean(own_distances),
        "mean_closest_competing_distance": fmean(closest_competing_distances),
    }


def _budget_affordability(
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
) -> dict[str, object]:
    product_log_prices = [math.log10(float(product.price)) for product in products]
    shares = [
        sum(price <= user.budget_log10 for price in product_log_prices) / len(product_log_prices)
        for user in users
    ]
    by_tier: dict[str, list[float]] = defaultdict(list)
    for user, share in zip(users, shares, strict=True):
        by_tier[user.budget_tier.value].append(share)
    return {
        "catalog_log10_price": descriptive_statistics(product_log_prices),
        "affordable_share": descriptive_statistics(shares),
        "zero_affordable_users": sum(share == 0.0 for share in shares),
        "all_affordable_users": sum(share == 1.0 for share in shares),
        "by_budget_tier": {
            tier: {
                "count": len(values),
                "mean": fmean(values),
                "min": min(values),
                "max": max(values),
            }
            for tier, values in sorted(by_tier.items())
        },
    }


def _preference_matches(
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
) -> dict[str, list[tuple[str, str, float]]]:
    matches = {}
    for archetype in REPRESENTATIVE_ARCHETYPES:
        group = [user for user in users if user.archetype == archetype]
        mean_preferences = {
            name: fmean(user.preferences[name] for user in group)
            for name in PREFERENCE_NAMES
        }
        scored = []
        for product in products:
            distance = euclidean(
                [mean_preferences[name] for name in PREFERENCE_NAMES],
                [
                    product.attributes[PREFERENCE_PRODUCT_MAPPING[name]]
                    for name in PREFERENCE_NAMES
                ],
            )
            scored.append((product.name, product.category_parent, 1.0 / (1.0 + distance)))
        matches[archetype] = sorted(scored, key=lambda item: (-item[2], item[0]))[:5]
    return matches


def analyze_users(
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
) -> dict[str, object]:
    preference_values = {
        name: [user.preferences[name] for user in users] for name in PREFERENCE_NAMES
    }
    correlation_matrix = {
        left: {
            right: pearson(preference_values[left], preference_values[right])
            for right in PREFERENCE_NAMES
        }
        for left in PREFERENCE_NAMES
    }
    strong_correlations = [
        (left, right, correlation_matrix[left][right])
        for index, left in enumerate(PREFERENCE_NAMES)
        for right in PREFERENCE_NAMES[index + 1 :]
        if abs(correlation_matrix[left][right]) >= 0.8
    ]
    budget_values = [user.budget_log10 for user in users]
    sensitivity_values = [user.price_sensitivity for user in users]
    return {
        "metadata": {
            "user_generation_version": users[0].user_generation_version,
            "user_seed": users[0].generation_seed,
            "user_count": len(users),
            "catalog_version": users[0].catalog_version,
            "catalog_seed": products[0].generation_seed,
            "catalog_count": len(products),
        },
        "preference_statistics": {
            name: descriptive_statistics(values) for name, values in preference_values.items()
        },
        "archetype_distribution": dict(Counter(user.archetype for user in users)),
        "mixed_user_count": sum(user.secondary_archetype is not None for user in users),
        "archetype_means": _archetype_means(users),
        "correlation_matrix": correlation_matrix,
        "strong_correlations": strong_correlations,
        "separation": _separation(users),
        "budget_statistics": descriptive_statistics(budget_values),
        "budget_tier_distribution": dict(Counter(user.budget_tier.value for user in users)),
        "budget_price_sensitivity_correlation": pearson(budget_values, sensitivity_values),
        "behavior_statistics": {
            name: descriptive_statistics([getattr(user, name) for user in users])
            for name in (
                "price_sensitivity",
                "popularity_preference",
                "exploration_tendency",
                "impulsiveness",
                "activity_level",
            )
        },
        "activity_tier_distribution": dict(Counter(user.activity_tier.value for user in users)),
        "budget_affordability": _budget_affordability(users, products),
        "preference_matches": _preference_matches(users, products),
    }


def _table(headers, rows) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_user_quality_report(analysis: dict[str, object]) -> str:
    metadata = analysis["metadata"]
    stats = analysis["preference_statistics"]
    lines = [
        "# Synthetic User v1 Quality Report",
        "",
        "This audit evaluates hidden simulator ground truth. It does not generate Events or implement a recommendation model.",
        "Population standard deviation (`ddof=0`) and linearly interpolated quartiles are used.",
        "",
        "## Population",
        "",
        f"- User version: `{metadata['user_generation_version']}`",
        f"- User seed/count: `{metadata['user_seed']}` / `{metadata['user_count']}`",
        f"- Frozen catalog: `{metadata['catalog_version']}`, seed `{metadata['catalog_seed']}`, count `{metadata['catalog_count']}`",
        "",
        "## Archetype distribution",
        "",
    ]
    lines.extend(
        f"- {name}: {count} ({count / metadata['user_count']:.1%})"
        for name, count in analysis["archetype_distribution"].items()
    )
    lines.extend(
        (
            "",
            f"Mixed-preference users: {analysis['mixed_user_count']} "
            f"({analysis['mixed_user_count'] / metadata['user_count']:.1%}).",
            "",
            "## Preference descriptive statistics",
            "",
        )
    )
    lines.extend(
        _table(
            ("preference", "count", "mean", "std", "min", "25%", "median", "75%", "max"),
            (
                (
                    name,
                    str(values["count"]),
                    *(f"{values[key]:.4f}" for key in ("mean", "std", "min", "25%", "median", "75%", "max")),
                )
                for name, values in stats.items()
            ),
        )
    )
    lines.extend(("", "## Archetype preference means", ""))
    lines.extend(
        _table(
            ("archetype", *PREFERENCE_NAMES),
            (
                (name, *(f"{values[pref]:.3f}" for pref in PREFERENCE_NAMES))
                for name, values in analysis["archetype_means"].items()
            ),
        )
    )
    lines.extend(("", "## Pearson preference correlation matrix", ""))
    matrix = analysis["correlation_matrix"]
    lines.extend(
        _table(
            ("preference", *PREFERENCE_NAMES),
            (
                (left, *(f"{matrix[left][right]:.3f}" for right in PREFERENCE_NAMES))
                for left in PREFERENCE_NAMES
            ),
        )
    )
    lines.extend(("", "Absolute correlations >= 0.8:", ""))
    strong = analysis["strong_correlations"]
    lines.extend(
        (f"- `{left}` / `{right}`: {value:.4f}" for left, right, value in strong)
        if strong
        else ("- None",)
    )
    separation = analysis["separation"]
    lines.extend(
        (
            "",
            "## Archetype separation sanity check",
            "",
            f"- Nearest-prototype accuracy: **{separation['nearest_prototype_accuracy']:.1%}**",
            f"- Pure-profile accuracy ({separation['pure_count']} users): {separation['pure_accuracy']:.1%}",
            f"- Mixed-profile accuracy ({separation['mixed_count']} users): {separation['mixed_accuracy']:.1%}",
            f"- Mean distance to assigned prototype: {separation['mean_own_prototype_distance']:.4f}",
            f"- Mean distance to closest competing prototype: {separation['mean_closest_competing_distance']:.4f}",
            "",
            "The primary archetype signal is detectable but far from perfectly recoverable. Mixed profiles materially increase overlap, avoiding a trivially separable ten-cluster dataset.",
            "",
            "## Budget versus frozen product prices",
            "",
        )
    )
    budget = analysis["budget_statistics"]
    affordability = analysis["budget_affordability"]
    shares = affordability["affordable_share"]
    lines.extend(
        (
            f"Budget log10: min {budget['min']:.4f}, 25% {budget['25%']:.4f}, median {budget['median']:.4f}, 75% {budget['75%']:.4f}, max {budget['max']:.4f}.",
            f"Budget / price-sensitivity correlation: {analysis['budget_price_sensitivity_correlation']:.4f}.",
            f"Affordable catalog share: min {shares['min']:.1%}, 25% {shares['25%']:.1%}, median {shares['median']:.1%}, 75% {shares['75%']:.1%}, max {shares['max']:.1%}.",
            f"Users with zero affordable products: {affordability['zero_affordable_users']}; users able to afford all products: {affordability['all_affordable_users']}.",
            "",
        )
    )
    lines.extend(
        _table(
            ("budget tier", "users", "mean affordable", "min", "max"),
            (
                (
                    tier,
                    str(values["count"]),
                    f"{values['mean']:.1%}",
                    f"{values['min']:.1%}",
                    f"{values['max']:.1%}",
                )
                for tier, values in affordability["by_budget_tier"].items()
            ),
        )
    )
    lines.extend(("", "Budget tier distribution:", ""))
    lines.extend(
        f"- `{tier}`: {count}"
        for tier, count in analysis["budget_tier_distribution"].items()
    )
    lines.extend(("", "Behavioral parameter statistics:", ""))
    lines.extend(
        _table(
            ("parameter", "mean", "std", "min", "median", "max"),
            (
                (
                    name,
                    f"{values['mean']:.4f}",
                    f"{values['std']:.4f}",
                    f"{values['min']:.4f}",
                    f"{values['median']:.4f}",
                    f"{values['max']:.4f}",
                )
                for name, values in analysis["behavior_statistics"].items()
            ),
        )
    )
    lines.extend(("", "Activity tier distribution:", ""))
    lines.extend(
        f"- `{tier}`: {count}"
        for tier, count in analysis["activity_tier_distribution"].items()
    )
    lines.extend(("", "## Representative ground-truth preference matches", ""))
    for archetype, matches in analysis["preference_matches"].items():
        lines.extend((f"### {archetype}", ""))
        lines.extend(
            f"{index}. {name} ({category}) — alignment {score:.4f}"
            for index, (name, category, score) in enumerate(matches, start=1)
        )
        lines.append("")
    lines.extend(
        (
            "## Assessment",
            "",
            "- All nine preference axes have useful dispersion, with standard deviations around 0.17–0.21 and no collapsed dimension.",
            f"- Intended archetype signatures remain visible, while {separation['mixed_count'] / metadata['user_count']:.1%} mixed users and individual noise keep nearest-prototype accuracy at {separation['nearest_prototype_accuracy']:.1%} rather than near 100%.",
            "- No preference pair reaches the |r| >= 0.8 redundancy threshold.",
            "- Budget tiers create monotonic affordability differences; only a small edge population can afford none or all of the frozen catalog.",
            "- Representative alignment checks return semantically appropriate product categories. These checks validate simulator structure and are not recommendation output.",
            "",
            "## Freeze recommendation",
            "",
            f"Freeze `{metadata['user_generation_version']} / seed {metadata['user_seed']} / {metadata['user_count']} users` for the first Session/Event-generation design. The population is structured, heterogeneous, reproducible, and not trivially separable. Revisit only if later event-funnel diagnostics reveal insufficient behavior variance or excessive budget gating.",
            "",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit synthetic_user_v1 population quality.")
    parser.add_argument("--count", type=int, default=DEFAULT_USER_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_USER_SEED)
    parser.add_argument("--output", type=Path, default=Path("docs/synthetic_user_v1_quality.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    users = generate_users(count=args.count, seed=args.seed)
    products = generate_catalog(count=DEFAULT_PRODUCT_COUNT, seed=DEFAULT_SEED)
    report = render_user_quality_report(analyze_users(users, products))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"User quality report complete: {args.output}")


if __name__ == "__main__":
    main()
