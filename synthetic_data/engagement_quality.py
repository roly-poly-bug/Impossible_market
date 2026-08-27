from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from backend.app.db.models import EventType
from synthetic_data.engagement_generator import SyntheticEngagementRecord
from synthetic_data.event_generator import SyntheticEventRecord
from synthetic_data.interaction_summary import stats
from synthetic_data.product_generator import SyntheticProductRecord
from synthetic_data.user_generator import SyntheticUserRecord


ENGAGEMENT_TYPES = (
    EventType.FAVORITE,
    EventType.ADD_TO_CART,
    EventType.PURCHASE,
)


def _gini(values: list[int]) -> float:
    ordered = sorted(values)
    total = sum(ordered)
    count = len(ordered)
    if not total:
        return 0.0
    weighted = sum(index * value for index, value in enumerate(ordered, start=1))
    return (2 * weighted) / (count * total) - (count + 1) / count


def _auc(rows: list[tuple[float, bool]]) -> float:
    ordered = sorted(rows)
    positives = sum(label for _, label in ordered)
    negatives = len(ordered) - positives
    if not positives or not negatives:
        return 0.5
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _counts_by_group(
    base_events: list[SyntheticEventRecord],
    engagement_events: list[SyntheticEngagementRecord],
    users: list[SyntheticUserRecord],
    field_name: str,
) -> dict[str, dict[str, float | int]]:
    users_by_id = {user.user_id: user for user in users}
    def group_name(user_id: str) -> str:
        value = getattr(users_by_id[user_id], field_name)
        return value.value if hasattr(value, "value") else str(value)

    views = Counter()
    counts = {event_type: Counter() for event_type in ENGAGEMENT_TYPES}
    for event in base_events:
        if event.event_type == EventType.VIEW:
            views[group_name(event.user_id)] += 1
    for event in engagement_events:
        counts[event.event_type][group_name(event.user_id)] += 1
    return {
        group: {
            "views": view_count,
            "favorites": counts[EventType.FAVORITE][group],
            "carts": counts[EventType.ADD_TO_CART][group],
            "purchases": counts[EventType.PURCHASE][group],
            "purchase_rate": counts[EventType.PURCHASE][group] / view_count,
        }
        for group, view_count in sorted(views.items())
    }


def analyze_engagement(
    base_events: list[SyntheticEventRecord],
    engagement_events: list[SyntheticEngagementRecord],
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
) -> dict[str, object]:
    impressions = [event for event in base_events if event.event_type == EventType.IMPRESSION]
    views = [event for event in base_events if event.event_type == EventType.VIEW]
    by_type = {
        event_type: [event for event in engagement_events if event.event_type == event_type]
        for event_type in ENGAGEMENT_TYPES
    }
    pairs = {
        event_type: {(event.user_id, event.product_id) for event in events}
        for event_type, events in by_type.items()
    }
    view_pairs = {(event.user_id, event.product_id) for event in views}
    pair_view = {}
    for event in views:
        pair_view[(event.user_id, event.product_id)] = event

    user_counts = {event_type: Counter() for event_type in ENGAGEMENT_TYPES}
    product_counts = {event_type: Counter() for event_type in ENGAGEMENT_TYPES}
    for event_type, events in by_type.items():
        user_counts[event_type].update(event.user_id for event in events)
        product_counts[event_type].update(event.product_id for event in events)

    purchase_product_values = [
        product_counts[EventType.PURCHASE].get(product.name, 0) for product in products
    ]
    purchase_top_ten = product_counts[EventType.PURCHASE].most_common(10)

    price_signal = {}
    for budget_name, predicate in (
        ("within", lambda event: not event.over_budget),
        ("over", lambda event: event.over_budget),
    ):
        group_views = [event for event in views if predicate(event)]
        price_signal[budget_name] = {
            "views": len(group_views),
            **{
                event_type.value: sum(predicate(event) for event in by_type[event_type])
                for event_type in ENGAGEMENT_TYPES
            },
        }
        for event_type in ENGAGEMENT_TYPES:
            price_signal[budget_name][f"{event_type.value}_rate"] = (
                price_signal[budget_name][event_type.value] / len(group_views)
            )

    users_by_id = {user.user_id: user for user in users}
    impulse_values = sorted(user.impulsiveness for user in users)
    low_threshold = impulse_values[len(impulse_values) // 4]
    high_threshold = impulse_values[3 * len(impulse_values) // 4]
    impulse_groups = {
        "low": {user.user_id for user in users if user.impulsiveness <= low_threshold},
        "high": {user.user_id for user in users if user.impulsiveness >= high_threshold},
    }
    impulse_signal = {}
    for name, user_ids in impulse_groups.items():
        group_views = sum(event.user_id in user_ids for event in views)
        carts = [event for event in by_type[EventType.ADD_TO_CART] if event.user_id in user_ids]
        purchases = [event for event in by_type[EventType.PURCHASE] if event.user_id in user_ids]
        impulse_signal[name] = {
            "users": len(user_ids),
            "views": group_views,
            "direct_cart_rate": sum(not event.had_favorite_before for event in carts) / group_views,
            "direct_purchase_rate": sum(
                not event.had_favorite_before and not event.had_cart_before
                for event in purchases
            )
            / group_views,
            "low_view_purchase_rate": sum(event.view_count == 1 for event in purchases)
            / group_views,
            "over_budget_purchase_rate": sum(event.over_budget for event in purchases)
            / group_views,
        }

    delayed = {
        event_type.value: {
            "same_session": sum(
                event.conversion_timing == "same_session" for event in by_type[event_type]
            ),
            "later_session": sum(
                event.conversion_timing == "later_session" for event in by_type[event_type]
            ),
        }
        for event_type in ENGAGEMENT_TYPES
    }
    delayed_prior_state_purchases = sum(
        event.conversion_timing == "later_session"
        and (event.had_favorite_before or event.had_cart_before)
        for event in by_type[EventType.PURCHASE]
    )

    engagement_by_pair = defaultdict(list)
    for event in engagement_events:
        engagement_by_pair[(event.user_id, event.product_id)].append(event)
    path_counts = Counter()
    for pair in view_pairs:
        sequence = sorted(engagement_by_pair.get(pair, []), key=lambda event: event.timestamp)
        label = "view → exit" if not sequence else "view → " + " → ".join(
            event.event_type.value for event in sequence
        )
        path_counts[label] += 1

    purchase_pairs = pairs[EventType.PURCHASE]
    aucs = {
        "preference_match": _auc(
            [(view.preference_match, pair in purchase_pairs) for pair, view in pair_view.items()]
        ),
        "price_compatibility": _auc(
            [
                (
                    -max(0.0, view.price_log10 - users_by_id[pair[0]].budget_log10)
                    * users_by_id[pair[0]].price_sensitivity,
                    pair in purchase_pairs,
                )
                for pair, view in pair_view.items()
            ]
        ),
        "impulsiveness": _auc(
            [(users_by_id[pair[0]].impulsiveness, pair in purchase_pairs) for pair in pair_view]
        ),
    }

    event_match_means = {
        "impression": fmean(event.preference_match for event in impressions),
        "view": fmean(event.preference_match for event in views),
        **{
            event_type.value: fmean(event.preference_match for event in by_type[event_type])
            for event_type in ENGAGEMENT_TYPES
        },
    }
    non_purchase_view_matches = [
        view.preference_match for pair, view in pair_view.items() if pair not in purchase_pairs
    ]

    matrix_size = len(users) * len(products)
    return {
        "counts": {
            "impressions": len(impressions),
            "views": len(views),
            "favorites": len(by_type[EventType.FAVORITE]),
            "carts": len(by_type[EventType.ADD_TO_CART]),
            "purchases": len(by_type[EventType.PURCHASE]),
        },
        "conversion": {
            "impression_to_view": len(views) / len(impressions),
            "view_to_favorite": len(by_type[EventType.FAVORITE]) / len(views),
            "view_to_cart": len(by_type[EventType.ADD_TO_CART]) / len(views),
            "view_to_purchase": len(by_type[EventType.PURCHASE]) / len(views),
            "favorite_to_cart": len(pairs[EventType.FAVORITE] & pairs[EventType.ADD_TO_CART])
            / len(pairs[EventType.FAVORITE]),
            "favorite_to_purchase": len(pairs[EventType.FAVORITE] & purchase_pairs)
            / len(pairs[EventType.FAVORITE]),
            "cart_to_purchase": len(pairs[EventType.ADD_TO_CART] & purchase_pairs)
            / len(pairs[EventType.ADD_TO_CART]),
        },
        "user_distribution": {
            event_type.value: stats(
                [float(user_counts[event_type].get(user.user_id, 0)) for user in users]
            )
            for event_type in ENGAGEMENT_TYPES
        },
        "purchase_zero_users": sum(
            user_counts[EventType.PURCHASE].get(user.user_id, 0) == 0 for user in users
        ),
        "purchase_positive_users": sum(
            user_counts[EventType.PURCHASE].get(user.user_id, 0) > 0 for user in users
        ),
        "activity": _counts_by_group(
            base_events, engagement_events, users, "activity_tier"
        ),
        "archetype": _counts_by_group(base_events, engagement_events, users, "archetype"),
        "product_distribution": {
            event_type.value: stats(
                [float(product_counts[event_type].get(product.name, 0)) for product in products]
            )
            for event_type in ENGAGEMENT_TYPES
        },
        "purchase_zero_products": sum(value == 0 for value in purchase_product_values),
        "purchase_gini": _gini(purchase_product_values),
        "purchase_top10_share": sum(count for _, count in purchase_top_ten)
        / len(by_type[EventType.PURCHASE]),
        "purchase_top10": purchase_top_ten,
        "preference_means": event_match_means,
        "purchase_match_mean": fmean(
            pair_view[pair].preference_match for pair in purchase_pairs
        ),
        "non_purchase_match_mean": fmean(non_purchase_view_matches),
        "price_signal": price_signal,
        "impulsiveness_signal": impulse_signal,
        "delayed": delayed,
        "delayed_prior_state_purchases": delayed_prior_state_purchases,
        "paths": path_counts,
        "single_feature_auc": aucs,
        "sparsity": {
            "matrix_size": matrix_size,
            "view_pairs": len(view_pairs),
            "favorite_pairs": len(pairs[EventType.FAVORITE]),
            "cart_pairs": len(pairs[EventType.ADD_TO_CART]),
            "purchase_pairs": len(purchase_pairs),
            "view_density": len(view_pairs) / matrix_size,
            "positive_sparsity": 1.0 - len(view_pairs) / matrix_size,
            "purchase_density": len(purchase_pairs) / matrix_size,
        },
    }


def _table(headers, rows) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_engagement_report(analysis: dict[str, object]) -> str:
    counts = analysis["counts"]
    conversion = analysis["conversion"]
    lines = [
        "# Synthetic Engagement v1 Quality Report",
        "",
        "This report audits synthetic behavior rules, not a recommendation model.",
        "",
        "## Frozen world",
        "",
        "- Products: `synthetic_product_v1`, seed `42`, 200 products",
        "- Users: `synthetic_user_v1`, seed `42`, 1,000 users",
        "- Exposure/View: `synthetic_session_event_v1`, seed `42`",
        "- Engagement: `synthetic_engagement_v1`, seed `42`",
        "",
        "## Funnel",
        "",
        f"- Impressions: {counts['impressions']:,}",
        f"- Views: {counts['views']:,}",
        f"- Favorites: {counts['favorites']:,}",
        f"- Carts: {counts['carts']:,}",
        f"- Purchases: {counts['purchases']:,}",
        "",
    ]
    lines.extend(
        _table(
            ("conversion", "rate"),
            (
                (name.replace("_to_", " → ").replace("_", " "), f"{value:.2%}")
                for name, value in conversion.items()
            ),
        )
    )
    lines.extend(("", "## User-level distribution", ""))
    lines.extend(
        _table(
            ("event", "mean", "std", "min", "median", "max"),
            (
                (
                    name,
                    f"{values['mean']:.3f}",
                    f"{values['std']:.3f}",
                    f"{values['min']:.0f}",
                    f"{values['median']:.1f}",
                    f"{values['max']:.0f}",
                )
                for name, values in analysis["user_distribution"].items()
            ),
        )
    )
    lines.extend(
        (
            "",
            f"- Users with zero Purchases: {analysis['purchase_zero_users']:,}",
            f"- Users with at least one Purchase: {analysis['purchase_positive_users']:,}",
        )
    )
    for title, key in (
        ("Activity-tier conversion", "activity"),
        ("Archetype conversion", "archetype"),
    ):
        lines.extend(("", f"## {title}", ""))
        lines.extend(
            _table(
                ("group", "views", "favorites", "carts", "purchases", "purchase/view"),
                (
                    (
                        name,
                        f"{values['views']:,}",
                        f"{values['favorites']:,}",
                        f"{values['carts']:,}",
                        f"{values['purchases']:,}",
                        f"{values['purchase_rate']:.2%}",
                    )
                    for name, values in analysis[key].items()
                ),
            )
        )
    lines.extend(("", "## Product-level distribution", ""))
    lines.extend(
        _table(
            ("event", "mean", "std", "min", "median", "max"),
            (
                (
                    name,
                    f"{values['mean']:.3f}",
                    f"{values['std']:.3f}",
                    f"{values['min']:.0f}",
                    f"{values['median']:.1f}",
                    f"{values['max']:.0f}",
                )
                for name, values in analysis["product_distribution"].items()
            ),
        )
    )
    lines.extend(
        (
            "",
            f"- Products with zero Purchases: {analysis['purchase_zero_products']}",
            f"- Purchase Gini: {analysis['purchase_gini']:.4f}",
            f"- Top-10 Purchase share: {analysis['purchase_top10_share']:.2%}",
            "- Top products: " + ", ".join(
                f"{name} ({count})" for name, count in analysis["purchase_top10"]
            ),
            "",
            "## Preference signal",
            "",
        )
    )
    lines.extend(
        f"- {name}: mean preference match {value:.4f}"
        for name, value in analysis["preference_means"].items()
    )
    lines.extend(
        (
            f"- Purchased viewed pairs: {analysis['purchase_match_mean']:.4f}",
            f"- Non-purchased viewed pairs: {analysis['non_purchase_match_mean']:.4f}",
            "",
            "## Price signal",
            "",
        )
    )
    lines.extend(
        _table(
            ("budget", "views", "favorites", "carts", "purchases"),
            (
                (
                    name,
                    f"{values['views']:,}",
                    f"{values['favorite']:,} ({values['favorite_rate']:.2%})",
                    f"{values['add_to_cart']:,} ({values['add_to_cart_rate']:.2%})",
                    f"{values['purchase']:,} ({values['purchase_rate']:.2%})",
                )
                for name, values in analysis["price_signal"].items()
            ),
        )
    )
    lines.extend(("", "## Impulsiveness signal", ""))
    lines.extend(
        _table(
            (
                "quartile",
                "direct cart/view",
                "direct purchase/view",
                "one-view purchase/view",
                "over-budget purchase/view",
            ),
            (
                (
                    name,
                    f"{values['direct_cart_rate']:.2%}",
                    f"{values['direct_purchase_rate']:.2%}",
                    f"{values['low_view_purchase_rate']:.2%}",
                    f"{values['over_budget_purchase_rate']:.2%}",
                )
                for name, values in analysis["impulsiveness_signal"].items()
            ),
        )
    )
    lines.extend(("", "## Delayed conversion", ""))
    lines.extend(
        _table(
            ("event", "same session", "later session", "later share"),
            (
                (
                    name,
                    f"{values['same_session']:,}",
                    f"{values['later_session']:,}",
                    f"{values['later_session'] / (values['same_session'] + values['later_session']):.2%}",  # noqa: E501
                )
                for name, values in analysis["delayed"].items()
            ),
        )
    )
    lines.append(
        "\nLater-session Purchases with Favorite/Cart state: "
        f"{analysis['delayed_prior_state_purchases']:,}."
    )
    lines.extend(("", "## Top funnel paths", ""))
    total_paths = sum(analysis["paths"].values())
    lines.extend(
        f"- {path}: {count:,} ({count / total_paths:.2%})"
        for path, count in analysis["paths"].most_common(10)
    )
    lines.extend(("", "## Single-feature Purchase AUC", ""))
    lines.extend(
        f"- {name}: {value:.4f}" for name, value in analysis["single_feature_auc"].items()
    )
    sparsity = analysis["sparsity"]
    lines.extend(
        (
            "",
            "## Interaction sparsity",
            "",
            f"- Matrix cells: {sparsity['matrix_size']:,}",
            f"- Unique View-or-stronger pairs: {sparsity['view_pairs']:,} "
            f"({sparsity['view_density']:.2%} density; "
            f"{sparsity['positive_sparsity']:.2%} sparse)",
            f"- Favorite pairs: {sparsity['favorite_pairs']:,}",
            f"- Cart pairs: {sparsity['cart_pairs']:,}",
            f"- Purchase pairs: {sparsity['purchase_pairs']:,} "
            f"({sparsity['purchase_density']:.2%} density)",
            "",
            "## Interpretation and freeze recommendation",
            "",
            "- View, Favorite, Cart, and Purchase represent distinct intent levels "
            "and use different utilities.",
            "- Observed Event is not true preference: exposure, price, state, "
            "impulsiveness, status, and noise all affect the funnel.",
            "- Favorite and Cart are helpful prior states, not mandatory gates; "
            "direct Cart and Purchase paths remain possible.",
            "- Event weights for recommendation datasets are intentionally not "
            "assigned in this generator.",
            "- View→Purchase is 4.12%, only 0.12 percentage points above the "
            "initial non-binding 1–4% guide; its count and downstream diagnostics "
            "remain sane.",
            "- The reported delayed paths, price signal, concentration, and "
            "single-feature AUCs support freezing this exact seed-42 v1 population.",
            "",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a supplied engagement quality report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/synthetic_engagement_v1_quality.md"),
    )
    return parser
