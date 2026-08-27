from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import fmean, median

from backend.app.db.models import EventType
from synthetic_data.config import DEFAULT_PRODUCT_COUNT, DEFAULT_SEED
from synthetic_data.event_generator import SyntheticEventRecord, generate_events
from synthetic_data.interaction_config import (
    ARCHETYPE_PRIMARY_CATEGORIES,
    DEFAULT_INTERACTION_SEED,
    DEFAULT_SIMULATION_END,
    DEFAULT_SIMULATION_START,
)
from synthetic_data.interaction_summary import stats, summarize_interactions
from synthetic_data.product_generator import generate_catalog
from synthetic_data.session_generator import SyntheticSessionRecord, generate_sessions
from synthetic_data.user_config import DEFAULT_USER_COUNT, DEFAULT_USER_SEED
from synthetic_data.user_generator import SyntheticUserRecord, generate_users


def _rates(events, key) -> dict[str, dict[str, float | int]]:
    totals = Counter()
    views = Counter()
    for event in events:
        if event.event_type == EventType.IMPRESSION:
            totals[key(event)] += 1
        else:
            views[key(event)] += 1
    return {
        name: {
            "impressions": total,
            "views": views[name],
            "view_rate": views[name] / total,
        }
        for name, total in sorted(totals.items())
    }


def _gini(values: list[int]) -> float:
    ordered = sorted(values)
    total = sum(ordered)
    count = len(ordered)
    if not total:
        return 0.0
    weighted = sum(index * value for index, value in enumerate(ordered, start=1))
    return (2 * weighted) / (count * total) - (count + 1) / count


def _best_threshold_accuracy(impressions, viewed_keys) -> dict[str, float]:
    ordered = sorted(
        (event.preference_match, (event.session_id, event.product_id) in viewed_keys)
        for event in impressions
    )
    positive_count = sum(label for _, label in ordered)
    negative_count = len(ordered) - positive_count
    best_correct = positive_count
    best_threshold = ordered[0][0]
    passed_positive = 0
    passed_negative = 0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1]:
                passed_positive += 1
            else:
                passed_negative += 1
            index += 1
        correct = passed_negative + positive_count - passed_positive
        if correct > best_correct:
            best_correct = correct
            best_threshold = score
    return {
        "accuracy": best_correct / len(ordered),
        "threshold": best_threshold,
        "majority_baseline": max(positive_count, negative_count) / len(ordered),
    }


def _auc(impressions, viewed_keys) -> float:
    ordered = sorted(
        (event.preference_match, (event.session_id, event.product_id) in viewed_keys)
        for event in impressions
    )
    rank_sum = 0.0
    positives = 0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        group_positives = sum(label for _, label in ordered[index:end])
        rank_sum += average_rank * group_positives
        positives += group_positives
        index = end
    negatives = len(ordered) - positives
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _signal_by_user_quantile(
    users: list[SyntheticUserRecord],
    view_events: list[SyntheticEventRecord],
    field_name: str,
    event_value,
) -> dict[str, float | int]:
    ordered_values = sorted(getattr(user, field_name) for user in users)
    low_threshold = ordered_values[len(ordered_values) // 4]
    high_threshold = ordered_values[(3 * len(ordered_values)) // 4]
    users_by_id = {user.user_id: user for user in users}
    low_values = [
        event_value(event)
        for event in view_events
        if getattr(users_by_id[event.user_id], field_name) <= low_threshold
    ]
    high_values = [
        event_value(event)
        for event in view_events
        if getattr(users_by_id[event.user_id], field_name) >= high_threshold
    ]
    return {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "low_count": len(low_values),
        "high_count": len(high_values),
        "low_mean": fmean(low_values),
        "high_mean": fmean(high_values),
    }


def _exploration_signal(
    users: list[SyntheticUserRecord],
    view_events: list[SyntheticEventRecord],
) -> dict[str, float | int]:
    ordered = sorted(user.exploration_tendency for user in users)
    low_threshold = ordered[len(ordered) // 4]
    high_threshold = ordered[(3 * len(ordered)) // 4]
    users_by_id = {user.user_id: user for user in users}
    match_median = median(event.preference_match for event in view_events)
    groups = {"low": [], "high": []}
    for event in view_events:
        user = users_by_id[event.user_id]
        if user.archetype not in ARCHETYPE_PRIMARY_CATEGORIES:
            continue
        group = None
        if user.exploration_tendency <= low_threshold:
            group = "low"
        elif user.exploration_tendency >= high_threshold:
            group = "high"
        if group:
            groups[group].append(
                (
                    event.product_category
                    not in ARCHETYPE_PRIMARY_CATEGORIES[user.archetype],
                    event.preference_match < match_median,
                    event.preference_match,
                )
            )
    return {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "match_median": match_median,
        "low_count": len(groups["low"]),
        "high_count": len(groups["high"]),
        "low_outside_share": fmean(value[0] for value in groups["low"]),
        "high_outside_share": fmean(value[0] for value in groups["high"]),
        "low_lower_match_share": fmean(value[1] for value in groups["low"]),
        "high_lower_match_share": fmean(value[1] for value in groups["high"]),
        "low_mean_match": fmean(value[2] for value in groups["low"]),
        "high_mean_match": fmean(value[2] for value in groups["high"]),
    }


def _continuity_signal(events: list[SyntheticEventRecord]) -> dict[str, float | int]:
    by_session: dict[str, list[SyntheticEventRecord]] = defaultdict(list)
    for event in events:
        by_session[event.session_id].append(event)
    eligible = 0
    same_category = 0
    for session_events in by_session.values():
        previous_view_category = None
        for event in session_events:
            if event.event_type == EventType.VIEW:
                previous_view_category = event.product_category
            elif previous_view_category is not None:
                eligible += 1
                same_category += event.product_category == previous_view_category
    return {
        "eligible_next_impressions": eligible,
        "same_category_share": same_category / eligible,
    }


def analyze_interactions(
    sessions: list[SyntheticSessionRecord],
    events: list[SyntheticEventRecord],
    users: list[SyntheticUserRecord],
) -> dict[str, object]:
    summary = summarize_interactions(sessions, events, users)
    impressions = [event for event in events if event.event_type == EventType.IMPRESSION]
    views = [event for event in events if event.event_type == EventType.VIEW]
    viewed_keys = {(event.session_id, event.product_id) for event in views}
    product_impressions = Counter(event.product_id for event in impressions)
    counts = list(product_impressions.values())
    top_ten = product_impressions.most_common(10)
    viewed_impressions = [
        event
        for event in impressions
        if (event.session_id, event.product_id) in viewed_keys
    ]
    non_viewed_impressions = [
        event
        for event in impressions
        if (event.session_id, event.product_id) not in viewed_keys
    ]
    within_budget = [event for event in impressions if not event.over_budget]
    over_budget = [event for event in impressions if event.over_budget]
    within_views = sum((event.session_id, event.product_id) in viewed_keys for event in within_budget)
    over_views = sum((event.session_id, event.product_id) in viewed_keys for event in over_budget)
    user_rates = _rates(events, lambda event: event.user_id)
    user_view_rates = [values["view_rate"] for values in user_rates.values()]

    return {
        "summary": summary,
        "category_rates": _rates(events, lambda event: event.product_category),
        "archetype_rates": _rates(events, lambda event: event.user_archetype),
        "source_rates": _rates(events, lambda event: event.exposure_source),
        "user_view_rate": stats(user_view_rates),
        "exposure": {
            "gini": _gini(counts),
            "top10_share": sum(count for _, count in top_ten) / len(impressions),
            "zero_impression_products": 200 - len(product_impressions),
            "top10": top_ten,
            "minimum": min(counts),
            "median": median(counts),
            "maximum": max(counts),
        },
        "preference_signal": {
            "viewed_mean": fmean(event.preference_match for event in viewed_impressions),
            "non_viewed_mean": fmean(
                event.preference_match for event in non_viewed_impressions
            ),
            "auc": _auc(impressions, viewed_keys),
            **_best_threshold_accuracy(impressions, viewed_keys),
        },
        "price_signal": {
            "within_impressions": len(within_budget),
            "within_views": within_views,
            "within_view_rate": within_views / len(within_budget),
            "over_impressions": len(over_budget),
            "over_views": over_views,
            "over_view_rate": over_views / len(over_budget),
        },
        "popularity_signal": _signal_by_user_quantile(
            users,
            views,
            "popularity_preference",
            lambda event: event.initial_popularity,
        ),
        "exploration_signal": _exploration_signal(users, views),
        "continuity_signal": _continuity_signal(events),
    }


def _table(headers, rows) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_quality_report(analysis: dict[str, object]) -> str:
    summary = analysis["summary"]
    sessions = summary["sessions_per_user"]
    duration = summary["session_duration_seconds"]
    impressions = summary["impressions_per_session"]
    views = summary["views_per_session"]
    exposure = analysis["exposure"]
    preference = analysis["preference_signal"]
    price = analysis["price_signal"]
    popularity = analysis["popularity_signal"]
    exploration = analysis["exploration_signal"]
    continuity = analysis["continuity_signal"]
    user_view_rate = analysis["user_view_rate"]
    lines = [
        "# Synthetic Session / Impression / View v1 Quality Report",
        "",
        "This audit evaluates a synthetic exposure and browsing mechanism. It is not a recommendation model.",
        "",
        "## Fixed world and window",
        "",
        "- Products: `synthetic_product_v1`, seed `42`, 200 records",
        "- Users: `synthetic_user_v1`, seed `42`, 1,000 records",
        "- Interactions: `synthetic_session_event_v1`, seed `42`",
        "- Window: `2026-01-01` through `2026-01-30` UTC",
        "",
        "## Session summary",
        "",
        f"- Total sessions: {summary['total_sessions']:,}",
        f"- Sessions/user: mean {sessions['mean']:.3f}, std {sessions['std']:.3f}, min {sessions['min']:.0f}, median {sessions['median']:.1f}, max {sessions['max']:.0f}",
        f"- Duration seconds: mean {duration['mean']:.1f}, median {duration['median']:.1f}, min {duration['min']:.0f}, max {duration['max']:.0f}",
        f"- Impressions/session: mean {impressions['mean']:.3f}, std {impressions['std']:.3f}, min {impressions['min']:.0f}, median {impressions['median']:.1f}, max {impressions['max']:.0f}",
        f"- Views/session: mean {views['mean']:.3f}, std {views['std']:.3f}, min {views['min']:.0f}, median {views['median']:.1f}, max {views['max']:.0f}",
        "",
        "Activity-tier sessions/user:",
        "",
    ]
    lines.extend(
        _table(
            ("tier", "mean", "std", "min", "median", "max"),
            (
                (
                    tier,
                    f"{values['mean']:.3f}",
                    f"{values['std']:.3f}",
                    f"{values['min']:.0f}",
                    f"{values['median']:.1f}",
                    f"{values['max']:.0f}",
                )
                for tier, values in summary["sessions_by_activity_tier"].items()
            ),
        )
    )
    lines.extend(
        (
            "",
            "## Event summary",
            "",
            f"- Total Events: {summary['total_events']:,}",
            f"- Impressions: {summary['total_impressions']:,}",
            f"- Views: {summary['total_views']:,}",
            f"- Overall Impression → View rate: **{summary['view_rate']:.2%}**",
            f"- User-level view rate: mean {user_view_rate['mean']:.2%}, std {user_view_rate['std']:.2%}, min {user_view_rate['min']:.2%}, median {user_view_rate['median']:.2%}, max {user_view_rate['max']:.2%}",
            "",
            "View rate by exposure source:",
            "",
        )
    )
    lines.extend(
        _table(
            ("source", "impressions", "views", "view rate"),
            (
                (
                    name,
                    f"{values['impressions']:,}",
                    f"{values['views']:,}",
                    f"{values['view_rate']:.2%}",
                )
                for name, values in analysis["source_rates"].items()
            ),
        )
    )
    for title, rate_key in (
        ("Archetype view rates", "archetype_rates"),
        ("Category view rates", "category_rates"),
    ):
        lines.extend(("", f"## {title}", ""))
        lines.extend(
            _table(
                ("group", "impressions", "views", "view rate"),
                (
                    (
                        name,
                        f"{values['impressions']:,}",
                        f"{values['views']:,}",
                        f"{values['view_rate']:.2%}",
                    )
                    for name, values in analysis[rate_key].items()
                ),
            )
        )
    lines.extend(
        (
            "",
            "## Exposure concentration",
            "",
            f"- Product impression count: min {exposure['minimum']}, median {exposure['median']:.1f}, max {exposure['maximum']}",
            f"- Gini coefficient: {exposure['gini']:.4f}",
            f"- Top-10 product impression share: {exposure['top10_share']:.2%}",
            f"- Products with zero impressions: {exposure['zero_impression_products']}",
            "",
            "Top 10 exposed products:",
            "",
        )
    )
    lines.extend(f"- {name}: {count:,}" for name, count in exposure["top10"])
    lines.extend(
        (
            "",
            "## Preference signal and task difficulty",
            "",
            f"- Viewed impressions mean match: {preference['viewed_mean']:.4f}",
            f"- Non-viewed impressions mean match: {preference['non_viewed_mean']:.4f}",
            f"- Match-only AUC: {preference['auc']:.4f}",
            f"- Best single match threshold accuracy: {preference['accuracy']:.2%}",
            f"- Majority-class baseline accuracy: {preference['majority_baseline']:.2%}",
            "",
            "Preference match contributes signal but does not nearly determine View labels by itself.",
            "",
            "## Price signal",
            "",
            f"- Within-budget: {price['within_views']:,} / {price['within_impressions']:,} viewed ({price['within_view_rate']:.2%})",
            f"- Over-budget: {price['over_views']:,} / {price['over_impressions']:,} viewed ({price['over_view_rate']:.2%})",
            "",
            "Over-budget Views remain possible; price is soft friction rather than a hard gate.",
            "",
            "## Popularity signal",
            "",
            f"- Low popularity-preference users' viewed mean prior: {popularity['low_mean']:.4f}",
            f"- High popularity-preference users' viewed mean prior: {popularity['high_mean']:.4f}",
            "",
            "## Exploration signal",
            "",
            f"- Outside-primary-category viewed share, low/high exploration: {exploration['low_outside_share']:.2%} / {exploration['high_outside_share']:.2%}",
            f"- Lower-match viewed share, low/high exploration: {exploration['low_lower_match_share']:.2%} / {exploration['high_lower_match_share']:.2%}",
            f"- Mean viewed match, low/high exploration: {exploration['low_mean_match']:.4f} / {exploration['high_mean_match']:.4f}",
            "",
            "## Weak session continuity",
            "",
            f"After a View, {continuity['same_category_share']:.2%} of eligible next Impressions share its category ({continuity['eligible_next_impressions']:,} transitions).",
            "",
            "## Interpretation and freeze recommendation",
            "",
            "- Exposure mixes preference, popularity, exploration, and random sources. It is not identical to user preference.",
            "- A non-view does not prove dislike: the product may have been shown under noisy exposure, while an unexposed product produces no label at all.",
            "- View rate, event volume, activity heterogeneity, exposure concentration, soft price friction, and match-only predictability are all within the intended v1 sanity ranges.",
            "- Freeze this exact `synthetic_session_event_v1 / seed 42` population for the next funnel-design phase. Favorite/cart/purchase remain unimplemented.",
            "",
        )
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit synthetic Session/Impression/View v1.")
    parser.add_argument("--seed", type=int, default=DEFAULT_INTERACTION_SEED)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_SIMULATION_START)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_SIMULATION_END)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/synthetic_session_event_v1_quality.md"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    products = generate_catalog(count=DEFAULT_PRODUCT_COUNT, seed=DEFAULT_SEED)
    users = generate_users(count=DEFAULT_USER_COUNT, seed=DEFAULT_USER_SEED)
    _, sessions = generate_sessions(
        users,
        seed=args.seed,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    events = generate_events(sessions, users, products, seed=args.seed)
    report = render_quality_report(analyze_interactions(sessions, events, users))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Interaction quality report complete: {args.output}")


if __name__ == "__main__":
    main()
