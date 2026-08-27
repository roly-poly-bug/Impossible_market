from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import fmean, median, pstdev

from backend.app.db.models import EventType
from synthetic_data.event_generator import SyntheticEventRecord
from synthetic_data.session_generator import SyntheticSessionRecord
from synthetic_data.user_generator import SyntheticUserRecord


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "25%": _percentile(values, 0.25),
        "median": median(values),
        "75%": _percentile(values, 0.75),
        "max": max(values),
    }


def summarize_interactions(
    sessions: list[SyntheticSessionRecord],
    events: list[SyntheticEventRecord],
    users: list[SyntheticUserRecord],
) -> dict[str, object]:
    user_by_id = {user.user_id: user for user in users}
    sessions_per_user = Counter(session.user_id for session in sessions)
    impressions_per_session = Counter(
        event.session_id for event in events if event.event_type == EventType.IMPRESSION
    )
    views_per_session = Counter(
        event.session_id for event in events if event.event_type == EventType.VIEW
    )
    impression_count = sum(impressions_per_session.values())
    view_count = sum(views_per_session.values())
    tier_sessions: dict[str, list[int]] = defaultdict(list)
    for user_id, count in sessions_per_user.items():
        tier_sessions[user_by_id[user_id].activity_tier.value].append(count)
    return {
        "total_sessions": len(sessions),
        "total_events": len(events),
        "total_impressions": impression_count,
        "total_views": view_count,
        "view_rate": view_count / impression_count,
        "sessions_per_user": stats(
            [float(sessions_per_user.get(user.user_id, 0)) for user in users]
        ),
        "sessions_by_activity_tier": {
            tier: stats([float(value) for value in values])
            for tier, values in sorted(tier_sessions.items())
        },
        "session_duration_seconds": stats(
            [(session.ended_at - session.started_at).total_seconds() for session in sessions]
        ),
        "impressions_per_session": stats(
            [float(impressions_per_session[session.session_id]) for session in sessions]
        ),
        "views_per_session": stats(
            [float(views_per_session.get(session.session_id, 0)) for session in sessions]
        ),
        "entry_type_distribution": dict(Counter(session.entry_type.value for session in sessions)),
        "exposure_source_distribution": dict(
            Counter(
                event.exposure_source
                for event in events
                if event.event_type == EventType.IMPRESSION
            )
        ),
    }


def format_interaction_summary(summary: dict[str, object]) -> str:
    sessions = summary["sessions_per_user"]
    durations = summary["session_duration_seconds"]
    impressions = summary["impressions_per_session"]
    views = summary["views_per_session"]
    lines = [
        f"Sessions: {summary['total_sessions']}",
        f"Events: {summary['total_events']}",
        f"Impressions: {summary['total_impressions']}",
        f"Views: {summary['total_views']}",
        f"Overall view rate: {summary['view_rate']:.2%}",
        "",
        "Sessions per user:",
        f"  mean={sessions['mean']:.3f}, std={sessions['std']:.3f}, "
        f"min={sessions['min']:.0f}, median={sessions['median']:.1f}, max={sessions['max']:.0f}",
        "",
        "Sessions per user by activity tier:",
    ]
    lines.extend(
        f"  {tier}: mean={values['mean']:.3f}, std={values['std']:.3f}, "
        f"min={values['min']:.0f}, median={values['median']:.1f}, max={values['max']:.0f}"
        for tier, values in summary["sessions_by_activity_tier"].items()
    )
    lines.extend(
        (
            "",
            "Session duration (seconds):",
            f"  mean={durations['mean']:.1f}, median={durations['median']:.1f}, "
            f"min={durations['min']:.0f}, max={durations['max']:.0f}",
            "",
            "Impressions per session:",
            f"  mean={impressions['mean']:.3f}, std={impressions['std']:.3f}, "
            f"min={impressions['min']:.0f}, median={impressions['median']:.1f}, max={impressions['max']:.0f}",
            "",
            "Views per session:",
            f"  mean={views['mean']:.3f}, std={views['std']:.3f}, "
            f"min={views['min']:.0f}, median={views['median']:.1f}, max={views['max']:.0f}",
            "",
            "Entry types:",
        )
    )
    lines.extend(f"  {name}: {count}" for name, count in summary["entry_type_distribution"].items())
    lines.extend(("", "Exposure sources:"))
    lines.extend(
        f"  {name}: {count}" for name, count in summary["exposure_source_distribution"].items()
    )
    return "\n".join(lines)

