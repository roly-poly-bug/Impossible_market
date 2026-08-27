from __future__ import annotations

from collections import defaultdict

from backend.app.db.models import EventType
from synthetic_data.event_generator import SyntheticEventRecord
from synthetic_data.interaction_config import INTERACTION_GENERATION_VERSION
from synthetic_data.product_generator import SyntheticProductRecord
from synthetic_data.session_generator import SimulationWindow, SyntheticSessionRecord
from synthetic_data.user_generator import SyntheticUserRecord


class InteractionValidationError(ValueError):
    """Raised before export or persistence when interaction records are invalid."""


def validate_interactions(
    sessions: list[SyntheticSessionRecord],
    events: list[SyntheticEventRecord],
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
    window: SimulationWindow,
    *,
    expected_seed: int,
) -> None:
    errors = []
    session_ids = [session.session_id for session in sessions]
    event_ids = [event.event_id for event in events]
    users_by_id = {user.user_id: user for user in users}
    product_ids = {product.name for product in products}
    sessions_by_id = {session.session_id: session for session in sessions}

    if len(session_ids) != len(set(session_ids)):
        errors.append("session IDs must be unique")
    if len(event_ids) != len(set(event_ids)):
        errors.append("event IDs must be unique")

    for session in sessions:
        if session.user_id not in users_by_id:
            errors.append(f"{session.session_id}: unknown user {session.user_id}")
        if not window.start <= session.started_at < window.end_exclusive:
            errors.append(f"{session.session_id}: start is outside the simulation window")
        if not session.started_at <= session.ended_at < window.end_exclusive:
            errors.append(f"{session.session_id}: invalid end timestamp")
        if session.generation_version != INTERACTION_GENERATION_VERSION:
            errors.append(f"{session.session_id}: invalid generation version")
        if session.generation_seed != expected_seed:
            errors.append(f"{session.session_id}: invalid generation seed")

    events_by_session: dict[str, list[SyntheticEventRecord]] = defaultdict(list)
    for event in events:
        events_by_session[event.session_id].append(event)
        session = sessions_by_id.get(event.session_id)
        if session is None:
            errors.append(f"{event.event_id}: unknown session {event.session_id}")
            continue
        if event.user_id not in users_by_id:
            errors.append(f"{event.event_id}: unknown user {event.user_id}")
        if event.product_id not in product_ids:
            errors.append(f"{event.event_id}: unknown product {event.product_id}")
        if event.user_id != session.user_id:
            errors.append(f"{event.event_id}: event user differs from session user")
        if not session.started_at <= event.timestamp <= session.ended_at:
            errors.append(f"{event.event_id}: timestamp is outside its session")
        if event.event_type not in (EventType.IMPRESSION, EventType.VIEW):
            errors.append(f"{event.event_id}: invalid v1 event type")
        if event.generation_version != INTERACTION_GENERATION_VERSION:
            errors.append(f"{event.event_id}: invalid generation version")
        if event.generation_seed != expected_seed:
            errors.append(f"{event.event_id}: invalid generation seed")

    for session in sessions:
        session_events = events_by_session.get(session.session_id, [])
        impressed_products = set()
        previous_timestamp = None
        impression_count = 0
        for event in session_events:
            if previous_timestamp is not None and event.timestamp <= previous_timestamp:
                errors.append(f"{session.session_id}: event timestamps must strictly increase")
                break
            previous_timestamp = event.timestamp
            if event.event_type == EventType.IMPRESSION:
                impression_count += 1
                if event.product_id in impressed_products:
                    errors.append(
                        f"{session.session_id}: duplicate impression for {event.product_id}"
                    )
                impressed_products.add(event.product_id)
            elif event.product_id not in impressed_products:
                errors.append(
                    f"{event.event_id}: View requires an earlier same-session Impression"
                )
        if impression_count != session.impression_target:
            errors.append(
                f"{session.session_id}: expected {session.impression_target} impressions, "
                f"found {impression_count}"
            )

    if errors:
        displayed = errors[:50]
        remaining = len(errors) - len(displayed)
        suffix = f"\n- ... and {remaining} more" if remaining else ""
        raise InteractionValidationError(
            "Interaction validation failed:\n- " + "\n- ".join(displayed) + suffix
        )

