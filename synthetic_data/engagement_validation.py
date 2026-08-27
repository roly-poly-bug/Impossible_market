from __future__ import annotations

from collections import defaultdict

from backend.app.db.models import EventType, ProductStatus
from synthetic_data.engagement_config import ENGAGEMENT_GENERATION_VERSION
from synthetic_data.engagement_generator import SyntheticEngagementRecord
from synthetic_data.event_generator import SyntheticEventRecord
from synthetic_data.interaction_config import INTERACTION_GENERATION_VERSION
from synthetic_data.product_generator import SyntheticProductRecord
from synthetic_data.session_generator import SimulationWindow, SyntheticSessionRecord
from synthetic_data.user_generator import SyntheticUserRecord


class EngagementValidationError(ValueError):
    """Raised before export or persistence when engagement records are invalid."""


def validate_engagement_events(
    engagement_events: list[SyntheticEngagementRecord],
    base_events: list[SyntheticEventRecord],
    sessions: list[SyntheticSessionRecord],
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
    window: SimulationWindow,
    *,
    expected_seed: int,
) -> None:
    errors = []
    event_ids = [event.event_id for event in engagement_events]
    if len(event_ids) != len(set(event_ids)):
        errors.append("engagement event IDs must be unique")

    valid_types = {EventType.FAVORITE, EventType.ADD_TO_CART, EventType.PURCHASE}
    users_by_id = {user.user_id: user for user in users}
    products_by_id = {product.name: product for product in products}
    sessions_by_id = {session.session_id: session for session in sessions}
    views_by_id = {
        event.event_id: event
        for event in base_events
        if event.event_type == EventType.VIEW
    }
    pair_type_counts = defaultdict(int)
    combined_by_session = defaultdict(list)
    for event in base_events:
        combined_by_session[event.session_id].append(event)

    for event in engagement_events:
        combined_by_session[event.session_id].append(event)
        session = sessions_by_id.get(event.session_id)
        product = products_by_id.get(event.product_id)
        source_view = views_by_id.get(event.source_view_event_id)
        pair_type_counts[(event.user_id, event.product_id, event.event_type)] += 1

        if event.event_type not in valid_types:
            errors.append(f"{event.event_id}: invalid engagement event type")
        if event.user_id not in users_by_id:
            errors.append(f"{event.event_id}: unknown user")
        if product is None:
            errors.append(f"{event.event_id}: unknown product")
        if session is None:
            errors.append(f"{event.event_id}: unknown session")
        elif event.user_id != session.user_id:
            errors.append(f"{event.event_id}: Session user mismatch")
        elif not session.started_at <= event.timestamp <= session.ended_at:
            errors.append(f"{event.event_id}: timestamp outside Session")
        if not window.start <= event.timestamp < window.end_exclusive:
            errors.append(f"{event.event_id}: timestamp outside simulation window")
        if source_view is None:
            errors.append(f"{event.event_id}: requires a valid prior View")
        elif (
            source_view.user_id != event.user_id
            or source_view.product_id != event.product_id
            or source_view.timestamp >= event.timestamp
        ):
            errors.append(f"{event.event_id}: prior View user/product/time mismatch")
        if event.source_view_session_id != (source_view.session_id if source_view else None):
            errors.append(f"{event.event_id}: source View Session mismatch")
        expected_timing = (
            "same_session"
            if event.session_id == event.source_view_session_id
            else "later_session"
        )
        if event.conversion_timing != expected_timing:
            errors.append(f"{event.event_id}: invalid conversion timing label")
        if event.event_type in (EventType.ADD_TO_CART, EventType.PURCHASE):
            if product is not None and product.status != ProductStatus.AVAILABLE:
                errors.append(f"{event.event_id}: Cart/Purchase product must be available")
        if event.generation_version != ENGAGEMENT_GENERATION_VERSION:
            errors.append(f"{event.event_id}: invalid engagement version")
        if event.generation_seed != expected_seed:
            errors.append(f"{event.event_id}: invalid engagement seed")
        if event.upstream_generation_version != INTERACTION_GENERATION_VERSION:
            errors.append(f"{event.event_id}: invalid upstream version")
        if event.upstream_generation_seed != 42:
            errors.append(f"{event.event_id}: invalid upstream seed")

    duplicates = [key for key, count in pair_type_counts.items() if count > 1]
    if duplicates:
        errors.append(f"duplicate user-product engagement conversions: {duplicates[:5]}")

    for session_id, events in combined_by_session.items():
        ordered = sorted(events, key=lambda event: event.timestamp)
        if any(
            left.timestamp >= right.timestamp
            for left, right in zip(ordered, ordered[1:], strict=False)
        ):
            errors.append(f"{session_id}: combined event timestamps must strictly increase")

    if errors:
        displayed = errors[:50]
        remaining = len(errors) - len(displayed)
        suffix = f"\n- ... and {remaining} more" if remaining else ""
        raise EngagementValidationError(
            "Engagement validation failed:\n- " + "\n- ".join(displayed) + suffix
        )
