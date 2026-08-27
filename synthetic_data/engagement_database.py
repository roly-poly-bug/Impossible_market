from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Event, EventType, Product, Session as BrowsingSession, User
from synthetic_data.engagement_config import ENGAGEMENT_GENERATION_VERSION
from synthetic_data.engagement_generator import SyntheticEngagementRecord
from synthetic_data.engagement_validation import validate_engagement_events
from synthetic_data.event_generator import SyntheticEventRecord
from synthetic_data.product_generator import SyntheticProductRecord
from synthetic_data.session_generator import SimulationWindow, SyntheticSessionRecord
from synthetic_data.user_generator import SyntheticUserRecord


@dataclass(frozen=True)
class EngagementWriteResult:
    events_created: int
    events_updated: int
    events_deleted: int


def write_engagement_events(
    database: Session,
    engagement_events: list[SyntheticEngagementRecord],
    base_events: list[SyntheticEventRecord],
    sessions: list[SyntheticSessionRecord],
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
    window: SimulationWindow,
    *,
    replace_existing: bool = False,
) -> EngagementWriteResult:
    if not engagement_events:
        raise ValueError("cannot write an empty engagement population")
    requested_seed = engagement_events[0].generation_seed
    validate_engagement_events(
        engagement_events,
        base_events,
        sessions,
        users,
        products,
        window,
        expected_seed=requested_seed,
    )

    existing = list(
        database.scalars(
            select(Event).where(Event.generation_version == ENGAGEMENT_GENERATION_VERSION)
        ).all()
    )
    requested_ids = {event.event_id for event in engagement_events}
    existing_ids = {event.event_key for event in existing}
    existing_seeds = {event.generation_seed for event in existing}
    population_differs = bool(existing) and (
        requested_ids != existing_ids or existing_seeds != {requested_seed}
    )
    if population_differs and not replace_existing:
        raise RuntimeError(
            "A different synthetic_engagement_v1 population exists. "
            "Re-run with --replace-existing to replace only engagement Events."
        )

    deleted = 0
    if replace_existing:
        protected = [
            event
            for event in existing
            if event.event_type
            not in (EventType.FAVORITE, EventType.ADD_TO_CART, EventType.PURCHASE)
        ]
        if protected:
            raise RuntimeError("Engagement population contains protected Event types.")
        for event in existing:
            database.delete(event)
        deleted = len(existing)
        database.flush()
        existing = []

    user_ids = {event.user_id for event in engagement_events}
    product_ids = {event.product_id for event in engagement_events}
    session_ids = {event.session_id for event in engagement_events}
    stored_users = {
        user.external_id: user
        for user in database.scalars(select(User)).all()
        if user.external_id in user_ids
    }
    stored_products = {
        product.name: product
        for product in database.scalars(select(Product)).all()
        if product.name in product_ids
    }
    stored_sessions = {
        session.session_key: session
        for session in database.scalars(select(BrowsingSession)).all()
        if session.session_key in session_ids
    }
    if (
        set(stored_users) != user_ids
        or set(stored_products) != product_ids
        or set(stored_sessions) != session_ids
    ):
        raise RuntimeError(
            "Frozen Product/User/Session world must be written before engagement Events."
        )

    stored_events = {event.event_key: event for event in existing}
    created = 0
    updated = 0
    if not stored_events:
        database.bulk_insert_mappings(
            Event,
            [
                {
                    "event_key": event.event_id,
                    "event_type": event.event_type,
                    "user_id": stored_users[event.user_id].id,
                    "product_id": stored_products[event.product_id].id,
                    "session_id": stored_sessions[event.session_id].id,
                    "occurred_at": event.timestamp,
                    "exposure_source": event.conversion_timing,
                    "generation_version": event.generation_version,
                    "generation_seed": event.generation_seed,
                }
                for event in engagement_events
            ],
        )
        created = len(engagement_events)
    else:
        for event in engagement_events:
            stored = stored_events[event.event_id]
            stored.event_type = event.event_type
            stored.user = stored_users[event.user_id]
            stored.product = stored_products[event.product_id]
            stored.session = stored_sessions[event.session_id]
            stored.occurred_at = event.timestamp
            stored.exposure_source = event.conversion_timing
            stored.generation_seed = event.generation_seed
            updated += 1

    database.commit()
    return EngagementWriteResult(created, updated, deleted)
