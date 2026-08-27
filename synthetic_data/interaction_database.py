from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Event, EventType, Product, Session as BrowsingSession, User
from synthetic_data.event_generator import SyntheticEventRecord
from synthetic_data.interaction_config import INTERACTION_GENERATION_VERSION
from synthetic_data.interaction_validation import validate_interactions
from synthetic_data.product_generator import SyntheticProductRecord
from synthetic_data.session_generator import SimulationWindow, SyntheticSessionRecord
from synthetic_data.user_generator import SyntheticUserRecord


@dataclass(frozen=True)
class InteractionWriteResult:
    sessions_created: int
    sessions_updated: int
    sessions_deleted: int
    events_created: int
    events_updated: int
    events_deleted: int


def _delete_existing_interactions(
    database: Session,
    sessions: list[BrowsingSession],
) -> tuple[int, int]:
    if not sessions:
        return 0, 0
    session_ids = [session.id for session in sessions]
    attached_events = list(
        database.scalars(select(Event).where(Event.session_id.in_(session_ids))).all()
    )
    protected = [
        event
        for event in attached_events
        if event.generation_version != INTERACTION_GENERATION_VERSION
        or event.event_type not in (EventType.IMPRESSION, EventType.VIEW)
    ]
    if protected:
        raise RuntimeError(
            "Synthetic Sessions have downstream or non-v1 Events and cannot be safely replaced."
        )
    for event in attached_events:
        database.delete(event)
    for session in sessions:
        database.delete(session)
    database.flush()
    return len(sessions), len(attached_events)


def write_interactions(
    database: Session,
    sessions: list[SyntheticSessionRecord],
    events: list[SyntheticEventRecord],
    users: list[SyntheticUserRecord],
    products: list[SyntheticProductRecord],
    window: SimulationWindow,
    *,
    replace_existing: bool = False,
) -> InteractionWriteResult:
    if not sessions or not events:
        raise ValueError("cannot write an empty Session/Event population")
    requested_seed = sessions[0].generation_seed
    validate_interactions(
        sessions,
        events,
        users,
        products,
        window,
        expected_seed=requested_seed,
    )

    requested_session_ids = {record.session_id for record in sessions}
    requested_event_ids = {record.event_id for record in events}
    existing_sessions = list(
        database.scalars(
            select(BrowsingSession).where(
                BrowsingSession.generation_version == INTERACTION_GENERATION_VERSION
            )
        ).all()
    )
    existing_events = list(
        database.scalars(
            select(Event).where(Event.generation_version == INTERACTION_GENERATION_VERSION)
        ).all()
    )
    existing_session_ids = {record.session_key for record in existing_sessions}
    existing_event_ids = {record.event_key for record in existing_events}
    existing_seeds = {
        *(record.generation_seed for record in existing_sessions),
        *(record.generation_seed for record in existing_events),
    }
    population_differs = (existing_sessions or existing_events) and (
        existing_session_ids != requested_session_ids
        or existing_event_ids != requested_event_ids
        or existing_seeds != {requested_seed}
    )
    if population_differs and not replace_existing:
        raise RuntimeError(
            "A different synthetic_session_event_v1 population already exists. "
            "Re-run with --replace-existing to replace only that interaction population."
        )

    sessions_deleted = 0
    events_deleted = 0
    if replace_existing:
        sessions_deleted, events_deleted = _delete_existing_interactions(
            database,
            existing_sessions,
        )
        existing_sessions = []
        existing_events = []

    requested_user_ids = {record.user_id for record in sessions}
    requested_product_ids = {record.product_id for record in events}
    stored_users = {
        user.external_id: user
        for user in database.scalars(select(User)).all()
        if user.external_id in requested_user_ids
    }
    stored_products = {
        product.name: product
        for product in database.scalars(select(Product)).all()
        if product.name in requested_product_ids
    }
    missing_users = requested_user_ids - stored_users.keys()
    missing_products = requested_product_ids - stored_products.keys()
    if missing_users or missing_products:
        raise RuntimeError(
            "Frozen Product/User world must be written before interactions: "
            f"{len(missing_users)} users and {len(missing_products)} products are missing."
        )

    stored_sessions = {record.session_key: record for record in existing_sessions}
    sessions_created = 0
    sessions_updated = 0
    for record in sessions:
        stored = stored_sessions.get(record.session_id)
        if stored is None:
            stored = BrowsingSession(session_key=record.session_id)
            database.add(stored)
            stored_sessions[record.session_id] = stored
            sessions_created += 1
        else:
            sessions_updated += 1
        stored.user = stored_users[record.user_id]
        stored.started_at = record.started_at
        stored.ended_at = record.ended_at
        stored.entry_type = record.entry_type
        stored.generation_version = record.generation_version
        stored.generation_seed = record.generation_seed
    database.flush()

    stored_events = {record.event_key: record for record in existing_events}
    events_created = 0
    events_updated = 0
    if not stored_events:
        database.bulk_insert_mappings(
            Event,
            [
                {
                    "event_key": record.event_id,
                    "event_type": record.event_type,
                    "user_id": stored_users[record.user_id].id,
                    "product_id": stored_products[record.product_id].id,
                    "session_id": stored_sessions[record.session_id].id,
                    "occurred_at": record.timestamp,
                    "exposure_source": record.exposure_source,
                    "generation_version": record.generation_version,
                    "generation_seed": record.generation_seed,
                }
                for record in events
            ],
        )
        events_created = len(events)
    else:
        for record in events:
            stored = stored_events[record.event_id]
            stored.event_type = record.event_type
            stored.user = stored_users[record.user_id]
            stored.product = stored_products[record.product_id]
            stored.session = stored_sessions[record.session_id]
            stored.occurred_at = record.timestamp
            stored.exposure_source = record.exposure_source
            stored.generation_version = record.generation_version
            stored.generation_seed = record.generation_seed
            events_updated += 1

    database.commit()
    return InteractionWriteResult(
        sessions_created=sessions_created,
        sessions_updated=sessions_updated,
        sessions_deleted=sessions_deleted,
        events_created=events_created,
        events_updated=events_updated,
        events_deleted=events_deleted,
    )

