from collections import defaultdict
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import Event, EventType, Session as BrowsingSession
from synthetic_data.database import write_catalog
from synthetic_data.event_generator import generate_events
from synthetic_data.interaction_config import INTERACTION_GENERATION_VERSION
from synthetic_data.interaction_database import write_interactions
from synthetic_data.interaction_export import export_events_csv, export_sessions_csv
from synthetic_data.interaction_validation import InteractionValidationError, validate_interactions
from synthetic_data.product_generator import generate_catalog
from synthetic_data.session_generator import generate_sessions
from synthetic_data.user_database import write_users
from synthetic_data.user_generator import generate_users


@pytest.fixture(scope="module")
def interaction_world():
    products = generate_catalog(count=200, seed=42)
    users = generate_users(count=1000, seed=42)
    window, sessions = generate_sessions(users, seed=42)
    events = generate_events(sessions, users, products, seed=42)
    return products, users, window, sessions, events


@pytest.fixture
def database() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_interaction_seed_is_reproducible(interaction_world) -> None:
    products, users, _, sessions, events = interaction_world
    second_window, second_sessions = generate_sessions(users, seed=42)
    second_events = generate_events(second_sessions, users, products, seed=42)
    _, different_sessions = generate_sessions(users, seed=43)

    assert sessions == second_sessions
    assert events == second_events
    assert sessions != different_sessions
    assert second_window.start == interaction_world[2].start


def test_ids_foreign_keys_versions_and_event_types_are_valid(interaction_world) -> None:
    products, users, window, sessions, events = interaction_world
    validate_interactions(sessions, events, users, products, window, expected_seed=42)

    assert len({session.session_id for session in sessions}) == len(sessions)
    assert len({event.event_id for event in events}) == len(events)
    assert {event.event_type for event in events} == {EventType.IMPRESSION, EventType.VIEW}
    assert all(session.generation_version == INTERACTION_GENERATION_VERSION for session in sessions)
    assert all(event.generation_version == INTERACTION_GENERATION_VERSION for event in events)


def test_session_and_event_timestamps_are_ordered(interaction_world) -> None:
    _, _, _, sessions, events = interaction_world
    events_by_session = defaultdict(list)
    for event in events:
        events_by_session[event.session_id].append(event)

    for session in sessions:
        session_events = events_by_session[session.session_id]
        assert session.started_at <= session_events[0].timestamp
        assert session_events[-1].timestamp <= session.ended_at
        assert all(
            left.timestamp < right.timestamp
            for left, right in zip(session_events, session_events[1:], strict=False)
        )


def test_every_view_has_prior_impression_and_no_duplicate_impression(interaction_world) -> None:
    _, _, _, sessions, events = interaction_world
    events_by_session = defaultdict(list)
    for event in events:
        events_by_session[event.session_id].append(event)

    for session in sessions:
        impressed = set()
        for event in events_by_session[session.session_id]:
            if event.event_type == EventType.IMPRESSION:
                assert event.product_id not in impressed
                impressed.add(event.product_id)
            else:
                assert event.product_id in impressed


def test_activity_creates_overlapping_session_heterogeneity(interaction_world) -> None:
    _, users, _, sessions, _ = interaction_world
    counts = defaultdict(int)
    for session in sessions:
        counts[session.user_id] += 1
    by_tier = defaultdict(list)
    for user in users:
        by_tier[user.activity_tier.value].append(counts[user.user_id])

    casual_mean = sum(by_tier["casual"]) / len(by_tier["casual"])
    heavy_mean = sum(by_tier["heavy"]) / len(by_tier["heavy"])
    assert heavy_mean > casual_mean + 2.0
    assert max(by_tier["casual"]) >= min(by_tier["heavy"])


def test_exposure_view_preference_and_price_signals_are_sane(interaction_world) -> None:
    _, _, _, _, events = interaction_world
    impressions = [event for event in events if event.event_type == EventType.IMPRESSION]
    views = [event for event in events if event.event_type == EventType.VIEW]
    viewed = {(event.session_id, event.product_id) for event in views}

    assert {event.exposure_source for event in impressions} == {
        "preference",
        "popular",
        "exploration",
        "random",
    }
    view_rate = len(views) / len(impressions)
    assert 0.10 <= view_rate <= 0.35
    viewed_match = [
        event.preference_match
        for event in impressions
        if (event.session_id, event.product_id) in viewed
    ]
    non_viewed_match = [
        event.preference_match
        for event in impressions
        if (event.session_id, event.product_id) not in viewed
    ]
    assert sum(viewed_match) / len(viewed_match) > sum(non_viewed_match) / len(non_viewed_match)
    assert any(event.over_budget for event in views)


def test_validation_rejects_view_without_impression_and_invalid_type(interaction_world) -> None:
    products, users, window, sessions, events = interaction_world
    first_impression = next(event for event in events if event.event_type == EventType.IMPRESSION)
    invalid_view = replace(first_impression, event_type=EventType.VIEW)
    with pytest.raises(InteractionValidationError, match="View requires"):
        validate_interactions(
            sessions,
            [invalid_view, *events[1:]],
            users,
            products,
            window,
            expected_seed=42,
        )

    invalid_type = replace(first_impression, event_type=EventType.FAVORITE)
    with pytest.raises(InteractionValidationError, match="invalid v1 event type"):
        validate_interactions(
            sessions,
            [invalid_type, *events[1:]],
            users,
            products,
            window,
            expected_seed=42,
        )


def test_interaction_csv_exports_are_reproducible(tmp_path, interaction_world) -> None:
    _, _, _, sessions, events = interaction_world
    first_sessions = export_sessions_csv(sessions, tmp_path / "sessions-first.csv")
    second_sessions = export_sessions_csv(sessions, tmp_path / "sessions-second.csv")
    first_events = export_events_csv(events, tmp_path / "events-first.csv")
    second_events = export_events_csv(events, tmp_path / "events-second.csv")

    assert first_sessions.read_bytes() == second_sessions.read_bytes()
    assert first_events.read_bytes() == second_events.read_bytes()
    assert len(first_sessions.read_text(encoding="utf-8").splitlines()) == len(sessions) + 1
    assert len(first_events.read_text(encoding="utf-8").splitlines()) == len(events) + 1


def test_database_write_is_idempotent(database: Session, interaction_world) -> None:
    products, users, window, sessions, events = interaction_world
    selected_sessions = sessions[:12]
    selected_ids = {session.session_id for session in selected_sessions}
    selected_events = [event for event in events if event.session_id in selected_ids]
    write_catalog(database, products)
    write_users(database, users)

    first = write_interactions(
        database,
        selected_sessions,
        selected_events,
        users,
        products,
        window,
    )
    second = write_interactions(
        database,
        selected_sessions,
        selected_events,
        users,
        products,
        window,
    )

    assert first.sessions_created == len(selected_sessions)
    assert first.events_created == len(selected_events)
    assert second.sessions_updated == len(selected_sessions)
    assert second.events_updated == len(selected_events)
    assert database.scalar(select(func.count()).select_from(BrowsingSession)) == len(
        selected_sessions
    )
    assert database.scalar(select(func.count()).select_from(Event)) == len(selected_events)
