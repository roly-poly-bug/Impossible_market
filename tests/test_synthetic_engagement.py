import csv
from collections import Counter, defaultdict
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import Event, EventType, ProductStatus
from synthetic_data.database import write_catalog
from synthetic_data.engagement_config import ENGAGEMENT_GENERATION_VERSION
from synthetic_data.engagement_database import write_engagement_events
from synthetic_data.engagement_export import export_engagement_csv
from synthetic_data.engagement_generator import generate_engagement_events
from synthetic_data.engagement_quality import analyze_engagement
from synthetic_data.engagement_validation import (
    EngagementValidationError,
    validate_engagement_events,
)
from synthetic_data.event_generator import generate_events
from synthetic_data.interaction_database import write_interactions
from synthetic_data.product_generator import generate_catalog
from synthetic_data.session_generator import generate_sessions
from synthetic_data.user_database import write_users
from synthetic_data.user_generator import generate_users


@pytest.fixture(scope="module")
def engagement_world():
    products = generate_catalog(count=200, seed=42)
    users = generate_users(count=1000, seed=42)
    window, sessions = generate_sessions(users, seed=42)
    base_events = generate_events(sessions, users, products, seed=42)
    engagement_events = generate_engagement_events(
        sessions, base_events, users, products, seed=42
    )
    return products, users, window, sessions, base_events, engagement_events


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


def test_engagement_generation_is_deterministic_and_seeded(engagement_world) -> None:
    products, users, _, sessions, base_events, events = engagement_world
    same = generate_engagement_events(sessions, base_events, users, products, seed=42)
    different = generate_engagement_events(sessions, base_events, users, products, seed=43)

    assert events == same
    assert events != different
    assert all(event.generation_version == ENGAGEMENT_GENERATION_VERSION for event in events)


def test_engagement_events_validate_prior_views_status_and_uniqueness(
    engagement_world,
) -> None:
    products, users, window, sessions, base_events, events = engagement_world
    validate_engagement_events(
        events, base_events, sessions, users, products, window, expected_seed=42
    )
    products_by_id = {product.name: product for product in products}
    valid_types = {EventType.FAVORITE, EventType.ADD_TO_CART, EventType.PURCHASE}

    assert {event.event_type for event in events} == valid_types
    assert len({event.event_id for event in events}) == len(events)
    assert len({(event.user_id, event.product_id, event.event_type) for event in events}) == len(
        events
    )
    assert all(
        products_by_id[event.product_id].status == ProductStatus.AVAILABLE
        for event in events
        if event.event_type in (EventType.ADD_TO_CART, EventType.PURCHASE)
    )


def test_combined_timestamps_and_delayed_conversions_are_valid(engagement_world) -> None:
    _, _, window, sessions, base_events, events = engagement_world
    combined = defaultdict(list)
    for event in [*base_events, *events]:
        combined[event.session_id].append(event.timestamp)

    assert all(
        left < right
        for timestamps in combined.values()
        for left, right in zip(sorted(timestamps), sorted(timestamps)[1:], strict=False)
    )
    delayed = [event for event in events if event.conversion_timing == "later_session"]
    assert delayed
    assert all(window.start <= event.timestamp < window.end_exclusive for event in delayed)
    assert any(
        event.event_type == EventType.PURCHASE
        and (event.had_favorite_before or event.had_cart_before)
        for event in delayed
    )


def test_funnel_price_impulsiveness_and_paths_are_sane(engagement_world) -> None:
    products, users, _, _, base_events, events = engagement_world
    analysis = analyze_engagement(base_events, events, users, products)
    conversion = analysis["conversion"]

    assert 0.08 <= conversion["view_to_favorite"] <= 0.16
    assert 0.04 <= conversion["view_to_cart"] <= 0.10
    assert 0.01 <= conversion["view_to_purchase"] <= 0.045
    assert 0.20 <= conversion["cart_to_purchase"] <= 0.50
    assert analysis["price_signal"]["over"]["purchase"] > 0
    assert (
        analysis["price_signal"]["within"]["purchase_rate"]
        > analysis["price_signal"]["over"]["purchase_rate"]
    )
    impulse = analysis["impulsiveness_signal"]
    assert impulse["high"]["direct_purchase_rate"] > impulse["low"]["direct_purchase_rate"]
    assert (
        impulse["high"]["over_budget_purchase_rate"]
        > impulse["low"]["over_budget_purchase_rate"]
    )
    assert len(analysis["paths"]) >= 6


def test_preference_signal_increases_without_easy_single_feature_prediction(
    engagement_world,
) -> None:
    products, users, _, _, base_events, events = engagement_world
    analysis = analyze_engagement(base_events, events, users, products)
    means = analysis["preference_means"]

    assert means["impression"] < means["view"] < means["favorite"]
    assert means["view"] < means["add_to_cart"]
    assert means["view"] < means["purchase"]
    assert max(analysis["single_feature_auc"].values()) < 0.75


def test_validation_rejects_missing_view_and_duplicate_purchase(engagement_world) -> None:
    products, users, window, sessions, base_events, events = engagement_world
    first = events[0]
    missing_view = replace(first, source_view_event_id="missing-view")
    with pytest.raises(EngagementValidationError, match="prior View"):
        validate_engagement_events(
            [missing_view, *events[1:]],
            base_events,
            sessions,
            users,
            products,
            window,
            expected_seed=42,
        )

    purchase = next(event for event in events if event.event_type == EventType.PURCHASE)
    duplicate = replace(purchase, event_id="duplicate-purchase")
    with pytest.raises(EngagementValidationError, match="duplicate user-product"):
        validate_engagement_events(
            [*events, duplicate],
            base_events,
            sessions,
            users,
            products,
            window,
            expected_seed=42,
        )


def test_engagement_csv_is_reproducible_and_pandas_friendly(
    tmp_path, engagement_world
) -> None:
    _, _, _, _, _, events = engagement_world
    first = export_engagement_csv(events, tmp_path / "first.csv")
    second = export_engagement_csv(events, tmp_path / "second.csv")

    assert first.read_bytes() == second.read_bytes()
    with first.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == len(events)
    assert {"user_id", "product_id", "session_id", "event_type", "timestamp"} <= set(
        rows[0]
    )
    assert {row["event_type"] for row in rows} == {
        "favorite",
        "add_to_cart",
        "purchase",
    }


def test_engagement_database_write_is_idempotent(database, engagement_world) -> None:
    products, users, window, sessions, base_events, events = engagement_world
    selected_user = next(
        event.user_id for event in events if event.event_type == EventType.PURCHASE
    )
    selected_sessions = [session for session in sessions if session.user_id == selected_user]
    session_ids = {session.session_id for session in selected_sessions}
    selected_base = [event for event in base_events if event.session_id in session_ids]
    selected_engagement = [event for event in events if event.user_id == selected_user]
    write_catalog(database, products)
    write_users(database, users)
    write_interactions(
        database,
        selected_sessions,
        selected_base,
        users,
        products,
        window,
    )

    first = write_engagement_events(
        database,
        selected_engagement,
        selected_base,
        selected_sessions,
        users,
        products,
        window,
    )
    second = write_engagement_events(
        database,
        selected_engagement,
        selected_base,
        selected_sessions,
        users,
        products,
        window,
    )

    assert first.events_created == len(selected_engagement)
    assert second.events_updated == len(selected_engagement)
    stored_count = database.scalar(
        select(func.count()).select_from(Event).where(
            Event.generation_version == ENGAGEMENT_GENERATION_VERSION
        )
    )
    assert stored_count == len(selected_engagement)
