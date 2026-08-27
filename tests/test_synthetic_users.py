from dataclasses import replace
from statistics import fmean

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import (
    ActivityTier,
    BudgetTier,
    Session as BrowsingSession,
    SyntheticUserProfile,
    User,
)
from synthetic_data.user_archetypes import ARCHETYPE_BY_NAME, USER_ARCHETYPES
from synthetic_data.user_config import (
    PREFERENCE_NAMES,
    PRODUCT_CATALOG_VERSION,
    USER_GENERATION_VERSION,
)
from synthetic_data.user_database import write_users
from synthetic_data.user_generator import UserValidationError, generate_users, validate_users


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


@pytest.fixture(scope="module")
def users():
    return generate_users(count=1000, seed=42)


def test_requested_count_seed_and_identity_are_reproducible(users) -> None:
    assert users == generate_users(count=1000, seed=42)
    assert users != generate_users(count=1000, seed=43)
    assert len(users) == 1000
    assert len({record.user_id for record in users}) == 1000


def test_preferences_archetypes_and_generation_metadata_are_valid(users) -> None:
    for record in users:
        assert record.archetype in ARCHETYPE_BY_NAME
        assert set(record.preferences) == set(PREFERENCE_NAMES)
        assert all(0.0 <= value <= 1.0 for value in record.preferences.values())
        assert record.catalog_version == PRODUCT_CATALOG_VERSION
        assert record.user_generation_version == USER_GENERATION_VERSION
        assert record.generation_seed == 42


def test_budget_and_behavioral_parameters_are_valid(users) -> None:
    for record in users:
        assert 5.0 <= record.budget_log10 <= 30.0
        assert isinstance(record.budget_tier, BudgetTier)
        assert isinstance(record.activity_tier, ActivityTier)
        for value in (
            record.price_sensitivity,
            record.popularity_preference,
            record.exploration_tendency,
            record.impulsiveness,
            record.activity_level,
        ):
            assert 0.0 <= value <= 1.0


def test_archetype_ground_truth_structure_is_detectable_but_overlapping(users) -> None:
    space_users = [record for record in users if record.archetype == "Space Enthusiast"]
    history_users = [record for record in users if record.archetype == "History Collector"]
    nature_users = [record for record in users if record.archetype == "Nature Explorer"]

    assert fmean(record.preferences["space_preference"] for record in space_users) > 0.75
    assert fmean(record.preferences["historical_preference"] for record in history_users) > 0.75
    assert fmean(record.preferences["nature_preference"] for record in nature_users) > 0.75
    assert sum(record.secondary_archetype is not None for record in users) >= 250

    nearest_prototype_correct = 0
    for record in users:
        predicted = min(
            USER_ARCHETYPES,
            key=lambda archetype: sum(
                (
                    record.preferences[name]
                    - archetype.preference_prototype[name]
                )
                ** 2
                for name in PREFERENCE_NAMES
            ),
        )
        nearest_prototype_correct += predicted.name == record.archetype
    accuracy = nearest_prototype_correct / len(users)
    assert 0.50 < accuracy < 0.90


def test_validation_rejects_missing_preference(users) -> None:
    preferences = dict(users[0].preferences)
    preferences.pop("power_preference")
    invalid = [replace(users[0], preferences=preferences), *users[1:]]

    with pytest.raises(UserValidationError, match="common nine-axis schema"):
        validate_users(invalid, expected_count=1000, expected_seed=42)


def test_database_write_is_idempotent_and_tracks_version(database: Session, users) -> None:
    first = write_users(database, users)
    second = write_users(database, generate_users(count=1000, seed=42))

    assert (first.created, first.updated, first.deleted) == (1000, 0, 0)
    assert (second.created, second.updated, second.deleted) == (0, 1000, 0)
    assert database.scalar(select(func.count()).select_from(User)) == 1000
    assert database.scalar(select(func.count()).select_from(SyntheticUserProfile)) == 1000
    profile = database.scalar(select(SyntheticUserProfile).limit(1))
    assert profile.user_generation_version == USER_GENERATION_VERSION
    assert set(profile.preference_values) == set(PREFERENCE_NAMES)


def test_different_population_requires_explicit_replacement(database: Session, users) -> None:
    write_users(database, users)
    different_users = generate_users(count=1000, seed=43)

    with pytest.raises(RuntimeError, match="--replace-existing"):
        write_users(database, different_users)

    result = write_users(database, different_users, replace_existing=True)
    assert (result.created, result.updated, result.deleted) == (1000, 0, 1000)
    assert database.scalar(select(func.count()).select_from(User)) == 1000


def test_replacement_refuses_users_with_downstream_sessions(database: Session) -> None:
    users = generate_users(count=10, seed=42)
    write_users(database, users)
    first_user = database.scalar(select(User).where(User.external_id == users[0].user_id))
    database.add(BrowsingSession(session_key="replacement-safety-check", user=first_user))
    database.commit()

    with pytest.raises(RuntimeError, match="Session or Event"):
        write_users(
            database,
            generate_users(count=10, seed=43),
            replace_existing=True,
        )
