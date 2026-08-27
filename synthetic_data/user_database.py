from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.models import Event, Session as BrowsingSession, SyntheticUserProfile, User
from synthetic_data.user_config import USER_GENERATION_VERSION
from synthetic_data.user_generator import SyntheticUserRecord, validate_users


@dataclass(frozen=True)
class UserWriteResult:
    created: int
    updated: int
    deleted: int


def _delete_existing_population(database: Session, profiles: list[SyntheticUserProfile]) -> int:
    if not profiles:
        return 0
    user_ids = [profile.user_id for profile in profiles]
    event_count = database.scalar(
        select(func.count()).select_from(Event).where(Event.user_id.in_(user_ids))
    )
    session_count = database.scalar(
        select(func.count()).select_from(BrowsingSession).where(BrowsingSession.user_id.in_(user_ids))
    )
    if event_count or session_count:
        raise RuntimeError(
            "Synthetic users already have Session or Event rows and cannot be safely replaced."
        )
    for profile in profiles:
        database.delete(profile.user)
    database.flush()
    return len(profiles)


def _apply_profile(profile: SyntheticUserProfile, record: SyntheticUserRecord) -> None:
    profile.archetype = record.archetype
    profile.secondary_archetype = record.secondary_archetype
    profile.secondary_archetype_weight = record.secondary_archetype_weight
    for name, value in record.preferences.items():
        setattr(profile, name, value)
    profile.budget_log10 = record.budget_log10
    profile.budget_tier = record.budget_tier
    profile.price_sensitivity = record.price_sensitivity
    profile.popularity_preference = record.popularity_preference
    profile.exploration_tendency = record.exploration_tendency
    profile.impulsiveness = record.impulsiveness
    profile.activity_level = record.activity_level
    profile.activity_tier = record.activity_tier
    profile.catalog_version = record.catalog_version
    profile.user_generation_version = record.user_generation_version
    profile.generation_seed = record.generation_seed
    profile.created_at = record.created_at


def write_users(
    database: Session,
    records: list[SyntheticUserRecord],
    *,
    replace_existing: bool = False,
) -> UserWriteResult:
    """Validate and upsert one deterministic synthetic user population."""
    if not records:
        raise ValueError("cannot write an empty synthetic user population")
    requested_seed = records[0].generation_seed
    validate_users(records, expected_count=len(records), expected_seed=requested_seed)
    requested_ids = {record.user_id for record in records}
    existing_profiles = list(
        database.scalars(
            select(SyntheticUserProfile).where(
                SyntheticUserProfile.user_generation_version == USER_GENERATION_VERSION
            )
        ).all()
    )
    existing_ids = {profile.user.external_id for profile in existing_profiles}
    existing_seeds = {profile.generation_seed for profile in existing_profiles}
    existing_catalog_versions = {profile.catalog_version for profile in existing_profiles}
    population_differs = existing_profiles and (
        existing_ids != requested_ids
        or existing_seeds != {requested_seed}
        or existing_catalog_versions != {records[0].catalog_version}
    )
    if population_differs and not replace_existing:
        raise RuntimeError(
            "A different synthetic_user_v1 population already exists. "
            "Re-run with --replace-existing to replace only that synthetic population."
        )

    deleted_count = (
        _delete_existing_population(database, existing_profiles) if replace_existing else 0
    )
    existing_users = {
        user.external_id: user
        for user in database.scalars(select(User).where(User.external_id.in_(requested_ids))).all()
    }
    created_count = 0
    updated_count = 0

    for record in records:
        user = existing_users.get(record.user_id)
        if user is None:
            user = User(external_id=record.user_id, created_at=record.created_at)
            user.synthetic_profile = SyntheticUserProfile()
            database.add(user)
            existing_users[record.user_id] = user
            created_count += 1
        elif user.synthetic_profile is None:
            raise RuntimeError(
                f"User identity {record.user_id} already exists without a synthetic profile."
            )
        elif user.synthetic_profile.user_generation_version != USER_GENERATION_VERSION:
            raise RuntimeError(
                f"User identity {record.user_id} belongs to another synthetic user version."
            )
        else:
            updated_count += 1

        user.created_at = record.created_at
        _apply_profile(user.synthetic_profile, record)

    database.commit()
    return UserWriteResult(created=created_count, updated=updated_count, deleted=deleted_count)
