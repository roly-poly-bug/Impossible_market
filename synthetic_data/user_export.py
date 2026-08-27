from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from synthetic_data.user_config import PREFERENCE_NAMES
from synthetic_data.user_generator import SyntheticUserRecord


USER_CSV_COLUMNS = (
    "user_id",
    "archetype",
    "secondary_archetype",
    "secondary_archetype_weight",
    *PREFERENCE_NAMES,
    "budget_log10",
    "budget_tier",
    "price_sensitivity",
    "popularity_preference",
    "exploration_tendency",
    "impulsiveness",
    "activity_level",
    "activity_tier",
    "catalog_version",
    "user_generation_version",
    "generation_seed",
    "created_at",
)


def _created_at(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _csv_row(record: SyntheticUserRecord) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "user_id": record.user_id,
        "archetype": record.archetype,
        "secondary_archetype": record.secondary_archetype or "",
        "secondary_archetype_weight": record.secondary_archetype_weight,
        "budget_log10": record.budget_log10,
        "budget_tier": record.budget_tier.value,
        "price_sensitivity": record.price_sensitivity,
        "popularity_preference": record.popularity_preference,
        "exploration_tendency": record.exploration_tendency,
        "impulsiveness": record.impulsiveness,
        "activity_level": record.activity_level,
        "activity_tier": record.activity_tier.value,
        "catalog_version": record.catalog_version,
        "user_generation_version": record.user_generation_version,
        "generation_seed": record.generation_seed,
        "created_at": _created_at(record.created_at),
    }
    row.update(record.preferences)
    return row


def _json_user(record: SyntheticUserRecord) -> dict[str, Any]:
    return {
        "user_id": record.user_id,
        "archetype": {
            "primary": record.archetype,
            "secondary": record.secondary_archetype,
            "secondary_weight": record.secondary_archetype_weight,
        },
        "preferences": dict(record.preferences),
        "budget": {
            "log10": record.budget_log10,
            "tier": record.budget_tier.value,
        },
        "behavior": {
            "price_sensitivity": record.price_sensitivity,
            "popularity_preference": record.popularity_preference,
            "exploration_tendency": record.exploration_tendency,
            "impulsiveness": record.impulsiveness,
            "activity_level": record.activity_level,
            "activity_tier": record.activity_tier.value,
        },
        "provenance": {
            "catalog_version": record.catalog_version,
            "user_generation_version": record.user_generation_version,
            "generation_seed": record.generation_seed,
            "created_at": _created_at(record.created_at),
        },
    }


def export_users_csv(records: Iterable[SyntheticUserRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=USER_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)
    return destination


def export_users_json(records: Iterable[SyntheticUserRecord], path: str | Path) -> Path:
    user_records = list(records)
    if not user_records:
        raise ValueError("cannot export an empty user population")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "user_generation_version": user_records[0].user_generation_version,
            "catalog_version": user_records[0].catalog_version,
            "generation_seed": user_records[0].generation_seed,
            "count": len(user_records),
            "preference_names": list(PREFERENCE_NAMES),
        },
        "users": [_json_user(record) for record in user_records],
    }
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return destination

