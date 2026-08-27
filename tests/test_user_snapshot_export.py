import csv
import json
from pathlib import Path

from synthetic_data.user_config import PREFERENCE_NAMES, USER_GENERATION_VERSION
from synthetic_data.user_export import USER_CSV_COLUMNS, export_users_csv, export_users_json
from synthetic_data.user_generator import generate_users


def test_user_csv_export_is_flat_and_complete(tmp_path) -> None:
    users = generate_users(count=1000, seed=42)
    output = export_users_csv(users, tmp_path / "users.csv")
    with output.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    assert len(rows) == 1000
    assert tuple(rows[0]) == USER_CSV_COLUMNS
    assert all(row[name] != "" for row in rows for name in PREFERENCE_NAMES)
    assert all(row["user_generation_version"] == USER_GENERATION_VERSION for row in rows)


def test_user_json_export_preserves_nested_ground_truth(tmp_path) -> None:
    users = generate_users(count=1000, seed=42)
    output = export_users_json(users, tmp_path / "users.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["metadata"]["count"] == 1000
    assert payload["metadata"]["generation_seed"] == 42
    assert payload["metadata"]["preference_names"] == list(PREFERENCE_NAMES)
    assert len(payload["users"]) == 1000
    assert set(payload["users"][0]["preferences"]) == set(PREFERENCE_NAMES)


def test_user_exports_are_byte_reproducible(tmp_path) -> None:
    first = generate_users(count=1000, seed=42)
    second = generate_users(count=1000, seed=42)
    first_csv = export_users_csv(first, tmp_path / "first.csv")
    second_csv = export_users_csv(second, tmp_path / "second.csv")
    first_json = export_users_json(first, tmp_path / "first.json")
    second_json = export_users_json(second, tmp_path / "second.json")

    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()


def test_tracked_user_snapshot_matches_generator(tmp_path) -> None:
    users = generate_users(count=1000, seed=42)
    generated_csv = export_users_csv(users, tmp_path / "users.csv")
    generated_json = export_users_json(users, tmp_path / "users.json")
    repository_root = Path(__file__).resolve().parents[1]

    assert generated_csv.read_bytes() == (
        repository_root / "data" / "synthetic_user_v1_seed42.csv"
    ).read_bytes()
    assert generated_json.read_bytes() == (
        repository_root / "data" / "synthetic_user_v1_seed42.json"
    ).read_bytes()
