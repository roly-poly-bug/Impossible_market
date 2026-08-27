import csv
import json
from decimal import Decimal
from pathlib import Path

from synthetic_data.config import ATTRIBUTE_NAMES, CATALOG_VERSION, TAG_VOCABULARY
from synthetic_data.export import CSV_COLUMNS, export_catalog_csv, export_catalog_json
from synthetic_data.product_generator import generate_catalog


def test_csv_export_is_flat_complete_and_preserves_price(tmp_path) -> None:
    catalog = generate_catalog(count=200, seed=42)
    output = export_catalog_csv(catalog, tmp_path / "catalog.csv")

    with output.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    assert len(rows) == 200
    assert tuple(rows[0]) == CSV_COLUMNS
    assert all(row[attribute] != "" for row in rows for attribute in ATTRIBUTE_NAMES)
    assert all(3 <= len(row["tags"].split("|")) <= 6 for row in rows)
    assert all(set(row["tags"].split("|")) <= TAG_VOCABULARY for row in rows)
    assert [Decimal(row["price"]) for row in rows] == [record.price for record in catalog]


def test_json_export_preserves_metadata_and_nested_fields(tmp_path) -> None:
    catalog = generate_catalog(count=200, seed=42)
    output = export_catalog_json(catalog, tmp_path / "catalog.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["metadata"] == {
        "catalog_version": CATALOG_VERSION,
        "generation_seed": 42,
        "count": 200,
        "attribute_names": list(ATTRIBUTE_NAMES),
    }
    assert len(payload["products"]) == 200
    for product, record in zip(payload["products"], catalog, strict=True):
        assert product["category"]["slug"] == record.category_slug
        assert product["tags"] == list(record.tags)
        assert Decimal(product["price"]) == record.price
        assert set(product["attributes"]) == set(ATTRIBUTE_NAMES)


def test_snapshot_exports_are_byte_reproducible(tmp_path) -> None:
    first_catalog = generate_catalog(count=200, seed=42)
    second_catalog = generate_catalog(count=200, seed=42)

    first_csv = export_catalog_csv(first_catalog, tmp_path / "first.csv")
    second_csv = export_catalog_csv(second_catalog, tmp_path / "second.csv")
    first_json = export_catalog_json(first_catalog, tmp_path / "first.json")
    second_json = export_catalog_json(second_catalog, tmp_path / "second.json")

    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()


def test_tracked_seed42_snapshot_matches_generator(tmp_path) -> None:
    catalog = generate_catalog(count=200, seed=42)
    generated_csv = export_catalog_csv(catalog, tmp_path / "catalog.csv")
    generated_json = export_catalog_json(catalog, tmp_path / "catalog.json")
    repository_root = Path(__file__).resolve().parents[1]

    assert generated_csv.read_bytes() == (
        repository_root / "data" / "synthetic_product_v1_seed42.csv"
    ).read_bytes()
    assert generated_json.read_bytes() == (
        repository_root / "data" / "synthetic_product_v1_seed42.json"
    ).read_bytes()
