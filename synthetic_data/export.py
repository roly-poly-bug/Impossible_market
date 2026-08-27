from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from synthetic_data.config import ATTRIBUTE_NAMES
from synthetic_data.product_generator import SyntheticProductRecord


CSV_COLUMNS = (
    "name",
    "description",
    "category_parent",
    "category_name",
    "category_slug",
    "tags",
    "price",
    "rarity",
    "reality_type",
    "status",
    "catalog_version",
    "generation_seed",
    *ATTRIBUTE_NAMES,
)


def _csv_row(record: SyntheticProductRecord) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "name": record.name,
        "description": record.description,
        "category_parent": record.category_parent,
        "category_name": record.category_name,
        "category_slug": record.category_slug,
        "tags": "|".join(record.tags),
        "price": str(record.price),
        "rarity": record.rarity,
        "reality_type": record.reality_type.value,
        "status": record.status.value,
        "catalog_version": record.catalog_version,
        "generation_seed": record.generation_seed,
    }
    row.update(record.attributes)
    return row


def _json_product(record: SyntheticProductRecord) -> dict[str, Any]:
    return {
        "name": record.name,
        "description": record.description,
        "category": {
            "parent": record.category_parent,
            "name": record.category_name,
            "slug": record.category_slug,
        },
        "tags": list(record.tags),
        "price": str(record.price),
        "rarity": record.rarity,
        "reality_type": record.reality_type.value,
        "status": record.status.value,
        "catalog_version": record.catalog_version,
        "generation_seed": record.generation_seed,
        "attributes": dict(record.attributes),
    }


def export_catalog_csv(records: Iterable[SyntheticProductRecord], path: str | Path) -> Path:
    """Write a deterministic, pandas-friendly flat catalog snapshot."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)
    return destination


def export_catalog_json(records: Iterable[SyntheticProductRecord], path: str | Path) -> Path:
    """Write a deterministic structured catalog snapshot with top-level metadata."""
    product_records = list(records)
    if not product_records:
        raise ValueError("cannot export an empty catalog")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "catalog_version": product_records[0].catalog_version,
            "generation_seed": product_records[0].generation_seed,
            "count": len(product_records),
            "attribute_names": list(ATTRIBUTE_NAMES),
        },
        "products": [_json_product(record) for record in product_records],
    }
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return destination
