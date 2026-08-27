from dataclasses import replace
from statistics import fmean, pstdev

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import Category, Product, ProductAttribute, ProductStatus, RealityType
from synthetic_data.config import ATTRIBUTE_NAMES, CATALOG_VERSION, PARENT_CATEGORY_TARGETS, TAG_VOCABULARY
from synthetic_data.database import write_catalog
from synthetic_data.product_generator import (
    CatalogValidationError,
    generate_catalog,
    summarize_catalog,
    validate_catalog,
)


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
def catalog():
    return generate_catalog(count=200, seed=42)


def test_fixed_seed_is_reproducible(catalog) -> None:
    assert catalog == generate_catalog(count=200, seed=42)
    assert catalog != generate_catalog(count=200, seed=43)


def test_catalog_has_200_unique_meaningful_products(catalog) -> None:
    names = [record.name for record in catalog]

    assert len(catalog) == 200
    assert len(names) == len(set(names))
    assert "Moon" in names
    assert "Time Machine" in names
    assert all(not name.startswith("Product ") for name in names)
    assert all(record.description.strip() for record in catalog)


def test_category_distribution_and_metadata_contract(catalog) -> None:
    summary = summarize_catalog(catalog)

    assert summary["category_distribution"] == PARENT_CATEGORY_TARGETS
    for record in catalog:
        assert 3 <= len(record.tags) <= 6
        assert set(record.tags) <= TAG_VOCABULARY
        assert set(record.attributes) == set(ATTRIBUTE_NAMES)
        assert all(0.0 <= value <= 1.0 for value in record.attributes.values())
        assert 0.0 <= record.rarity <= 1.0
        assert isinstance(record.reality_type, RealityType)
        assert isinstance(record.status, ProductStatus)
        assert record.price > 0
        assert record.catalog_version == CATALOG_VERSION


def test_rarity_and_novelty_are_related_but_not_identical(catalog) -> None:
    differences = [abs(record.rarity - record.attributes["novelty"]) for record in catalog]

    assert fmean(differences) > 0.03
    assert fmean(record.rarity for record in catalog) > 0.5


def test_category_prototypes_create_detectable_attribute_structure(catalog) -> None:
    planets = [record.attributes["space_affinity"] for record in catalog if record.category_slug == "planet"]
    artifacts = [
        record.attributes["space_affinity"]
        for record in catalog
        if record.category_slug == "historical-artifact"
    ]

    assert fmean(planets) - fmean(artifacts) > 0.75
    assert pstdev(planets) < 0.12


def test_validation_rejects_missing_common_attribute(catalog) -> None:
    invalid_attributes = dict(catalog[0].attributes)
    invalid_attributes.pop("power")
    invalid = [replace(catalog[0], attributes=invalid_attributes), *catalog[1:]]

    with pytest.raises(CatalogValidationError, match="common nine-axis schema"):
        validate_catalog(invalid, expected_count=200)


def test_database_write_is_idempotent_and_preserves_decimal_precision(
    database: Session,
    catalog,
) -> None:
    first = write_catalog(database, catalog)
    second = write_catalog(database, generate_catalog(count=200, seed=42))

    assert (first.created, first.updated, first.deleted) == (200, 0, 0)
    assert (second.created, second.updated, second.deleted) == (0, 200, 0)
    assert database.scalar(select(func.count()).select_from(Product)) == 200
    assert database.scalar(select(func.count()).select_from(ProductAttribute)) == 200 * 9

    expected_most_expensive = max(catalog, key=lambda record: record.price)
    stored = database.scalar(
        select(Product).where(Product.name == expected_most_expensive.name)
    )
    assert stored.price == expected_most_expensive.price
    assert len(stored.attribute_values) == 9
    assert stored.generation_seed == 42


def test_database_category_hierarchy_matches_generated_products(
    database: Session,
    catalog,
) -> None:
    write_catalog(database, catalog)
    moon = database.scalar(select(Product).where(Product.name == "Moon"))
    space = database.scalar(select(Category).where(Category.slug == "space"))

    assert moon.category.slug == "satellite"
    assert moon.category.parent is space


def test_different_seed_requires_explicit_replacement(database: Session, catalog) -> None:
    write_catalog(database, catalog)
    different_catalog = generate_catalog(count=200, seed=43)

    with pytest.raises(RuntimeError, match="--replace-existing"):
        write_catalog(database, different_catalog)

    result = write_catalog(database, different_catalog, replace_existing=True)
    assert result.deleted == 200
    assert result.created == 200
    assert database.scalar(select(func.count()).select_from(Product)) == 200
