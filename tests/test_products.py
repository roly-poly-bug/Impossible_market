from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import Category, Product, ProductAttribute, Tag, product_tags
from backend.app.db.seed_products import CATEGORIES, PRODUCTS, TAG_NAMES, seed_products
from backend.app.db.session import get_db
from backend.app.main import app


@pytest.fixture
def catalog() -> Generator[tuple[TestClient, Session], None, None]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    database = Session(engine)

    def override_get_db():
        yield database

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app), database
    finally:
        app.dependency_overrides.clear()
        database.close()
        engine.dispose()


def test_product_list_succeeds_with_lightweight_category(catalog: tuple[TestClient, Session]) -> None:
    client, database = catalog
    seed_products(database)

    response = client.get("/api/products")

    assert response.status_code == 200
    assert len(response.json()) == len(PRODUCTS)
    assert response.json()[0]["category"] == "Satellite"
    assert response.json()[0]["status"] == "available"
    assert "tags" not in response.json()[0]
    assert "attributes" not in response.json()[0]


def test_product_detail_contains_metadata(catalog: tuple[TestClient, Session]) -> None:
    client, database = catalog
    seed_products(database)
    moon = database.scalar(select(Product).where(Product.name == "Moon"))

    response = client.get(f"/api/products/{moon.id}")
    body = response.json()

    assert response.status_code == 200
    assert body["name"] == "Moon"
    assert body["category"] == {"id": moon.category.id, "name": "Satellite", "slug": "satellite"}
    assert {tag["slug"] for tag in body["tags"]} == {"space", "natural", "historic", "exclusive"}
    assert body["attributes"]["space_affinity"] == 1.0
    assert body["attributes"]["danger"] == 0.65
    assert body["status"] == "available"


def test_category_hierarchy_and_product_relation(catalog: tuple[TestClient, Session]) -> None:
    _, database = catalog
    seed_products(database)

    moon = database.scalar(select(Product).where(Product.name == "Moon"))
    satellite = database.scalar(select(Category).where(Category.slug == "satellite"))

    assert moon.category is satellite
    assert satellite.parent.slug == "space"
    assert moon in satellite.products


def test_product_tag_many_to_many_relation(catalog: tuple[TestClient, Session]) -> None:
    _, database = catalog
    seed_products(database)

    tyrannosaurus = database.scalar(select(Product).where(Product.name == "Tyrannosaurus Rex"))
    dangerous = database.scalar(select(Tag).where(Tag.slug == "dangerous"))

    assert dangerous in tyrannosaurus.tags
    assert tyrannosaurus in dangerous.products


def test_product_attributes_are_retrievable(catalog: tuple[TestClient, Session]) -> None:
    _, database = catalog
    seed_products(database)

    time_machine = database.scalar(select(Product).where(Product.name == "Time Machine"))

    assert time_machine.attribute_values["technology_level"] == 1.0
    assert time_machine.attribute_values["novelty"] == 1.0


def test_missing_product_returns_404(catalog: tuple[TestClient, Session]) -> None:
    client, _ = catalog

    response = client.get("/api/products/999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Product not found"}


def test_seed_is_idempotent_across_metadata_tables(catalog: tuple[TestClient, Session]) -> None:
    client, database = catalog

    assert seed_products(database) == len(PRODUCTS)
    assert seed_products(database) == 0

    assert database.scalar(select(func.count()).select_from(Product)) == len(PRODUCTS)
    assert database.scalar(select(func.count()).select_from(Category)) == len(CATEGORIES)
    assert database.scalar(select(func.count()).select_from(Tag)) == len(TAG_NAMES)
    assert database.scalar(select(func.count()).select_from(ProductAttribute)) == sum(
        len(product["attributes"]) for product in PRODUCTS
    )
    assert database.scalar(select(func.count()).select_from(product_tags)) == sum(
        len(product["tags"]) for product in PRODUCTS
    )

    products = client.get("/api/products").json()
    assert next(product for product in products if product["name"] == "Time Machine")["price"] == (
        "1234567890123456789012345678"
    )


def test_product_attribute_rejects_values_outside_normalized_range(
    catalog: tuple[TestClient, Session],
) -> None:
    _, database = catalog
    seed_products(database)
    moon = database.scalar(select(Product).where(Product.name == "Moon"))
    database.add(
        ProductAttribute(product=moon, attribute_name="invalid_score", numeric_value=1.5)
    )

    with pytest.raises(IntegrityError):
        database.commit()
