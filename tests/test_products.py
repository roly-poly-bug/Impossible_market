import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.base import Base
from backend.app.db.models import Product
from backend.app.db.seed_products import PRODUCTS, seed_products
from backend.app.db.session import get_db
from backend.app.main import app


@pytest.fixture
def catalog() -> tuple[TestClient, Session]:
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


def test_product_list_succeeds(catalog: tuple[TestClient, Session]) -> None:
    client, database = catalog
    seed_products(database)

    response = client.get("/api/products")

    assert response.status_code == 200
    assert len(response.json()) == len(PRODUCTS)


def test_product_detail_succeeds(catalog: tuple[TestClient, Session]) -> None:
    client, database = catalog
    seed_products(database)
    product = database.scalar(select(Product).where(Product.name == "Moon"))

    response = client.get(f"/api/products/{product.id}")

    assert response.status_code == 200
    assert response.json()["name"] == "Moon"
    assert response.json()["price"] == "4800000000000000"


def test_missing_product_returns_404(catalog: tuple[TestClient, Session]) -> None:
    client, _ = catalog

    response = client.get("/api/products/999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Product not found"}


def test_seed_is_idempotent_and_products_are_retrievable(catalog: tuple[TestClient, Session]) -> None:
    client, database = catalog

    assert seed_products(database) == len(PRODUCTS)
    assert seed_products(database) == 0

    response = client.get("/api/products")
    products = response.json()
    assert len(products) == len(PRODUCTS)
    assert next(product for product in products if product["name"] == "Time Machine")["price"] == (
        "1234567890123456789012345678"
    )
