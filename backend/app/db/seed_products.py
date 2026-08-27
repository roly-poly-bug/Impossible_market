from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Product
from backend.app.db.session import SessionLocal, init_database


PRODUCTS = (
    {
        "name": "Moon",
        "description": "Earth's only natural satellite, offered with tides and phases included.",
        "category": "Celestial Body",
        "price": Decimal("4800000000000000"),
        "rarity": 1.0,
        "image_url": None,
    },
    {
        "name": "Mars",
        "description": "A rust-colored planet with excellent views and a demanding commute.",
        "category": "Celestial Body",
        "price": Decimal("2100000000000000000000000"),
        "rarity": 1.0,
        "image_url": None,
    },
    {
        "name": "Pacific Ocean",
        "description": "The world's largest ocean. Marine life and international boundaries sold separately.",
        "category": "Natural Wonder",
        "price": Decimal("990000000000000000"),
        "rarity": 0.99,
        "image_url": None,
    },
    {
        "name": "Tyrannosaurus Rex",
        "description": "One responsibly reconstructed apex predator. Requires a very sturdy enclosure.",
        "category": "Extinct Creature",
        "price": Decimal("875000000000"),
        "rarity": 0.97,
        "image_url": None,
    },
    {
        "name": "Time Machine",
        "description": "A prototype temporal vehicle. Warranty becomes complicated before purchase.",
        "category": "Impossible Technology",
        "price": Decimal("1234567890123456789012345678"),
        "rarity": 1.0,
        "image_url": None,
    },
    {
        "name": "Roman Empire",
        "description": "A vast historical civilization supplied as-is, including roads and administrative overhead.",
        "category": "Historical Era",
        "price": Decimal("440000000000000000000"),
        "rarity": 0.96,
        "image_url": None,
    },
    {
        "name": "International Space Station",
        "description": "A legendary orbital laboratory with panoramic windows and zero-gravity delivery.",
        "category": "Spacecraft",
        "price": Decimal("150000000000000"),
        "rarity": 0.995,
        "image_url": None,
    },
)


def seed_products(database: Session) -> int:
    """Insert missing catalog products and return the number added."""
    existing_names = set(database.scalars(select(Product.name)).all())
    new_products = [Product(**product) for product in PRODUCTS if product["name"] not in existing_names]
    database.add_all(new_products)
    database.commit()
    return len(new_products)


def main() -> None:
    init_database()
    with SessionLocal() as database:
        created_count = seed_products(database)
    print(f"Product seed complete: {created_count} product(s) added.")


if __name__ == "__main__":
    main()
