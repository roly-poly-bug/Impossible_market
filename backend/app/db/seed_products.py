from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import (
    Category,
    Product,
    ProductAttribute,
    ProductStatus,
    RealityType,
    Tag,
)
from backend.app.db.session import SessionLocal, backup_legacy_database, init_database


CATEGORIES = (
    {"name": "Space", "slug": "space", "parent": None},
    {"name": "Natural World", "slug": "natural-world", "parent": None},
    {"name": "History", "slug": "history", "parent": None},
    {"name": "Technology", "slug": "technology", "parent": None},
    {"name": "Planet", "slug": "planet", "parent": "space"},
    {"name": "Satellite", "slug": "satellite", "parent": "space"},
    {"name": "Spacecraft", "slug": "spacecraft", "parent": "space"},
    {"name": "Natural Wonder", "slug": "natural-wonder", "parent": "natural-world"},
    {"name": "Prehistoric Creature", "slug": "prehistoric-creature", "parent": "natural-world"},
    {"name": "Empire", "slug": "empire", "parent": "history"},
    {"name": "Impossible Technology", "slug": "impossible-technology", "parent": "technology"},
)

TAG_NAMES = (
    "creature",
    "dangerous",
    "exclusive",
    "historic",
    "luxury",
    "mysterious",
    "natural",
    "prehistoric",
    "rare",
    "space",
    "technology",
)

PRODUCTS = (
    {
        "name": "Moon",
        "description": "Earth's only natural satellite, offered with tides and phases included.",
        "category": "satellite",
        "price": Decimal("4800000000000000"),
        "rarity": 1.0,
        "image_url": None,
        "status": ProductStatus.AVAILABLE,
        "reality_type": RealityType.REAL,
        "tags": ("space", "natural", "historic", "exclusive"),
        "attributes": {
            "space_affinity": 1.0,
            "danger": 0.65,
            "luxury": 0.95,
            "novelty": 0.85,
            "historical_value": 0.80,
        },
    },
    {
        "name": "Mars",
        "description": "A rust-colored planet with excellent views and a demanding commute.",
        "category": "planet",
        "price": Decimal("2100000000000000000000000"),
        "rarity": 1.0,
        "image_url": None,
        "status": ProductStatus.COMING_SOON,
        "reality_type": RealityType.REAL,
        "tags": ("space", "natural", "exclusive", "mysterious"),
        "attributes": {
            "space_affinity": 1.0,
            "danger": 0.90,
            "luxury": 0.98,
            "novelty": 0.92,
            "historical_value": 0.72,
        },
    },
    {
        "name": "Pacific Ocean",
        "description": "The world's largest ocean. Marine life and international boundaries sold separately.",
        "category": "natural-wonder",
        "price": Decimal("990000000000000000"),
        "rarity": 0.99,
        "image_url": None,
        "status": ProductStatus.UNAVAILABLE,
        "reality_type": RealityType.REAL,
        "tags": ("natural", "exclusive", "dangerous", "mysterious"),
        "attributes": {
            "nature_score": 1.0,
            "danger": 0.82,
            "luxury": 0.88,
            "novelty": 0.78,
            "historical_value": 0.70,
        },
    },
    {
        "name": "Tyrannosaurus Rex",
        "description": "One responsibly reconstructed apex predator. Requires a very sturdy enclosure.",
        "category": "prehistoric-creature",
        "price": Decimal("875000000000"),
        "rarity": 0.97,
        "image_url": None,
        "status": ProductStatus.SOLD_OUT,
        "reality_type": RealityType.HISTORICAL,
        "tags": ("prehistoric", "creature", "dangerous", "rare"),
        "attributes": {
            "space_affinity": 0.0,
            "danger": 1.0,
            "luxury": 0.60,
            "novelty": 0.95,
            "historical_value": 0.85,
        },
    },
    {
        "name": "Time Machine",
        "description": "A prototype temporal vehicle. Warranty becomes complicated before purchase.",
        "category": "impossible-technology",
        "price": Decimal("1234567890123456789012345678"),
        "rarity": 1.0,
        "image_url": None,
        "status": ProductStatus.COMING_SOON,
        "reality_type": RealityType.SPECULATIVE,
        "tags": ("technology", "mysterious", "exclusive", "historic"),
        "attributes": {
            "technology_level": 1.0,
            "danger": 0.93,
            "luxury": 0.99,
            "novelty": 1.0,
            "historical_value": 1.0,
        },
    },
    {
        "name": "Roman Empire",
        "description": "A vast historical civilization supplied as-is, including roads and administrative overhead.",
        "category": "empire",
        "price": Decimal("440000000000000000000"),
        "rarity": 0.96,
        "image_url": None,
        "status": ProductStatus.UNAVAILABLE,
        "reality_type": RealityType.HISTORICAL,
        "tags": ("historic", "luxury", "exclusive", "rare"),
        "attributes": {
            "historical_value": 1.0,
            "danger": 0.76,
            "luxury": 0.90,
            "novelty": 0.88,
            "technology_level": 0.32,
        },
    },
    {
        "name": "International Space Station",
        "description": "A legendary orbital laboratory with panoramic windows and zero-gravity delivery.",
        "category": "spacecraft",
        "price": Decimal("150000000000000"),
        "rarity": 0.995,
        "image_url": None,
        "status": ProductStatus.SOLD_OUT,
        "reality_type": RealityType.REAL,
        "tags": ("space", "technology", "historic", "exclusive"),
        "attributes": {
            "space_affinity": 1.0,
            "danger": 0.72,
            "luxury": 0.84,
            "novelty": 0.82,
            "technology_level": 0.94,
        },
    },
)


def _seed_categories(database: Session) -> dict[str, Category]:
    categories = {category.slug: category for category in database.scalars(select(Category)).all()}
    for data in CATEGORIES:
        category = categories.get(data["slug"])
        if category is None:
            category = Category(name=data["name"], slug=data["slug"])
            database.add(category)
            categories[data["slug"]] = category
        category.name = data["name"]
        category.parent = categories.get(data["parent"])
    database.flush()
    return categories


def _seed_tags(database: Session) -> dict[str, Tag]:
    tags = {tag.slug: tag for tag in database.scalars(select(Tag)).all()}
    for name in TAG_NAMES:
        tag = tags.get(name)
        if tag is None:
            tag = Tag(name=name, slug=name)
            database.add(tag)
            tags[name] = tag
        tag.name = name
    database.flush()
    return tags


def seed_products(database: Session) -> int:
    """Upsert the small catalog and all metadata without creating duplicates."""
    categories = _seed_categories(database)
    tags = _seed_tags(database)
    existing_products = {product.name: product for product in database.scalars(select(Product)).all()}
    created_count = 0

    for data in PRODUCTS:
        product = existing_products.get(data["name"])
        if product is None:
            product = Product(name=data["name"])
            database.add(product)
            existing_products[data["name"]] = product
            created_count += 1

        product.description = data["description"]
        product.category = categories[data["category"]]
        product.price = data["price"]
        product.rarity = data["rarity"]
        product.image_url = data["image_url"]
        product.status = data["status"]
        product.reality_type = data["reality_type"]
        product.catalog_version = "development_seed_v1"
        product.generation_seed = None
        product.tags = [tags[tag_slug] for tag_slug in data["tags"]]

        attributes = {attribute.attribute_name: attribute for attribute in product.attributes}
        for attribute_name, numeric_value in data["attributes"].items():
            attribute = attributes.get(attribute_name)
            if attribute is None:
                attribute = ProductAttribute(attribute_name=attribute_name)
                product.attributes.append(attribute)
            attribute.numeric_value = numeric_value

    database.commit()
    return created_count


def main() -> None:
    backup_path = backup_legacy_database()
    if backup_path is not None:
        print(f"Legacy database backed up to {backup_path}.")

    init_database()
    with SessionLocal() as database:
        created_count = seed_products(database)
    print(f"Product metadata seed complete: {created_count} product(s) added.")


if __name__ == "__main__":
    main()
