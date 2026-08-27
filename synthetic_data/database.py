from dataclasses import dataclass
import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.models import Category, Event, Product, ProductAttribute, Tag
from synthetic_data.config import CATALOG_VERSION, TAG_VOCABULARY
from synthetic_data.product_generator import SyntheticProductRecord, validate_catalog
from synthetic_data.vocabularies import CATEGORY_SPECS


@dataclass(frozen=True)
class CatalogWriteResult:
    created: int
    updated: int
    deleted: int


def slugify(value: str) -> str:
    normalized = value.lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def _upsert_categories(database: Session) -> dict[str, Category]:
    categories = {category.slug: category for category in database.scalars(select(Category)).all()}
    parent_names = dict.fromkeys(spec.parent for spec in CATEGORY_SPECS)
    for parent_name in parent_names:
        parent_slug = slugify(parent_name)
        parent = categories.get(parent_slug)
        if parent is None:
            parent = Category(name=parent_name, slug=parent_slug)
            database.add(parent)
            categories[parent_slug] = parent
        parent.name = parent_name
        parent.parent = None

    for spec in CATEGORY_SPECS:
        category = categories.get(spec.slug)
        if category is None:
            category = Category(name=spec.name, slug=spec.slug)
            database.add(category)
            categories[spec.slug] = category
        category.name = spec.name
        category.parent = categories[slugify(spec.parent)]

    database.flush()
    return categories


def _upsert_tags(database: Session) -> dict[str, Tag]:
    tags = {tag.slug: tag for tag in database.scalars(select(Tag)).all()}
    for tag_name in sorted(TAG_VOCABULARY):
        tag = tags.get(tag_name)
        if tag is None:
            tag = Tag(name=tag_name, slug=tag_name)
            database.add(tag)
            tags[tag_name] = tag
        tag.name = tag_name
    database.flush()
    return tags


def _delete_existing_catalog(database: Session, products: list[Product]) -> int:
    if not products:
        return 0
    product_ids = [product.id for product in products]
    event_count = database.scalar(
        select(func.count()).select_from(Event).where(Event.product_id.in_(product_ids))
    )
    if event_count:
        raise RuntimeError(
            "Synthetic products already have event records and cannot be safely replaced."
        )
    for product in products:
        database.delete(product)
    database.flush()
    return len(products)


def write_catalog(
    database: Session,
    records: list[SyntheticProductRecord],
    *,
    replace_existing: bool = False,
) -> CatalogWriteResult:
    """Validate and upsert one deterministic catalog into an existing database."""
    validate_catalog(records, expected_count=len(records))
    requested_names = {record.name for record in records}
    requested_seed = records[0].generation_seed
    existing_catalog = list(
        database.scalars(
            select(Product).where(Product.catalog_version == CATALOG_VERSION)
        ).all()
    )
    existing_names = {product.name for product in existing_catalog}
    existing_seeds = {product.generation_seed for product in existing_catalog}
    catalog_differs = existing_catalog and (
        existing_names != requested_names or existing_seeds != {requested_seed}
    )
    if catalog_differs and not replace_existing:
        raise RuntimeError(
            "A different synthetic_product_v1 catalog already exists. "
            "Re-run with --replace-existing to replace only that synthetic catalog."
        )

    deleted_count = _delete_existing_catalog(database, existing_catalog) if replace_existing else 0
    categories = _upsert_categories(database)
    tags = _upsert_tags(database)
    existing_products = {
        product.name: product
        for product in database.scalars(select(Product).where(Product.name.in_(requested_names))).all()
    }
    created_count = 0
    updated_count = 0

    for record in records:
        product = existing_products.get(record.name)
        if product is None:
            product = Product(name=record.name)
            database.add(product)
            existing_products[record.name] = product
            created_count += 1
        else:
            updated_count += 1

        product.description = record.description
        product.category = categories[record.category_slug]
        product.price = record.price
        product.rarity = record.rarity
        product.image_url = record.image_url
        product.status = record.status
        product.reality_type = record.reality_type
        product.catalog_version = record.catalog_version
        product.generation_seed = record.generation_seed
        product.tags = [tags[tag_name] for tag_name in record.tags]

        desired_attributes = record.attributes
        existing_attributes = {
            attribute.attribute_name: attribute for attribute in product.attributes
        }
        for attribute in list(product.attributes):
            if attribute.attribute_name not in desired_attributes:
                product.attributes.remove(attribute)
        for attribute_name, numeric_value in desired_attributes.items():
            attribute = existing_attributes.get(attribute_name)
            if attribute is None:
                attribute = ProductAttribute(attribute_name=attribute_name)
                product.attributes.append(attribute)
            attribute.numeric_value = numeric_value

    database.commit()
    return CatalogWriteResult(
        created=created_count,
        updated=updated_count,
        deleted=deleted_count,
    )
