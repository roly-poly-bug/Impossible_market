import enum
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base
from backend.app.db.types import MoneyAmount


class EventType(str, enum.Enum):
    IMPRESSION = "impression"
    VIEW = "view"
    FAVORITE = "favorite"
    ADD_TO_CART = "add_to_cart"
    PURCHASE = "purchase"


class ProductStatus(str, enum.Enum):
    AVAILABLE = "available"
    SOLD_OUT = "sold_out"
    UNAVAILABLE = "unavailable"
    COMING_SOON = "coming_soon"


class RealityType(str, enum.Enum):
    REAL = "real"
    HISTORICAL = "historical"
    FICTIONAL = "fictional"
    ABSTRACT = "abstract"
    SPECULATIVE = "speculative"


product_tags = Table(
    "product_tags",
    Base.metadata,
    Column("product_id", ForeignKey("products.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sessions: Mapped[list["Session"]] = relationship(back_populates="user")
    events: Mapped[list["Event"]] = relationship(back_populates="user")


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"), nullable=True, index=True)

    parent: Mapped["Category | None"] = relationship(remote_side=[id], back_populates="children")
    children: Mapped[list["Category"]] = relationship(back_populates="parent")
    products: Mapped[list["Product"]] = relationship(back_populates="category")


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True)

    products: Mapped[list["Product"]] = relationship(secondary=product_tags, back_populates="tags")


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("rarity >= 0 AND rarity <= 1", name="ck_products_rarity_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"), index=True)
    price: Mapped[Decimal] = mapped_column(MoneyAmount())
    rarity: Mapped[float] = mapped_column(Float)
    image_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    status: Mapped[ProductStatus] = mapped_column(
        Enum(
            ProductStatus,
            values_callable=lambda statuses: [status.value for status in statuses],
            native_enum=False,
            create_constraint=True,
            length=32,
        ),
        default=ProductStatus.AVAILABLE,
        server_default=ProductStatus.AVAILABLE.value,
        index=True,
    )
    reality_type: Mapped[RealityType] = mapped_column(
        Enum(
            RealityType,
            values_callable=lambda reality_types: [reality_type.value for reality_type in reality_types],
            native_enum=False,
            create_constraint=True,
            length=32,
        ),
        default=RealityType.SPECULATIVE,
        server_default=RealityType.SPECULATIVE.value,
        index=True,
    )
    catalog_version: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    generation_seed: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    category: Mapped[Category] = relationship(back_populates="products")
    tags: Mapped[list[Tag]] = relationship(
        secondary=product_tags,
        back_populates="products",
        order_by=Tag.name,
    )
    attributes: Mapped[list["ProductAttribute"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
        order_by="ProductAttribute.attribute_name",
    )
    events: Mapped[list["Event"]] = relationship(back_populates="product")

    @property
    def category_name(self) -> str:
        return self.category.name

    @property
    def attribute_values(self) -> dict[str, float]:
        return {attribute.attribute_name: attribute.numeric_value for attribute in self.attributes}


class ProductAttribute(Base):
    __tablename__ = "product_attributes"
    __table_args__ = (
        UniqueConstraint("product_id", "attribute_name", name="uq_product_attributes_product_name"),
        CheckConstraint("numeric_value >= 0 AND numeric_value <= 1", name="ck_product_attributes_value_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    attribute_name: Mapped[str] = mapped_column(String(100), index=True)
    numeric_value: Mapped[float] = mapped_column(Float)

    product: Mapped[Product] = relationship(back_populates="attributes")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User | None] = relationship(back_populates="sessions")
    events: Mapped[list["Event"]] = relationship(back_populates="session")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[EventType] = mapped_column(Enum(EventType), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    user: Mapped[User | None] = relationship(back_populates="events")
    product: Mapped[Product] = relationship(back_populates="events")
    session: Mapped[Session] = relationship(back_populates="events")
