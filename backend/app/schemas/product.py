from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from backend.app.db.models import ProductStatus, RealityType


class CategoryResponse(BaseModel):
    id: int
    name: str
    slug: str

    model_config = ConfigDict(from_attributes=True)


class TagResponse(BaseModel):
    id: int
    name: str
    slug: str

    model_config = ConfigDict(from_attributes=True)


class ProductBaseResponse(BaseModel):
    id: int
    name: str
    description: str
    price: Decimal = Field(ge=0)
    rarity: float = Field(ge=0, le=1)
    image_url: str | None
    status: ProductStatus
    reality_type: RealityType
    catalog_version: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("price")
    def serialize_price(self, price: Decimal) -> str:
        """Keep prices exact when JSON is consumed by JavaScript."""
        return format(price, "f")


class ProductListResponse(ProductBaseResponse):
    # Preserve the original lightweight list response shape.
    category: str = Field(validation_alias="category_name")


class ProductDetailResponse(ProductBaseResponse):
    category: CategoryResponse
    tags: list[TagResponse]
    attributes: dict[str, float] = Field(validation_alias="attribute_values")
