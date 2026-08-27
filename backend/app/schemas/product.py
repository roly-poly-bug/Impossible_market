from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class ProductResponse(BaseModel):
    id: int
    name: str
    description: str
    category: str
    price: Decimal = Field(ge=0)
    rarity: float = Field(ge=0, le=1)
    image_url: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_serializer("price")
    def serialize_price(self, price: Decimal) -> str:
        """Keep prices exact when JSON is consumed by JavaScript."""
        return format(price, "f")
