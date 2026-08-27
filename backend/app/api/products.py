from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Product
from backend.app.db.session import get_db
from backend.app.schemas.product import ProductResponse


router = APIRouter(prefix="/products", tags=["products"])
DatabaseSession = Annotated[Session, Depends(get_db)]


@router.get("", response_model=list[ProductResponse])
def list_products(database: DatabaseSession) -> list[Product]:
    return list(database.scalars(select(Product).order_by(Product.id)).all())


@router.get("/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, database: DatabaseSession) -> Product:
    product = database.get(Product, product_id)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found",
        )
    return product
