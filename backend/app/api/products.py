from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from backend.app.db.models import Product
from backend.app.db.session import get_db
from backend.app.schemas.product import ProductDetailResponse, ProductListResponse


router = APIRouter(prefix="/products", tags=["products"])
DatabaseSession = Annotated[Session, Depends(get_db)]


@router.get("", response_model=list[ProductListResponse])
def list_products(database: DatabaseSession) -> list[Product]:
    statement = select(Product).options(joinedload(Product.category)).order_by(Product.id)
    return list(database.scalars(statement).all())


@router.get("/{product_id}", response_model=ProductDetailResponse)
def get_product(product_id: int, database: DatabaseSession) -> Product:
    statement = (
        select(Product)
        .where(Product.id == product_id)
        .options(
            joinedload(Product.category),
            selectinload(Product.tags),
            selectinload(Product.attributes),
        )
    )
    product = database.scalar(statement)
    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found",
        )
    return product
