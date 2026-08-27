from fastapi import APIRouter

from backend.app.api.products import router as products_router
from backend.app.schemas.health import HealthResponse


router = APIRouter()
router.include_router(products_router)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Report whether the API process is available."""
    return HealthResponse(status="ok", project="Impossible Market")
