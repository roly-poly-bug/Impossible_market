from fastapi import APIRouter

from backend.app.schemas.health import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Report whether the API process is available."""
    return HealthResponse(status="ok", project="Impossible Market")
