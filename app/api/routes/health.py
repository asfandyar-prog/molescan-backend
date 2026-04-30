from fastapi import APIRouter

from app.core.config import settings
from app.models.classifier import classifier

router = APIRouter()


@router.get("/health", tags=["infra"])
async def health_check() -> dict:
    """Liveness / readiness probe used by Docker and the faculty server."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "model_loaded": classifier.is_loaded,
        "tta_enabled": settings.tta_enabled,
    }
