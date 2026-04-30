import logging
from datetime import date

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
import io

from app.core.config import settings
from app.models.classifier import classifier
from app.schemas.prediction import ClassLabel, PredictionResponse, UncertaintyLevel

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Recommendation copy ───────────────────────────────────────────────────────

_RECOMMENDATIONS: dict[ClassLabel, str] = {
    ClassLabel.healthy: (
        "No immediate action required. Continue routine self-examination "
        "every 3 months and consult a dermatologist for your annual check-up."
    ),
    ClassLabel.suspicious: (
        "This mole shows features that warrant professional evaluation. "
        "Please book a dermatology appointment within 4 weeks."
    ),
    ClassLabel.malignant: (
        "Urgent clinical review is strongly recommended. Please contact a "
        "dermatologist or visit an urgent-care clinic as soon as possible."
    ),
}


def _uncertainty(confidence: float) -> UncertaintyLevel:
    if confidence >= settings.confidence_high:
        return UncertaintyLevel.low
    if confidence >= settings.confidence_medium:
        return UncertaintyLevel.medium
    return UncertaintyLevel.high


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Mole image captured by the digital microscope"),
    location: str = Form(..., description="Body location (e.g. 'left forearm')"),
    picture_date: date = Form(..., description="Capture date in YYYY-MM-DD format"),
) -> PredictionResponse:
    """
    Classify a mole image as **healthy / suspicious / malignant**.

    - Accepts JPEG or PNG uploads from the mobile app.
    - Applies LayerNorm TTA when `TTA_ENABLED=true` (default).
    - Returns a structured response with recommendation text.
    """
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not ready — please retry shortly.")

    # Validate MIME type
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Use JPEG or PNG.",
        )

    # Decode image
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        logger.warning("Image decode failed: %s", exc)
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    # Run classifier
    try:
        result = classifier.predict(image)
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Inference failed — see server logs.")

    return PredictionResponse(
        prediction=result.label,
        confidence=round(result.confidence, 4),
        uncertainty=_uncertainty(result.confidence),
        recommendation=_RECOMMENDATIONS[result.label],
        location=location,
        picture_date=picture_date,
    )
