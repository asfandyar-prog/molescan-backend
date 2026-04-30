from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ClassLabel(str, Enum):
    healthy = "healthy"
    suspicious = "suspicious"
    malignant = "malignant"


class UncertaintyLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# ── Response ──────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """API response schema — matches the agreed mobile contract."""

    prediction: ClassLabel
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    uncertainty: UncertaintyLevel
    recommendation: str = Field(..., description="Human-readable clinical recommendation")
    location: str = Field(..., description="Body location reported by the mobile app")
    picture_date: date = Field(..., description="Date the image was captured")


# ── Internal ──────────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """Internal model passed from the classifier to the route handler."""

    label: ClassLabel
    confidence: float
    class_probs: dict[ClassLabel, float]
