import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.core.config import settings
from app.models.classifier import classifier

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the classifier on startup; clean up on shutdown."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    classifier.load()
    yield
    logger.info("Shutting down — releasing model resources.")
    # torch model will be garbage-collected automatically


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Dermatology mole classification API using a fine-tuned ViT with "
        "LayerNorm TTA for mobile distribution shift compensation.\n\n"
        "Supervised by **Prof. Balázs Harangi**, University of Debrecen.\n"
        "Submitted to **CITDS 2026**."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten before production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(api_router)
