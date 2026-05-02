"""
MoleScan ViT Classifier
=======================
Fine-tuned Vision Transformer (ViT-B/16) for 3-class mole classification
(healthy / suspicious / malignant).

At inference, this module can route forward passes through a LayerNorm-only
entropy-minimization TTA wrapper (TENT, Wang et al. ICLR 2021; LayerNorm
extension following TTT++, Liu et al. NeurIPS 2021). The wrapper itself
lives in app.models.tta and is a faithful port of the thesis implementation.

DEPLOYMENT NOTE — batch-size-1 is a known issue:
    Entropy minimization is degenerate on a single image: the model can
    drive entropy to zero by collapsing to any one class. The thesis adapts
    on full test loaders. A FastAPI endpoint receives one image at a time.
    The batch-size-1 fix is configured at the call site (see settings.tta_*
    and the docstring on `predict()`); it is not silently absorbed here.

TODO (Asfand):
    - Drop fine-tuned ISIC weights into settings.model_weights_path.
    - Decide on the batch-size-1 deployment fix (see settings.tta_mode).
    - Calibrate uncertainty thresholds on a held-out ISIC split.
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from app.models.tta import LayerNormTTA
from app.schemas.prediction import ClassLabel, PredictionResult

logger = logging.getLogger(__name__)

CLASS_LABELS = [ClassLabel.healthy, ClassLabel.suspicious, ClassLabel.malignant]


class MoleScanClassifier:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None       # base ViTForImageClassification
        self.tta_model = None   # LayerNormTTA wrapper (or None if disabled)
        self.processor = None
        self._loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model and weights. Called once at startup via FastAPI lifespan."""
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor

            logger.info("Loading ViT model from %s", settings.model_checkpoint)
            self.processor = ViTImageProcessor.from_pretrained(settings.model_checkpoint)
            self.model = ViTForImageClassification.from_pretrained(
                settings.model_checkpoint,
                num_labels=settings.num_classes,
                ignore_mismatched_sizes=True,
            )

            weights_path = Path(settings.model_weights_path)
            if weights_path.exists():
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                logger.info("Fine-tuned weights loaded from %s", weights_path)
            else:
                logger.warning(
                    "No weights at %s — using ImageNet-pretrained backbone only. "
                    "Predictions will not be clinically meaningful until ISIC "
                    "fine-tuning is complete.",
                    weights_path,
                )

            self.model.to(self.device).eval()

            if settings.tta_enabled:
                self.tta_model = LayerNormTTA(
                    self.model,
                    lr=settings.tta_learning_rate,
                    steps=settings.tta_steps,
                    episodic=settings.tta_episodic,
                    entropy_threshold=settings.tta_entropy_threshold,
                ).to(self.device)
                logger.info(
                    "TTA wrapper attached: lr=%.0e steps=%d episodic=%s "
                    "entropy_threshold=%s adaptable_params=%d",
                    settings.tta_learning_rate,
                    settings.tta_steps,
                    settings.tta_episodic,
                    settings.tta_entropy_threshold,
                    self.tta_model.num_adaptable_params,
                )
            else:
                logger.info("TTA disabled — using base ViT forward pass.")

            self._loaded = True
            logger.info("Classifier ready on %s", self.device)

        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> PredictionResult:
        """Run classification on a single PIL image.

        Routing depends on `settings.tta_enabled` and the deployment fix
        chosen for batch-size-1 (see settings.tta_entropy_threshold). With
        a high entropy threshold, single confident requests bypass adaptation
        and only rare batched/uncertain inputs trigger TTA.
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call .load() first.")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"]

        if self.tta_model is not None:
            # LayerNormTTA returns logits and handles its own grad context.
            logits = self.tta_model(pixel_values)
        else:
            with torch.no_grad():
                logits = self.model(pixel_values=pixel_values).logits

        probs = F.softmax(logits, dim=-1).squeeze(0)
        confidence, pred_idx = probs.max(dim=-1)

        label = CLASS_LABELS[pred_idx.item()]
        class_probs = {CLASS_LABELS[i]: probs[i].item() for i in range(len(CLASS_LABELS))}

        return PredictionResult(
            label=label,
            confidence=confidence.item(),
            class_probs=class_probs,
        )


# Module-level singleton — imported by routes
classifier = MoleScanClassifier()