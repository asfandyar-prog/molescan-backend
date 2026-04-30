"""
MoleScan ViT Classifier
=======================
Fine-tuned Vision Transformer (ViT-B/16) for 3-class mole classification.

Novel contribution:
    LayerNorm TTA — at inference time we run a few gradient steps that update
    *only* the LayerNorm affine parameters (γ, β) to adapt the model to the
    mobile camera's distribution shift, then reset them after prediction.

TODO (Asfand):
    - Load fine-tuned weights from `settings.model_weights_path`
    - Implement `_layer_norm_tta()` using the ISIC validation loss as surrogate
    - Calibrate `_uncertainty()` thresholds on a held-out ISIC split
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings
from app.schemas.prediction import ClassLabel, PredictionResult

logger = logging.getLogger(__name__)

CLASS_LABELS = [ClassLabel.healthy, ClassLabel.suspicious, ClassLabel.malignant]


class MoleScanClassifier:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model and weights.  Called once at startup."""
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
                    "No weights found at %s — using ImageNet pre-trained backbone only.",
                    weights_path,
                )

            self.model.to(self.device).eval()
            self._loaded = True
            logger.info("Classifier ready on %s", self.device)

        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image: Image.Image) -> PredictionResult:
        """Run classification (+ optional LayerNorm TTA) on a PIL image."""
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call .load() first.")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        if settings.tta_enabled:
            logits = self._layer_norm_tta(inputs)
        else:
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = F.softmax(logits, dim=-1).squeeze()
        confidence, pred_idx = probs.max(dim=-1)

        label = CLASS_LABELS[pred_idx.item()]
        class_probs = {CLASS_LABELS[i]: probs[i].item() for i in range(len(CLASS_LABELS))}

        return PredictionResult(
            label=label,
            confidence=confidence.item(),
            class_probs=class_probs,
        )

    # ── LayerNorm TTA (novel contribution) ────────────────────────────────────

    def _layer_norm_tta(self, inputs: dict) -> torch.Tensor:
        """
        Adapt LayerNorm affine parameters to the test image distribution,
        then restore original parameters after prediction.

        This compensates for the distribution shift introduced by the
        digital microscope used in the mobile app.

        Reference:  TTT++ / Test-Time Training with Self-Supervision
                    Wang et al. 2020 (Tent)
        """
        # TODO: implement full TTA loop
        # Skeleton:
        #   1. collect all LayerNorm modules
        #   2. save original (weight, bias) pairs
        #   3. optimise with AdamW for `settings.tta_steps` steps
        #      using entropy minimisation as the surrogate loss
        #   4. run final forward pass
        #   5. restore original parameters
        #   6. return logits

        # Fallback — standard forward pass until TTA is implemented
        logger.debug("TTA not yet implemented — falling back to standard forward pass.")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits


# Module-level singleton — imported by routes
classifier = MoleScanClassifier()
