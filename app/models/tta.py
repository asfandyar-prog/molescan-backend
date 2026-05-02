"""
LayerNorm Test-Time Adaptation for HuggingFace ViTForImageClassification.

Ported from:
    Asfand Yar, "Predictive Self-Supervised Vision Transformers under
    Test-Time Distribution Shifts with Lightweight TTA", BSc Thesis,
    University of Debrecen, 2026.

Mechanism (per thesis §3.5):
    Freeze all model parameters; unfreeze only the LayerNorm affine
    parameters γ (weight) and β (bias). At test time, minimize softmax
    entropy on each incoming batch. Episodic reset restores the original
    model state after every prediction.

Total adapted parameters for ViT-B/16: 2 × 768 × 12 = 18,432
(≈0.02% of the 86M backbone parameters).

References (thesis bibliography):
    [3]  Wang et al., TENT: Fully Test-Time Adaptation by Entropy
         Minimization, ICLR 2021.
    [10] Sun et al., Test-Time Training with Self-Supervision for
         Generalization under Distribution Shifts, ICML 2020.
    [11] Liu et al., TTT++: When Does Self-Supervised Test-Time Training
         Fail or Thrive?, NeurIPS 2021.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from transformers import ViTForImageClassification


# ---------------------------------------------------------------------------
# Entropy loss (thesis §3.5.2)
# ---------------------------------------------------------------------------

def softmax_entropy(logits: Tensor) -> Tensor:
    """Per-sample softmax entropy, numerically stable via log-softmax.

    H(p) = -Σ_c p_c log p_c
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


# ---------------------------------------------------------------------------
# LayerNorm parameter selection (thesis §3.5.3)
# ---------------------------------------------------------------------------

def collect_layernorm_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
    return params


def freeze_non_layernorm_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)


# ---------------------------------------------------------------------------
# TTA wrapper
# ---------------------------------------------------------------------------

class LayerNormTTA(nn.Module):
    """Episodic LayerNorm-only entropy minimization for ViTForImageClassification.

    Hyperparameters are pinned to the thesis configuration (Appendix A.1):
        lr     = 1e-4
        steps  = 1
        optim  = Adam
        reset  = episodic (True)

    Args:
        model: a HuggingFace ViTForImageClassification (or any model exposing
               a `.logits` attribute on its forward output).
        lr: learning rate for LayerNorm γ/β updates. Thesis default: 1e-4.
        steps: gradient steps per batch. Thesis default: 1.
        episodic: reset to pre-adaptation state after each batch. Thesis
                  default: True. Set False for continual TTA (drift-prone).
        entropy_threshold: skip adaptation on samples with entropy below
                           this value. Thesis code supports this; thesis
                           text does not formally analyse it. Set None to
                           always adapt.
    """

    def __init__(
        self,
        model: ViTForImageClassification,
        lr: float = 1e-4,
        steps: int = 1,
        episodic: bool = True,
        entropy_threshold: float | None = None,
    ) -> None:
        super().__init__()

        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        self.lr = lr
        self.steps = steps
        self.episodic = episodic
        self.entropy_threshold = entropy_threshold

        # Deep copy so the caller's model is never mutated.
        self.model = copy.deepcopy(model)
        self._original_state = copy.deepcopy(self.model.state_dict())

        freeze_non_layernorm_params(self.model)
        self._params = collect_layernorm_params(self.model)
        if not self._params:
            raise RuntimeError(
                "No LayerNorm parameters found in model. "
                "LayerNormTTA requires a transformer-style architecture."
            )
        self.optimizer = torch.optim.Adam(self._params, lr=self.lr)

    def reset(self) -> None:
        """Restore pre-adaptation weights and reinstantiate the optimizer.

        Reinstantiation is required because `load_state_dict` replaces
        parameter tensors in-place; the old optimizer would otherwise be
        holding stale references.
        """
        self.model.load_state_dict(self._original_state)
        self._params = collect_layernorm_params(self.model)
        self.optimizer = torch.optim.Adam(self._params, lr=self.lr)

    def _logits(self, pixel_values: Tensor) -> Tensor:
        """HuggingFace forward returns an ImageClassifierOutput; we want logits."""
        return self.model(pixel_values=pixel_values).logits

    @torch.enable_grad()
    def forward(self, pixel_values: Tensor) -> Tensor:
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            self.model.train()
            logits = self._logits(pixel_values)

            if self.entropy_threshold is not None:
                batch_entropy = softmax_entropy(logits)
                mask = batch_entropy >= self.entropy_threshold
                if mask.sum() == 0:
                    break
                loss = softmax_entropy(logits[mask]).mean()
            else:
                loss = softmax_entropy(logits).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            return self._logits(pixel_values)

    @property
    def num_adaptable_params(self) -> int:
        return sum(p.numel() for p in self._params)