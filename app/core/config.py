from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "Molescan Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Model ─────────────────────────────────────────────────────────────────
    model_checkpoint: str = "google/vit-base-patch16-224"
    model_weights_path: str = "weights/molescan_vit.pt"
    num_classes: int = 3                        # healthy / suspicious / malignant
    image_size: int = 224

    # ── TTA (thesis Appendix A.1 hyperparameters; do NOT silently change) ────
    # Mechanism:  TENT (Wang et al., ICLR 2021), LayerNorm-only extension.
    # Reference:  app/models/tta.py
    tta_enabled: bool = True
    tta_learning_rate: float = 1e-4             # thesis canonical
    tta_steps: int = 1                          # thesis canonical
    tta_episodic: bool = True                   # reset weights after each call
    # Batch-size-1 safety: when set, samples with entropy below this value
    # skip adaptation. Set high (e.g. 99.0) to effectively disable adaptation
    # on solo confident requests; set None to always adapt (degenerate at b=1).
    tta_entropy_threshold: float | None = None

    # ── Inference thresholds (uncertainty band → recommendation) ─────────────
    confidence_high: float = 0.85
    confidence_medium: float = 0.60

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


settings = Settings()