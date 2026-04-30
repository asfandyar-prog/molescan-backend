from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = "Molescan Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    # Model
    model_checkpoint: str = "google/vit-base-patch16-224"
    model_weights_path: str = "weights/molescan_vit.pt"
    num_classes: int = 3  # healthy / suspicious / malignant
    image_size: int = 224

    # TTA (LayerNorm TTA — novel contribution)
    tta_enabled: bool = True
    tta_steps: int = 10
    tta_learning_rate: float = 1e-3

    # Inference thresholds
    confidence_high: float = 0.85
    confidence_medium: float = 0.60

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


settings = Settings()
