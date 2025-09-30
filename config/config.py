from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    """Paths and data-related options.

    Environment variables (with nesting via __):
      - DATA__RAW_PATH
      - DATA__PROCESSED_DIR
      - DATA__EXTERNAL_DIR
    """

    raw_path: Path = Field(default=Path("data/raw/CTG.xls"), description="UCI CTG dataset path (.xls/.xlsx/.csv)")
    processed_dir: Path = Field(default=Path("data/processed"))
    external_dir: Path = Field(default=Path("data/external"))

    @field_validator("raw_path")
    @classmethod
    def validate_raw_extension(cls, value: Path) -> Path:
        allowed = {".xls", ".xlsx", ".csv"}
        if value.suffix.lower() not in allowed:
            raise ValueError(f"raw_path must have one of extensions {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def validate_dirs(self) -> "DataSettings":
        # No creation here; just validate that parent directories look sane
        for directory in [self.processed_dir, self.external_dir]:
            if not isinstance(directory, Path):
                raise ValueError("processed_dir/external_dir must be Path instances")
        return self


class ModelSettings(BaseSettings):
    """Model hyperparameters and training configuration.

    Environment variables:
      - MODEL__ALGORITHM (random_forest|xgboost|lightgbm|logreg)
      - MODEL__N_ESTIMATORS
      - MODEL__LEARNING_RATE
      - MODEL__MAX_DEPTH
      - MODEL__RANDOM_STATE
      - MODEL__CLASS_WEIGHTS (JSON string mapping class -> weight)
      - MODEL__THRESHOLD_SUSPECT
      - MODEL__THRESHOLD_PATHOLOGIC
      - MODEL__MODEL_PATH
    """

    algorithm: Literal["random_forest", "xgboost", "lightgbm", "logreg"] = "random_forest"
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: Optional[int] = None
    random_state: int = 42
    class_weights: Optional[Dict[int, float]] = None

    # Inference decision thresholds for probability outputs (if supported)
    threshold_suspect: float = 0.5
    threshold_pathologic: float = 0.7

    model_path: Path = Field(default=Path("models/saved/baseline_rf.joblib"))

    @field_validator("n_estimators")
    @classmethod
    def n_estimators_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_estimators must be positive")
        return value

    @field_validator("learning_rate")
    @classmethod
    def learning_rate_range(cls, value: float) -> float:
        if not (0.0 < value <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        return value

    @field_validator("max_depth")
    @classmethod
    def max_depth_positive(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("max_depth must be positive when provided")
        return value

    @field_validator("class_weights")
    @classmethod
    def validate_class_weights(cls, value: Optional[Dict[int, float]]) -> Optional[Dict[int, float]]:
        if value is None:
            return value
        allowed_classes = {1, 2, 3}  # NSP: 1 Normal, 2 Suspect, 3 Pathologic
        for k, v in value.items():
            if k not in allowed_classes:
                raise ValueError("class_weights keys must be in {1,2,3} for NSP labels")
            if v <= 0:
                raise ValueError("class_weights values must be positive")
        return value

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ModelSettings":
        if not (0.0 <= self.threshold_suspect <= 1.0):
            raise ValueError("threshold_suspect must be in [0,1]")
        if not (0.0 <= self.threshold_pathologic <= 1.0):
            raise ValueError("threshold_pathologic must be in [0,1]")
        if self.threshold_suspect > self.threshold_pathologic:
            raise ValueError("threshold_suspect must be <= threshold_pathologic")
        return self


class FeatureSettings(BaseSettings):
    """Feature engineering settings.

    Environment variables:
      - FEATURES__TARGET_COLUMN
      - FEATURES__IMPUTATION_STRATEGY (mean|median|most_frequent)
      - FEATURES__SAMPLING_RATE_HZ
      - FEATURES__WINDOW_SECONDS
    """

    target_column: str = "NSP"
    imputation_strategy: Literal["mean", "median", "most_frequent"] = "median"
    sampling_rate_hz: int = 4  # Typical CTG is ~4 Hz after downsampling
    window_seconds: int = 60

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, value: str) -> str:
        if not value or value.strip() == "":
            raise ValueError("target_column must be a non-empty string")
        return value

    @field_validator("sampling_rate_hz")
    @classmethod
    def sampling_rate_plausible(cls, value: int) -> int:
        if not (1 <= value <= 1000):
            raise ValueError("sampling_rate_hz must be in [1, 1000] Hz")
        return value

    @field_validator("window_seconds")
    @classmethod
    def window_seconds_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("window_seconds must be positive")
        return value


class APISettings(BaseSettings):
    """FastAPI server settings.

    Environment variables:
      - API__HOST
      - API__PORT
      - API__RELOAD
      - API__CORS_ORIGINS (comma-separated)
    """

    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    @field_validator("port")
    @classmethod
    def port_range(cls, value: int) -> int:
        if not (1 <= value <= 65535):
            raise ValueError("port must be in [1, 65535]")
        return value

    @field_validator("cors_origins")
    @classmethod
    def sanitize_origins(cls, value: List[str]) -> List[str]:
        return [v.strip() for v in value if v and v.strip()]


class LoggingSettings(BaseSettings):
    """Logging configuration.

    Environment variables:
      - LOGGING__LEVEL (DEBUG|INFO|WARNING|ERROR|CRITICAL)
      - LOGGING__FORMAT
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class AppConfig(BaseSettings):
    """Top-level application configuration with nested settings.

    Supports nested env via double-underscore, e.g.:
      DATA__RAW_PATH=...  MODEL__N_ESTIMATORS=...  API__PORT=...
    An optional .env file at project root will be loaded automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "FetalGuard CTG Classifier"
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def medical_consistency_checks(self) -> "AppConfig":
        # Example medical domain guardrails
        if self.features.sampling_rate_hz * self.features.window_seconds < 10:
            # Ensure at least a minimal number of samples per window
            raise ValueError(
                "Insufficient samples per window; increase sampling_rate_hz or window_seconds",
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Cached access to application settings."""
    try:
        return AppConfig()
    except ValidationError as exc:
        # Re-raise with a clearer prefix for easier debugging in CLI/API logs
        raise RuntimeError(f"Configuration validation failed: {exc}") from exc


