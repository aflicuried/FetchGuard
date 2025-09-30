from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "FetalGuard CTG Classifier"
    data_raw_path: Path = Path("data/raw/CTG.xls")
    model_path: Path = Path("models/saved/baseline_rf.joblib")


settings = Settings()


