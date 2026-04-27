from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / '.env')

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'datathon_fase05')
MODEL_ARTIFACT_PATH = os.getenv('MODEL_ARTIFACT_PATH', 'models/model.joblib')
MODEL_NAME = os.getenv('MODEL_NAME', 'datathon_baseline')
MODEL_VERSION = os.getenv('MODEL_VERSION', '0.1.0')
SERVICE_HOST = os.getenv('SERVICE_HOST', '0.0.0.0')
SERVICE_PORT = int(os.getenv('SERVICE_PORT', '8000'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

@dataclass
class ModelConfig:
    random_state: int
    test_size: float
    solver: str
    penalty: str
    C: float

@dataclass
class MonitoringConfig:
    drift_warning_threshold: float
    drift_alert_threshold: float


def load_yaml_config(path: str) -> dict[str, object]:
    with open(Path(path), 'r', encoding='utf-8') as config_file:
        return yaml.safe_load(config_file) or {}


def load_model_config() -> ModelConfig:
    config = load_yaml_config(BASE_DIR / 'configs' / 'model_config.yaml')
    return ModelConfig(
        random_state=int(config.get('random_state', 42)),
        test_size=float(config.get('test_size', 0.2)),
        solver=str(config.get('solver', 'liblinear')),
        penalty=str(config.get('penalty', 'l2')),
        C=float(config.get('C', 1.0)),
    )


def load_monitoring_config() -> MonitoringConfig:
    config = load_yaml_config(BASE_DIR / 'configs' / 'monitoring_config.yaml')
    return MonitoringConfig(
        drift_warning_threshold=float(config.get('drift_warning_threshold', 0.1)),
        drift_alert_threshold=float(config.get('drift_alert_threshold', 0.2)),
    )
