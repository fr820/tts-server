import pytest
from fastapi.testclient import TestClient

from tts_server.config import AppConfig


@pytest.fixture
def app_config() -> AppConfig:
    cfg = AppConfig()
    cfg.backend.name = "mock"
    cfg.backend.warmup = False
    cfg.backend.options.update({"first_chunk_delay_ms": 0, "chunk_interval_ms": 0})
    cfg.server.log_dir = "logs"
    return cfg


@pytest.fixture
def client(app_config, tmp_path):
    app_config.server.log_dir = str(tmp_path / "logs")
    from tts_server.main import create_app

    with TestClient(create_app(app_config)) as client:
        yield client
