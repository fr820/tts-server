"""Configuration: defaults -> YAML file -> environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from tts_server.models import AudioFormat


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_s: float = 120.0
    log_dir: str = "logs"


class BackendConfig(BaseModel):
    name: str = "mock"
    device: str = "auto"
    dtype: str = "bf16"
    model_path: str | None = None
    compile: bool = False
    warmup: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class AudioConfig(BaseModel):
    default_format: AudioFormat = AudioFormat.PCM_S16LE
    sample_rate: int = 24000
    channels: int = 1


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


def _yaml_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("TTS_CONFIG")
    if env:
        return Path(env)
    default = Path("config.yaml")
    return default if default.exists() else None


def load_config(path: str | None = None) -> AppConfig:
    data: dict[str, Any] = {}
    yaml_path = _yaml_path(path)
    if yaml_path is not None:
        data = yaml.safe_load(yaml_path.read_text()) or {}

    cfg = AppConfig.model_validate(data)

    if backend := os.environ.get("TTS_BACKEND"):
        cfg.backend.name = backend
    if host := os.environ.get("TTS_HOST"):
        cfg.server.host = host
    if port := os.environ.get("TTS_PORT"):
        cfg.server.port = int(port)
    if device := os.environ.get("TTS_DEVICE"):
        cfg.backend.device = device
    return cfg
