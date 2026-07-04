import pytest

from tts_server.config import AppConfig, load_config


def test_defaults_without_yaml_or_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no config.yaml here
    for var in ("TTS_CONFIG", "TTS_BACKEND", "TTS_HOST", "TTS_PORT", "TTS_DEVICE"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.backend.name == "mock"
    assert cfg.server.port == 8000
    assert cfg.audio.sample_rate == 24000


def test_yaml_file_overrides_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TTS_BACKEND", raising=False)
    (tmp_path / "myconf.yaml").write_text(
        "server:\n  port: 9000\nbackend:\n  name: qwen3\n  device: cuda\n"
    )
    cfg = load_config(str(tmp_path / "myconf.yaml"))
    assert cfg.server.port == 9000
    assert cfg.backend.name == "qwen3"
    assert cfg.backend.device == "cuda"
    assert cfg.audio.channels == 1  # untouched section keeps defaults


def test_env_overrides_yaml(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("backend:\n  name: qwen3\n")
    monkeypatch.setenv("TTS_BACKEND", "mock")
    monkeypatch.setenv("TTS_PORT", "8123")
    cfg = load_config()
    assert cfg.backend.name == "mock"
    assert cfg.server.port == 8123


def test_tts_config_env_points_to_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TTS_BACKEND", raising=False)
    p = tmp_path / "elsewhere.yaml"
    p.write_text("server:\n  host: 127.0.0.1\n")
    monkeypatch.setenv("TTS_CONFIG", str(p))
    assert load_config().server.host == "127.0.0.1"
