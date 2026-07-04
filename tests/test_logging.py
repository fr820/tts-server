import json
import logging

from tts_server.logging_config import request_id_var, setup_logging, truncate_text


def test_creates_log_file_and_writes_json(tmp_path):
    log_file = setup_logging(str(tmp_path / "logs"))
    assert log_file.parent.name == "logs"
    logger = logging.getLogger("tts_server.test")
    request_id_var.set("req-123")
    logger.info("hello %s", "world")
    for handler in logging.getLogger("tts_server").handlers:
        handler.flush()
    record = json.loads(log_file.read_text().strip().splitlines()[-1])
    assert record["message"] == "hello world"
    assert record["request_id"] == "req-123"
    assert record["level"] == "INFO"


def test_new_file_per_setup(tmp_path):
    f1 = setup_logging(str(tmp_path / "logs"))
    import time

    time.sleep(1.1)  # filename has second granularity
    f2 = setup_logging(str(tmp_path / "logs"))
    assert f1 != f2


def test_truncate_text():
    assert truncate_text("short") == "short"
    long = "x" * 200
    out = truncate_text(long)
    assert len(out) <= 84 and out.endswith("...")
