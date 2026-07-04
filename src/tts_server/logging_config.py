"""Structured JSON logging. One log file per server startup; every record
carries the current request ID from a ContextVar. Never log secrets or
full user text — use truncate_text for text fields."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def truncate_text(s: str, limit: int = 80) -> str:
    return s if len(s) <= limit else s[:limit] + "..."


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "request_id": request_id_var.get(),
            }
        )


def setup_logging(log_dir: str = "logs") -> Path:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / f"server-{datetime.now():%Y%m%d-%H%M%S}.log"

    logger = logging.getLogger("tts_server")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(console)
    return log_file
