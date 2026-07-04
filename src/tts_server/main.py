"""Application factory: config, logging, backend lifecycle, middleware,
error handling, and router registration."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tts_server.api import elevenlabs_compat, health, native, openai_compat
from tts_server.backends.registry import create_backend
from tts_server.config import AppConfig, load_config
from tts_server.errors import TTSServerError, UnsupportedFeatureError
from tts_server.logging_config import request_id_var, setup_logging
from tts_server.models import TTSRequest

logger = logging.getLogger("tts_server.main")


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_file = setup_logging(cfg.server.log_dir)
        logger.info("starting; backend=%s log_file=%s", cfg.backend.name, log_file)
        backend = create_backend(cfg)
        try:
            await backend.load()
            if cfg.backend.warmup:
                await backend.synthesize(
                    TTSRequest(text="warmup", sample_rate=cfg.audio.sample_rate)
                )
                logger.info("warmup complete")
            app.state.backend = backend
            app.state.config = cfg
            yield
        finally:
            await backend.close()
            logger.info("backend closed; shutdown complete")

    app = FastAPI(title="tts-server", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request_id_var.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(TTSServerError)
    async def tts_error_handler(request: Request, exc: TTSServerError):
        body: dict = {"error": {"code": exc.code, "message": exc.message}}
        if isinstance(exc, UnsupportedFeatureError) and exc.capabilities:
            body["error"]["capabilities"] = exc.capabilities.model_dump()
        return JSONResponse(status_code=exc.status_code, content=body)

    app.include_router(health.router)
    app.include_router(openai_compat.router)
    app.include_router(elevenlabs_compat.router)
    app.include_router(native.router)
    return app


app = create_app()
