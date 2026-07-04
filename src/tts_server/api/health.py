from __future__ import annotations

from fastapi import APIRouter, Request, Response

from tts_server.metrics import render_metrics

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request):
    backend = request.app.state.backend
    health = await backend.health()
    return {
        "status": "ok" if health.ok and health.loaded else "degraded",
        "backend_name": backend.name,
        "backend": health.model_dump(),
    }


@router.get("/metrics")
async def metrics_endpoint():
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
