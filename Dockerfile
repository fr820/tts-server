# CUDA runtime base so the qwen3 extra can use the GPU; the mock backend
# runs in this same image with no GPU attached.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
RUN uv python install 3.12

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY config.example.yaml ./config.yaml
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENV TTS_BACKEND=mock
CMD ["uv", "run", "--no-sync", "uvicorn", "tts_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
