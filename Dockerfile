# CUDA runtime base so the qwen3 extra can use the GPU; the mock backend
# runs in this same image with no GPU attached. Override `CUDA_IMAGE` to
# point at a reachable registry (nvcr.io, a mirror) if Docker Hub is blocked.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    HF_HOME=/models
RUN uv python install 3.12

# `INSTALL_QWEN3=1` adds the GPU stack (torch/transformers/qwen-tts). Default
# off so the mock image stays small and builds without a GPU or heavy downloads.
ARG INSTALL_QWEN3=0
COPY pyproject.toml uv.lock ./
RUN if [ "$INSTALL_QWEN3" = "1" ]; then QWEN3_FLAG="--extra qwen3"; else QWEN3_FLAG=""; fi \
    && uv sync --frozen --no-install-project --no-dev $QWEN3_FLAG

# qwen3's audio stack pulls the `sox` PyPI wheel, which is a wrapper around the
# `sox` CLI and prints "SoX could not be found!" at import when the binary is
# absent (and would crash any code path that actually shells out to it). Install
# the binary on the GPU path only. Placed after the heavy uv sync so the torch
# layer stays cached; mock builds skip it (the mock backend has no audio deps).
RUN if [ "$INSTALL_QWEN3" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends sox \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY src/ src/
COPY scripts/ scripts/
COPY config.example.yaml ./config.yaml
RUN if [ "$INSTALL_QWEN3" = "1" ]; then QWEN3_FLAG="--extra qwen3"; else QWEN3_FLAG=""; fi \
    && uv sync --frozen --no-dev $QWEN3_FLAG

EXPOSE 8000
ENV TTS_BACKEND=mock
CMD ["uv", "run", "--no-sync", "uvicorn", "tts_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
