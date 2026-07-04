from tts_server import metrics


def test_counters_and_render():
    metrics.REQUESTS.labels(backend="mock", api="openai").inc()
    metrics.TTFA_SECONDS.labels(backend="mock").observe(0.05)
    metrics.RTF.labels(backend="mock").observe(0.3)
    payload, content_type = metrics.render_metrics()
    text = payload.decode()
    assert "tts_requests_total" in text
    assert "tts_ttfa_seconds" in text
    assert "tts_rtf" in text
    assert "tts_gpu_memory_mb" in text
    assert content_type.startswith("text/plain")


def test_active_sessions_gauge():
    metrics.ACTIVE_SESSIONS.inc()
    metrics.ACTIVE_SESSIONS.dec()
    payload, _ = metrics.render_metrics()
    assert "tts_active_sessions" in payload.decode()
