def test_healthz_reports_backend(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["backend_name"] == "mock"
    assert body["backend"]["loaded"] is True


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "tts_requests_total" in resp.text


def test_request_id_echoed(client):
    resp = client.get("/healthz", headers={"X-Request-ID": "trace-me"})
    assert resp.headers["x-request-id"] == "trace-me"


def test_request_id_generated_when_absent(client):
    resp = client.get("/healthz")
    assert len(resp.headers["x-request-id"]) == 32
