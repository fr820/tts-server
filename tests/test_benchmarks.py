import json

from benchmarks.common import percentiles, write_results


def test_percentiles():
    stats = percentiles([float(i) for i in range(1, 101)])
    assert stats["p50"] == 50.5
    assert stats["p90"] == 90.1
    assert stats["p95"] == 95.05
    assert stats["min"] == 1.0 and stats["max"] == 100.0


def test_write_results_labels_mock(tmp_path):
    results = {
        "bench": "http",
        "backend": "mock",
        "concurrency": 2,
        "requests": 10,
        "failures": 0,
        "ttfa_ms": percentiles([10.0, 12.0]),
        "latency_ms": percentiles([100.0, 110.0]),
        "rtf": percentiles([0.1, 0.2]),
        "throughput_rps": 5.0,
    }
    json_path, md_path = write_results(results, out_dir=str(tmp_path))
    saved = json.loads(json_path.read_text())
    assert saved["backend"] == "mock"
    assert "mock backend" in saved["note"]
    assert "| p50" in md_path.read_text()
