"""API route tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from transformer_circuit_visualizer.api import create_app
from tests.fakes import FakeModelServiceForTests


def make_client() -> TestClient:
    return TestClient(create_app(model_service=FakeModelServiceForTests()))


def test_health_endpoint() -> None:
    client = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint() -> None:
    client = make_client()

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["models"] == ["fake-gpt2-small"]


def test_analyze_endpoint_with_fake_service() -> None:
    client = make_client()

    response = client.post(
        "/analyze",
        json={"prompt": "Hello circuits", "model_name": "fake-gpt2-small", "top_k": 3},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["tokens"] == ["<|bos|>", "Hello", "circuits"]
    assert body["token_ids"] == [0, 1001, 1002]
    assert len(body["final_token_predictions"]) == 3
    assert len(body["logit_lens"]) == 3
    assert body["attention"]["layer"] == 0
    assert body["attention"]["head"] == 0


def test_attention_endpoint_with_fake_service() -> None:
    client = make_client()

    response = client.post(
        "/attention",
        json={"prompt": "Hello circuits", "model_name": "fake-gpt2-small", "layer": 1, "head": 1},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["metadata"]["n_layers"] == 3
    assert body["attention"]["layer"] == 1
    assert body["attention"]["head"] == 1
    assert len(body["attention"]["pattern"]) == 3


def test_attention_endpoint_rejects_out_of_range_layer() -> None:
    client = make_client()

    response = client.post(
        "/attention",
        json={"prompt": "Hello", "model_name": "fake-gpt2-small", "layer": 99, "head": 0},
    )

    assert response.status_code == 400
    assert "out of range" in response.json()["detail"]


def test_heads_summary_endpoint_with_fake_service() -> None:
    client = make_client()

    response = client.post(
        "/heads/summary",
        json={"prompt": "Hello circuits", "model_name": "fake-gpt2-small"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["tokens"] == ["<|bos|>", "Hello", "circuits"]
    assert len(body["head_summaries"]) == 6
    assert body["head_summaries"][0]["layer"] == 0
    assert body["head_summaries"][0]["head"] == 0
    assert "average_attention_entropy" in body["head_summaries"][0]
    assert "output_norm" in body["head_summaries"][0]


def test_ablate_head_endpoint_with_fake_service() -> None:
    client = make_client()

    response = client.post(
        "/ablate/head",
        json={
            "prompt": "Hello circuits",
            "model_name": "fake-gpt2-small",
            "layer": 1,
            "head": 0,
            "top_k": 3,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["layer"] == 1
    assert body["head"] == 0
    assert len(body["before"]) == 3
    assert len(body["after"]) == 3
    assert len(body["deltas"]) == 3
    assert body["deltas"][0]["logit_delta"] < 0


def test_ablate_head_endpoint_rejects_out_of_range_head() -> None:
    client = make_client()

    response = client.post(
        "/ablate/head",
        json={
            "prompt": "Hello",
            "model_name": "fake-gpt2-small",
            "layer": 0,
            "head": 99,
            "top_k": 3,
        },
    )

    assert response.status_code == 400
    assert "out of range" in response.json()["detail"]
