"""API route tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from transformer_circuit_visualizer.api import create_app
from transformer_circuit_visualizer.model_service import MockModelService


def make_client() -> TestClient:
    return TestClient(create_app(model_service=MockModelService()))


def test_health_endpoint() -> None:
    client = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint() -> None:
    client = make_client()

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["models"] == ["mock-gpt2-small"]


def test_analyze_endpoint_with_mock_service() -> None:
    client = make_client()

    response = client.post(
        "/analyze",
        json={"prompt": "Hello circuits", "model_name": "mock-gpt2-small", "top_k": 3},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["tokens"] == ["<|bos|>", "Hello", "circuits"]
    assert body["token_ids"] == [0, 1001, 1002]
    assert len(body["final_token_predictions"]) == 3
    assert len(body["logit_lens"]) == 3
    assert body["attention"]["layer"] == 0
    assert body["attention"]["head"] == 0


def test_attention_endpoint_with_mock_service() -> None:
    client = make_client()

    response = client.post(
        "/attention",
        json={"prompt": "Hello circuits", "model_name": "mock-gpt2-small", "layer": 1, "head": 1},
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
        json={"prompt": "Hello", "model_name": "mock-gpt2-small", "layer": 99, "head": 0},
    )

    assert response.status_code == 400
    assert "out of range" in response.json()["detail"]
