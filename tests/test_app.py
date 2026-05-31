import pytest
from flask.testing import FlaskClient

from app.rag_pipeline import RAGResult


def test_health_endpoint(client: FlaskClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_chat_returns_answer_format(client: FlaskClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_answer_question(_question: str, _top_k: int = 4) -> RAGResult:
        return {
            "answer": "Mock policy answer.",
            "citations": [
                {
                    "document_title": "PTO Policy",
                    "file_name": "pto_policy.md",
                    "page": "",
                }
            ],
            "snippets": [
                {
                    "document_title": "PTO Policy",
                    "snippet": "Employees accrue PTO...",
                }
            ],
        }

    monkeypatch.setattr("app.app.answer_question", mock_answer_question)

    response = client.post("/chat", json={"question": "How does PTO work?"})
    data = response.get_json()

    assert response.status_code == 200
    assert data["answer"] == "Mock policy answer."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["document_title"] == "PTO Policy"
    assert isinstance(data["snippets"], list)


def test_chat_requires_question(client: FlaskClient) -> None:
    response = client.post("/chat", json={"question": ""})
    data = response.get_json()

    assert response.status_code == 400
    assert "answer" in data


def test_chat_rejects_missing_body(client: FlaskClient) -> None:
    response = client.post("/chat", data="not json", content_type="text/plain")

    assert response.status_code == 400
