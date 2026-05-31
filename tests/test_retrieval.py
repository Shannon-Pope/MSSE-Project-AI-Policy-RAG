import pytest
from unittest.mock import MagicMock

from app.rag_pipeline import retrieve_chunks


@pytest.fixture
def mock_collection(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    collection = MagicMock()
    monkeypatch.setattr("app.rag_pipeline.get_collection", lambda: collection)
    return collection


def test_retrieve_chunks_returns_correct_shape(mock_collection: MagicMock) -> None:
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "documents": [["Employees accrue PTO at 1.54 hours per pay period.", "All travel requires manager approval."]],
        "metadatas": [[
            {"document_title": "PTO Policy", "file_name": "pto.md", "page": ""},
            {"document_title": "Travel Policy", "file_name": "travel.md", "page": "2"},
        ]],
        "distances": [[0.12, 0.25]],
    }

    chunks = retrieve_chunks("How does PTO work?")

    assert len(chunks) == 2
    assert chunks[0]["document_title"] == "PTO Policy"
    assert chunks[0]["file_name"] == "pto.md"
    assert chunks[0]["page"] == ""
    assert "accrue PTO" in chunks[0]["text"]
    assert chunks[1]["page"] == "2"


def test_retrieve_chunks_empty_collection(mock_collection: MagicMock) -> None:
    mock_collection.count.return_value = 0

    chunks = retrieve_chunks("Any question")

    assert chunks == []
    mock_collection.query.assert_not_called()


def test_retrieve_chunks_caps_n_results_at_collection_size(mock_collection: MagicMock) -> None:
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "documents": [["Doc A", "Doc B"]],
        "metadatas": [[
            {"document_title": "A", "file_name": "a.md", "page": ""},
            {"document_title": "B", "file_name": "b.md", "page": ""},
        ]],
        "distances": [[0.1, 0.2]],
    }

    retrieve_chunks("question", top_k=10)

    assert mock_collection.query.call_args.kwargs["n_results"] == 2


def test_retrieve_chunks_handles_none_metadata(mock_collection: MagicMock) -> None:
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "documents": [["Some policy text."]],
        "metadatas": [[None]],
        "distances": [[0.1]],
    }

    chunks = retrieve_chunks("test question")

    assert len(chunks) == 1
    assert chunks[0]["document_title"] == ""
    assert chunks[0]["file_name"] == ""
    assert chunks[0]["page"] == ""


def test_retrieve_chunks_handles_none_results(mock_collection: MagicMock) -> None:
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "documents": None,
        "metadatas": None,
        "distances": None,
    }

    chunks = retrieve_chunks("test question")

    assert chunks == []
