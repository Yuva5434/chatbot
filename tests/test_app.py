from pathlib import Path
from types import SimpleNamespace

import app


class FakeOllamaClient:
    def chat(self, **kwargs):
        assert kwargs["model"] == "llama3.2"
        assert kwargs["stream"] is True
        assert kwargs["messages"][-1]["content"] == "Hello"
        yield SimpleNamespace(message=SimpleNamespace(content="Hi"))
        yield SimpleNamespace(message=SimpleNamespace(content=" there"))


def test_conversation_messages_persist_and_delete_cascades(tmp_path: Path):
    database_path = tmp_path / "history.db"
    app.initialize_database(database_path)
    conversation_id = app.create_conversation("Scratchpad", database_path)
    app.save_message(conversation_id, "user", "Hello", database_path)
    app.save_message(conversation_id, "assistant", "Hi there", database_path)

    assert app.list_conversations(database_path)[0]["title"] == "Scratchpad"
    assert app.get_messages(conversation_id, database_path) == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    with app.get_connection(database_path) as connection:
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    assert app.get_messages(conversation_id, database_path) == []


def test_ollama_reply_streams_text():
    messages = [{"role": "user", "content": "Hello"}]

    assert "".join(app.stream_reply(FakeOllamaClient(), "llama3.2", messages)) == "Hi there"


def test_model_name_handles_sdk_models_and_mappings():
    assert app.model_name(SimpleNamespace(model="llama3.2")) == "llama3.2"
    assert app.model_name({"name": "mistral"}) == "mistral"
