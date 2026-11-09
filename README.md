# Fieldnotes

A local-first Ollama chat app built with Python and Streamlit. SQLite stores conversation history on your machine.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- A downloaded Ollama model (the default is `llama3.2`)

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2
Copy-Item .env.example .env
streamlit run app.py
```

Streamlit opens the app in your browser. Ollama normally runs at `http://localhost:11434`; if needed, start it in another terminal with `ollama serve`.

Set `OLLAMA_MODEL` in `.env` to choose a default model and `OLLAMA_URL` to use a different Ollama host. The sidebar lists models available from Ollama and lets you switch models. `.env` is loaded automatically.

## Data and privacy

SQLite creates `data/chat_history.db` on first run. Conversation history remains on this machine and is not sent to any hosted API; prompts are sent to the Ollama server configured by `OLLAMA_URL`. The Ollama model and its own configuration determine inference behavior. Set `CHATBOT_DATA_DIR` to change where the database is stored. The database is excluded from Git.

## Development

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```
