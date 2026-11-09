import os
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ollama
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("CHATBOT_DATA_DIR", BASE_DIR / "data"))
DATABASE_PATH = DATA_DIR / "chat_history.db"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


@contextmanager
def get_connection(database_path: Path = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_path: Path = DATABASE_PATH) -> None:
    with get_connection(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New conversation',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
                ON messages(conversation_id, id);
            """
        )


def list_conversations(database_path: Path = DATABASE_PATH) -> list[dict[str, Any]]:
    with get_connection(database_path) as connection:
        rows = connection.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_messages(conversation_id: int, database_path: Path = DATABASE_PATH) -> list[dict[str, Any]]:
    with get_connection(database_path) as connection:
        rows = connection.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_message(
    conversation_id: int,
    role: str,
    content: str,
    database_path: Path = DATABASE_PATH,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(database_path) as connection:
        connection.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )


def create_conversation(title: str, database_path: Path = DATABASE_PATH) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title[:48] or "New conversation", now, now),
        )
        return int(cursor.lastrowid)


def stream_reply(
    client: Any,
    model: str,
    messages: Sequence[Mapping[str, str]],
) -> Iterator[str]:
    for chunk in client.chat(model=model, messages=messages, stream=True):
        content = chunk.message.content
        if content:
            yield content


@st.cache_resource
def get_ollama_client(host: str) -> ollama.Client:
    return ollama.Client(host=host)


def model_name(model: Any) -> str:
    if isinstance(model, Mapping):
        return str(model.get("model") or model.get("name") or "")
    return str(getattr(model, "model", "") or getattr(model, "name", ""))


def main() -> None:
    st.set_page_config(page_title="Nova Companion", page_icon="◈", layout="centered")
    initialize_database()

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        :root {
            --bg: #02070d;
            --bg-2: #071822;
            --panel: rgba(10, 19, 29, 0.96);
            --panel-strong: rgba(14, 27, 39, 0.98);
            --sidebar: linear-gradient(180deg, #05131b 0%, #0a1824 100%);
            --line: rgba(103, 214, 255, 0.15);
            --ink: #edf7ff;
            --muted: #9ebad4;
            --blue: #67d6ff;
            --blue-strong: #3b82f6;
            --blue-deep: #173b9d;
            --chip: rgba(103, 214, 255, 0.08);
            --shadow: 0 24px 52px rgba(0, 0, 0, 0.46);
        }
        * { box-sizing: border-box; }
        html, body, [class*="st-"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 12% 12%, rgba(59, 130, 246, 0.18), transparent 20%),
                radial-gradient(circle at 88% 18%, rgba(103, 214, 255, 0.12), transparent 18%),
                linear-gradient(135deg, var(--bg) 0%, var(--bg-2) 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1280px !important;
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid var(--line);
            box-shadow: inset -1px 0 0 rgba(255,255,255,0.03);
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.32rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.04em !important;
            color: var(--ink) !important;
        }
        .eyebrow {
            margin: 0 0 12px;
            color: var(--blue);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }
        .page-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 1rem;
            padding-top: 0.2rem;
        }
        .header-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.4rem 0.75rem;
            background: var(--chip);
            border: 1px solid rgba(103, 214, 255, 0.2);
            color: var(--blue);
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
            box-shadow: 0 0 18px rgba(103, 214, 255, 0.08);
        }
        .welcome-panel {
            background: linear-gradient(180deg, rgba(15, 24, 34, 0.98), rgba(11, 19, 27, 0.96));
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.5rem 1.5rem 1.2rem;
            margin: 1rem 0 1.5rem;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }
        .welcome-panel::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(103,214,255,0.08), transparent 35%, rgba(59,130,246,0.08));
            pointer-events: none;
        }
        .welcome-panel h2 {
            position: relative;
            font-size: 2.25rem !important;
            line-height: 1.06 !important;
            margin: 0.15rem 0 0.5rem !important;
            color: var(--ink) !important;
            letter-spacing: -0.05em !important;
        }
        .welcome-copy {
            position: relative;
            color: var(--muted);
            margin: 0;
            font-size: 0.96rem;
            line-height: 1.6;
        }
        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(103, 214, 255, 0.18);
            background: linear-gradient(180deg, rgba(16, 26, 38, 1), rgba(9, 17, 25, 1));
            color: var(--ink);
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }
        .stButton > button:hover {
            border-color: rgba(103, 214, 255, 0.3);
            transform: translateY(-1px);
            box-shadow: 0 12px 20px rgba(59,130,246,0.08);
        }
        .stChatInput textarea {
            border-radius: 16px !important;
            border: 1px solid rgba(103, 214, 255, 0.18) !important;
            background: rgba(8, 17, 25, 0.9) !important;
            color: var(--ink) !important;
            font-size: 1rem !important;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }
        [data-testid="stChatMessage"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(12, 20, 29, 0.96);
            padding: 0.8rem 0.9rem;
            box-shadow: 0 10px 20px rgba(0,0,0,0.16);
        }
        [data-testid="stChatMessage"] p {
            color: var(--ink);
            margin: 0.2rem 0;
        }
        [data-testid="stStatusWidget"] { visibility: hidden; }
        .stAlert { border-radius: 12px; }
        .agent-bot {
            width: 56px;
            height: 56px;
            border-radius: 18px;
            object-fit: cover;
            border: 1px solid rgba(103,214,255,0.24);
            background: rgba(17,28,39,1);
            box-shadow: 0 12px 24px rgba(0,0,0,0.2);
            display: block;
            position: relative;
            z-index: 1;
        }
        .stSidebar .stButton > button {
            font-size: 0.92rem !important;
            padding: 0.7rem 0.8rem !important;
        }
        @media (max-width: 640px) {
            h1 { font-size: 2.5rem !important; }
            .welcome-panel { padding: 1.1rem; }
            .page-header { flex-direction: column; align-items: flex-start; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    client = get_ollama_client(OLLAMA_URL)
    available_models: list[str] = []
    ollama_error: str | None = None
    try:
        available_models = [model_name(model) for model in client.list().models]
        available_models = [name for name in available_models if name]
    except Exception as error:
        ollama_error = str(error)

    with st.sidebar:
        st.markdown("# ◈ Nova")
        st.caption("Companion AI workspace")
        if st.button("＋  New conversation", use_container_width=True, type="primary"):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("YOUR THREADS")
        for conversation in list_conversations():
            conversation_id = conversation["id"]
            is_active = conversation_id == st.session_state.conversation_id
            label = str(conversation["title"])
            if st.button(
                f"{'▸ ' if is_active else ''}{label}",
                key=f"conversation-{conversation_id}",
                use_container_width=True,
                type="secondary" if is_active else "tertiary",
            ):
                st.session_state.conversation_id = conversation_id
                st.session_state.messages = get_messages(conversation_id)
                st.rerun()
        st.divider()

        if available_models:
            selected_default = DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0]
            model = st.selectbox("OLLAMA MODEL", available_models, index=available_models.index(selected_default))
        else:
            model = None
            st.caption("Model: not available")
        if ollama_error:
            st.warning("Ollama is offline. Start Ollama and make sure a model is available.")
        elif not available_models:
            st.warning(
                "Ollama is ready, but no models are installed. Run `ollama pull <model-name>` "
                "or set OLLAMA_MODEL to an installed model."
            )
        else:
            st.caption(f"● Connected · {OLLAMA_URL}")
        st.caption("Conversation history is stored locally in SQLite.")

    st.markdown(
        '<div class="page-header"><p class="eyebrow">AI AGENT CONVERSATION</p><span class="header-pill">LIVE</span></div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.messages:
        welcome_model = model or "your model"
        st.markdown(
            f"""
            <div class="welcome-panel">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                    <img
                        class="agent-bot"
                        src="data:image/svg+xml;utf8,
                        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120'>
                            <defs>
                                <linearGradient id='g' x1='0' x2='1'>
                                    <stop offset='0%' stop-color='#dfeeff'/>
                                    <stop offset='100%' stop-color='#8bb7ff'/>
                                </linearGradient>
                            </defs>
                            <rect width='120' height='120' rx='24' fill='#12161b'/>
                            <rect x='24' y='24' width='72' height='72' rx='18' fill='url(#g)' stroke='#b7d7ff' stroke-width='3'/>
                            <rect x='39' y='44' width='42' height='18' rx='9' fill='#12161b'/>
                            <circle cx='45' cy='58' r='5' fill='#dfeeff'/>
                            <circle cx='75' cy='58' r='5' fill='#dfeeff'/>
                            <rect x='50' y='72' width='20' height='6' rx='3' fill='#12161b'/>
                            <path d='M38 30 L52 18 L68 30' fill='none' stroke='#dfeeff' stroke-width='5' stroke-linecap='round'/>
                            <circle cx='92' cy='90' r='8' fill='#9dc4ff'/>
                        </svg>"
                        alt="AI Assistant"
                    />
                    <span style="font-size:0.8rem; color:#9dc4ff; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">Agent Ready</span>
                </div>
                <h2>Let’s build<br>the next step <span style='color:#9dc4ff'><em>together.</em></span></h2>
                <p class="welcome-copy">A local AI assistant, powered by <strong>{welcome_model}</strong>. Tell me what you want to plan, fix, or create.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        suggestions = [
            "Help me break a big task into a clear first step.",
            "Explain a difficult idea to me with a simple example.",
            "Help me draft a thoughtful message about a difficult topic.",
        ]
        columns = st.columns(3)
        for column, prompt in zip(columns, suggestions):
            with column:
                if st.button(prompt, key=f"suggestion-{prompt}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Write what’s on your mind...", max_chars=12000)
    if prompt is None:
        prompt = st.session_state.pop("pending_prompt", None)
    if prompt:
        if model is None:
            st.warning("No Ollama model is available yet. Install one with `ollama pull <model-name>` and reload the app.")
            st.stop()
        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = create_conversation(prompt)
        conversation_id = st.session_state.conversation_id
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        save_message(conversation_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(stream_reply(client, model, st.session_state.messages))
                assistant_message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(assistant_message)
                save_message(conversation_id, "assistant", answer)
            except Exception as error:
                st.error(f"Could not get a reply from Ollama: {error}")

    st.caption("Local models can make mistakes. Check anything that matters.")


if __name__ == "__main__":
    main()
