import json
import os

from dotenv import load_dotenv
import requests
import streamlit as st
import websockets.sync.client
from websockets.exceptions import ConnectionClosedError


load_dotenv()

API_HTTP_URL = os.getenv("API_HTTP_URL", "http://localhost:8000")
CHAT_WS_URL = os.getenv("CHAT_WS_URL", "ws://localhost:8000/chat")

st.set_page_config(page_title="Chat", page_icon=":material/chat:", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp, .main {
        background:
            radial-gradient(circle at top, rgba(66, 92, 154, 0.18), transparent 24%),
            linear-gradient(180deg, #0a0d12 0%, #0d1118 50%, #090c12 100%) !important;
        color: #e8edf7 !important;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbar"] {
        right: 1rem;
    }
    [data-testid="stAppViewContainer"] > .main {
        min-height: 100vh;
    }
    .block-container {
        max-width: 1080px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    .shell {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 1.5rem;
        width: 100%;
    }
    .title {
        font-size: 4rem;
        line-height: 0.95;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0;
        color: #f4f7fb;
    }
    .subtitle {
        max-width: 44rem;
        color: #96a3b8;
        font-size: 1.02rem;
        line-height: 1.7;
        margin: 0 auto;
    }
    .status-row {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        border-radius: 999px;
        padding: 0.48rem 0.9rem;
        background: rgba(18, 24, 35, 0.92);
        border: 1px solid rgba(135, 149, 175, 0.14);
        color: #b1bdcf;
        font-size: 0.92rem;
    }
    .status-dot {
        width: 0.55rem;
        height: 0.55rem;
        border-radius: 999px;
        background: #4ade80;
        box-shadow: 0 0 0 0.2rem rgba(74, 222, 128, 0.12);
    }
    .chat-wrap, .graph-wrap {
        width: min(100%, 1020px);
        border: 1px solid rgba(135, 149, 175, 0.12);
        background: rgba(16, 21, 31, 0.88);
        border-radius: 26px;
        box-shadow: 0 28px 80px rgba(0, 0, 0, 0.34);
    }
    .chat-wrap {
        padding: 0.9rem;
        text-align: left;
    }
    .graph-wrap {
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        text-align: left;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #eef2f8;
        margin-bottom: 0.9rem;
    }
    .stChatMessage {
        border: 1px solid rgba(135, 149, 175, 0.09);
        background: rgba(21, 28, 40, 0.94);
        border-radius: 18px;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] {
        color: #e7edf8;
    }
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(180deg, #5c83ff 0%, #3453d4 100%);
    }
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(180deg, #1d2433 0%, #0f141d 100%);
    }
    [data-testid="stChatInput"] {
        background: rgba(12, 17, 25, 0.98);
        border: 1px solid rgba(135, 149, 175, 0.14);
        border-radius: 18px;
    }
    [data-testid="stChatInput"] textarea {
        color: #f4f7fb !important;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        border: 1px solid rgba(135, 149, 175, 0.1);
        background: rgba(21, 28, 40, 0.92);
        border-radius: 18px;
        padding: 0.9rem 1rem;
    }
    .metric-label {
        color: #91a0b7;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.45rem;
        font-weight: 700;
    }
    .metric-value {
        color: #f3f6fb;
        font-size: 1.4rem;
        font-weight: 800;
    }
    @media (max-width: 900px) {
        .title {
            font-size: 2.7rem;
        }
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_json(path: str) -> dict:
    try:
        response = requests.get(f"{API_HTTP_URL}{path}", timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


if "messages" not in st.session_state:
    st.session_state.messages = []


health = get_json("/health")
backend_ready = health.get("status") == "ok"
kb_ready = bool(health.get("ingested"))

if backend_ready and not st.session_state.messages:
    greeting = get_json("/greeting")
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": greeting.get("answer", "Welcome. Ask a question about your documents."),
        }
    ]

st.markdown('<div class="shell">', unsafe_allow_html=True)
st.markdown('<h1 class="title">Chat with your documents</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="subtitle">
    Ask grounded questions over your loaded document knowledge base.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="status-row">
      <div class="status-pill">
        <span class="status-dot" style="background: {'#4ade80' if backend_ready else '#f97316'};"></span>
        Backend {'connected' if backend_ready else 'unavailable'}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Ask about methods, datasets, metrics, problems, or solutions", disabled=not backend_ready)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        answer_placeholder = st.empty()
        answer_text = ""
        try:
            with websockets.sync.client.connect(
                CHAT_WS_URL,
                open_timeout=None,
                close_timeout=10,
            ) as ws:
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in st.session_state.messages[:-1]
                ]
                ws.send(json.dumps({"message": prompt, "history": history}))
                while True:
                    payload = json.loads(ws.recv())
                    payload_type = payload.get("type")
                    if payload_type == "status":
                        status.info("Generating answer...")
                        continue
                    if payload_type == "token":
                        status.empty()
                        answer_text += payload.get("token", "")
                        answer_placeholder.markdown(answer_text)
                        continue
                    if payload_type == "error":
                        status.empty()
                        answer_placeholder.empty()
                        st.error(payload["message"])
                        st.stop()
                    if payload_type == "answer":
                        status.empty()
                        answer_text = payload["answer"]
                        answer_placeholder.markdown(answer_text)
                        with st.expander("Retrieved context"):
                            st.json(payload["contexts"])
                        st.session_state.messages.append({"role": "assistant", "content": answer_text})
                        break
        except ConnectionClosedError:
            status.empty()
            if answer_text:
                answer_placeholder.markdown(answer_text)
            st.error("The websocket connection closed unexpectedly. Check whether the API container is still running.")
        except Exception as exc:
            status.empty()
            if answer_text:
                answer_placeholder.markdown(answer_text)
            st.error(str(exc))
st.markdown("</div>", unsafe_allow_html=True)

if backend_ready and not kb_ready:
    st.warning("The backend is online, but the knowledge base is not ready yet.")

st.markdown("</div>", unsafe_allow_html=True)
