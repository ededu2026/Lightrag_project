# LightRAG Q&A

FastAPI Q&A service built on the official LightRAG pipeline, a `/chat` websocket, and a Streamlit UI.

## Stack

- Stella (`dunzhang/stella_en_400M_v5`) for embeddings
- Ollama with `qwen2.5:3b` for the final answer
- LightRAG with `NetworkXStorage`
- FastAPI for the API
- Streamlit for the frontend

## Recommended Run

On Apple Silicon, prefer running the backend and Ollama natively on macOS to use local hardware acceleration. Use Docker mainly as a compatibility fallback.

## Apple Silicon Run

1. Install Ollama on macOS and pull the model:

```bash
ollama pull qwen2.5:3b
```

2. Create the virtual environment and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Start the backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. In another terminal, start the frontend:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Then open:

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## Docker Run

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- UI: `http://localhost:8501`
- Ollama: `http://localhost:11434`

The Streamlit UI is chat-only. Parsing and ingestion must be triggered through backend endpoints. `POST /ingest` ingests only existing markdown files and does not run parsing implicitly.

## Endpoints

- `GET /health`
- `POST /parse`
- `GET /parse-status`
- `POST /ingest`
- `GET /ingest-status`
- `GET /graph/export/graphml`
- `GET /graph/export/json`
- `POST /ask`
- `WS /chat`

## Local Guide

Detailed local setup and usage are in [LOCAL_USAGE.md](/Users/edson/Documents/Codex_teste/LOCAL_USAGE.md).
