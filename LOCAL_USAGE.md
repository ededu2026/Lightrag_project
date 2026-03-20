# Local Usage

This project runs best on Apple Silicon outside Docker.

## Apple Silicon

Recommended for Mac Mini M4.

1. Install local dependencies:

```bash
brew install tesseract
brew install --cask ollama
ollama pull qwen2.5:3b
```

2. Check [.env](/Users/edson/Documents/Codex_teste/.env):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
LIGHTRAG_WORKING_DIR=rag_storage
LIGHTRAG_QUERY_MODE=mix
LLM_TIMEOUT=1800
MAX_ASYNC=1
EMBEDDING_TIMEOUT=600
EMBEDDING_FUNC_MAX_ASYNC=1
EMBEDDING_BATCH_NUM=2
STELLA_MODEL_NAME=dunzhang/stella_en_400M_v5
STELLA_DEVICE=auto
STELLA_BATCH_SIZE=4
RAW_DATA_DIR=data
DATA_DIR=parsed_data
PARSED_ASSETS_DIR=parsed_assets
ENABLE_IMAGE_OCR=true
ENABLE_PAGE_OCR_FALLBACK=false
```

`STELLA_DEVICE=auto` prefers `mps` on macOS and falls back to `cpu`.

3. Create the virtual environment and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Run the backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5. In another terminal, run the frontend:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Open:

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

## Docker

Use Docker only if you need an isolated environment. On macOS, it is slower for ingestion because the heavy work stays inside Linux containers.

```bash
docker compose up --build
```

In Docker, the compose file overrides:

- `OLLAMA_BASE_URL=http://ollama:11434`
- `STELLA_DEVICE=cpu`

## Endpoints

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/parse
curl http://localhost:8000/parse-status
curl -X POST http://localhost:8000/ingest
curl http://localhost:8000/ingest-status
curl http://localhost:8000/graph/export/graphml -o knowledge_graph.graphml
curl http://localhost:8000/graph/export/json -o knowledge_graph.json
```

## Notes

- Parse and ingest progress are shown in the API terminal.
- `POST /ingest` uses the official LightRAG pipeline.
- If ingestion is still slow, reduce `STELLA_BATCH_SIZE` to `2`.
- If LightRAG times out during entity extraction, increase `LLM_TIMEOUT`.
