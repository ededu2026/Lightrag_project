from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, ORJSONResponse

from app.runtime import get_parser, get_store
from app.schemas import AskRequest, AskResponse, ChatPayload


app = FastAPI(default_response_class=ORJSONResponse, title="Chat API")


@app.on_event("startup")
async def startup() -> None:
    await get_store().initialize()


@app.on_event("shutdown")
async def shutdown() -> None:
    await get_store().finalize()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "parsed_dir": str(get_parser().output_dir),
        "ingested": get_store().ready,
    }


@app.post("/parse")
def parse() -> dict:
    result = get_parser().parse_all()
    return {
        "files": result.files,
        "markdowns": result.markdowns,
        "images": result.images,
        "output_dir": str(get_parser().output_dir),
    }


@app.get("/parse-status")
def parse_status() -> dict:
    return get_parser().get_progress()


@app.get("/ingest-status")
def ingest_status() -> dict:
    return get_store().get_progress()


@app.get("/graph/export/{format_name}")
def graph_export(format_name: str) -> FileResponse:
    store = get_store()
    if not store.ready:
        raise HTTPException(status_code=400, detail="Knowledge graph is not ingested yet.")
    if format_name == "graphml":
        path = store.export_graphml_path()
        return FileResponse(path=path, media_type="application/xml", filename=path.name)
    if format_name == "json":
        path = store.export_json_path()
        return FileResponse(path=path, media_type="application/json", filename=path.name)
    raise HTTPException(status_code=404, detail="Unsupported export format.")


@app.post("/ingest")
async def ingest() -> dict:
    parser = get_parser()
    if not any(parser.output_dir.glob("*.md")):
        raise HTTPException(status_code=400, detail="No parsed markdown files found. Run POST /parse first.")
    return await get_store().ingest()


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    result = await get_store().query(payload.message)
    return AskResponse(
        answer=result["answer"],
        intent=result.get("mode", "mix"),
        contexts=result.get("contexts", []),
    )


@app.websocket("/chat")
async def chat(websocket: WebSocket) -> None:
    await websocket.accept()
    store = get_store()
    try:
        while True:
            payload = ChatPayload.model_validate_json(await websocket.receive_text())
            if not store.ready:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Knowledge base is not ingested yet. Run Parse and Ingest first.",
                    }
                )
                continue
            await websocket.send_json({"type": "status", "scope": "qa", "stage": "answering"})
            result = await store.query(payload.message)
            await websocket.send_json(
                {
                    "type": "answer",
                    "answer": result["answer"],
                    "intent": result.get("mode", "mix"),
                    "contexts": result.get("contexts", []),
                }
            )
    except WebSocketDisconnect:
        return
