from __future__ import annotations

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, ORJSONResponse

from app.runtime import get_parser, get_store, get_workflow
from app.schemas import AskRequest, AskResponse, ChatPayload


app = FastAPI(default_response_class=ORJSONResponse, title="Chat API")


@app.on_event("startup")
async def startup() -> None:
    await get_store().initialize()
    get_workflow()


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


@app.get("/greeting")
async def greeting() -> dict:
    return await get_workflow().greetings()


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
    history = [item.model_dump() for item in payload.history]
    result = await get_workflow().invoke(payload.message, history=history)
    return AskResponse(
        answer=result["answer"],
        intent=result.get("intent", "lightrag_qa"),
        node=result.get("node", result.get("intent", "lightrag_qa")),
        path=result.get("path", ["intent_classifier", result.get("intent", "lightrag_qa")]),
        contexts=result.get("contexts", []),
    )


@app.websocket("/chat")
async def chat(websocket: WebSocket) -> None:
    await websocket.accept()
    workflow = get_workflow()
    try:
        while True:
            payload = ChatPayload.model_validate_json(await websocket.receive_text())
            await websocket.send_json({"type": "status", "scope": "qa", "stage": "answering"})
            history = [item.model_dump() for item in payload.history]
            async for event in workflow.stream(payload.message, history=history):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        return
