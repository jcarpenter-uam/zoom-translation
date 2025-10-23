import os
import sys

import uvicorn
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from services.cache_service import TranscriptCache
from services.connection_manager_service import ConnectionManager
from services.debug_service import log_pipeline_step
from services.rag_service import dbconnect
from services.rtms_receiver_service import create_transcribe_router

log_pipeline_step("SYSTEM", "Initializing database connection...", detailed=False)
db_engine = dbconnect()

if db_engine:
    log_pipeline_step(
        "SYSTEM", "Database setup successful. Engine is ready.", detailed=False
    )
else:
    sys.exit("Database connection failed. Application cannot start.")

app = FastAPI(
    title="Real-Time Transcription and Translation API",
    description="A WebSocket API to stream audio for real-time transcription and translation.",
)

DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
log_pipeline_step(
    "SYSTEM",
    f"Debug mode is: {'ON' if DEBUG_MODE else 'OFF'}",
    detailed=False,
)

transcript_cache = TranscriptCache()
viewer_manager = ConnectionManager(cache=transcript_cache)

rtms_router = create_transcribe_router(
    viewer_manager=viewer_manager,
    DEBUG_MODE=DEBUG_MODE,
)
app.include_router(rtms_router)


@app.websocket("/ws/view_transcript")
async def websocket_viewer_endpoint(websocket: WebSocket):
    await viewer_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        viewer_manager.disconnect(websocket)


app.mount("/", StaticFiles(directory="web/dist", html=True), name="web")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
