import asyncio
import base64
import json
import os
from datetime import datetime, timezone
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from services.debug_service import save_audio_to_wav
from services.transcription_service import (
    STATUS_CONTINUE_FRAME,
    STATUS_FIRST_FRAME,
    STATUS_LAST_FRAME,
    TranscriptionService,
)
from services.translation_service import QwenTranslationService
from services.vad_service import VADService

load_dotenv()

app = FastAPI(
    title="Real-Time Transcription and Translation API",
    description="A WebSocket API to stream audio for real-time transcription (iFlyTek) and translation (Alibaba Qwen).",
)


# It defaults to "False" if the variable is not set.
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
print(f"  - Debug mode is: {'ON' if DEBUG_MODE else 'OFF'}")


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Viewer connected. Total viewers: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Viewer disconnected. Total viewers: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Broadcasts a message to all connected viewers."""
        if self.active_connections:
            tasks = [conn.send_text(message) for conn in self.active_connections]
            await asyncio.gather(*tasks)


transcription_manager = ConnectionManager()
translation_manager = ConnectionManager()


@app.websocket("/ws/view_transcription")
async def websocket_viewer_endpoint(websocket: WebSocket):
    await transcription_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        transcription_manager.disconnect(websocket)


@app.websocket("/ws/view_translation")
async def websocket_translation_viewer_endpoint(websocket: WebSocket):
    await translation_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        translation_manager.disconnect(websocket)


@app.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Transcription client connected.")

    loop = asyncio.get_running_loop()
    translation_service = QwenTranslationService()
    transcription_buffer = ""
    transcription_sentence_id = 0
    translation_sentence_id = 0
    transcription_service = None
    current_speaker = "Unknown"

    audio_frames = [] if DEBUG_MODE else None

    async def handle_translation(sentence_to_translate: str, speaker_name: str):
        nonlocal translation_sentence_id
        translation_sentence_id += 1
        current_translation_id = f"tr-{translation_sentence_id}"
        print(f"Translating for {speaker_name}: '{sentence_to_translate}'")
        full_translation = ""
        async for translated_chunk in translation_service.translate_stream(
            text_to_translate=sentence_to_translate
        ):
            full_translation = translated_chunk
            interim_message = {
                "type": "interim",
                "id": current_translation_id,
                "data": full_translation,
                "userName": speaker_name,
            }
            await translation_manager.broadcast(json.dumps(interim_message))
        final_message = {
            "type": "final",
            "id": current_translation_id,
            "data": full_translation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "userName": speaker_name,
        }
        await translation_manager.broadcast(json.dumps(final_message))
        print(f"Translation complete: '{full_translation}'")

    async def on_transcription_message(data: dict):
        nonlocal transcription_buffer, transcription_sentence_id
        result_data = data.get("result", {})
        is_new_utterance = not transcription_buffer and result_data.get("pgs") != "rpl"
        if is_new_utterance:
            transcription_sentence_id += 1
        current_text = "".join(
            cw.get("w", "") for w in result_data.get("ws", []) for cw in w.get("cw", [])
        )
        if result_data.get("pgs") == "rpl":
            full_sentence = transcription_buffer[: -len(current_text)] + current_text
        else:
            full_sentence = transcription_buffer + current_text
        interim_message = {
            "type": "interim",
            "id": f"t-{transcription_sentence_id}",
            "data": full_sentence,
            "userName": current_speaker,
        }
        await transcription_manager.broadcast(json.dumps(interim_message))
        transcription_buffer = full_sentence
        if result_data.get("ls"):
            final_chunk = transcription_buffer.strip()
            if final_chunk:
                print(
                    f"VAD-based final sentence detected for {current_speaker}: '{final_chunk}'"
                )
                final_message = {
                    "type": "final",
                    "id": f"t-{transcription_sentence_id}",
                    "data": final_chunk,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "userName": current_speaker,
                }
                await transcription_manager.broadcast(json.dumps(final_message))
                asyncio.create_task(handle_translation(final_chunk, current_speaker))
            transcription_buffer = ""

    async def on_service_error(error_message: str):
        print(f"Transcription Error: {error_message}")
        await websocket.close(code=1011, reason=f"Transcription Error: {error_message}")

    async def on_service_close():
        print("Transcription service connection closed as expected.")

    vad_service = VADService(aggressiveness=1, padding_duration_ms=550)

    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)

            current_speaker = message.get("userName", "Unknown")
            audio_chunk = base64.b64decode(message.get("audio"))

            if DEBUG_MODE:
                audio_frames.append(audio_chunk)

            for event, data in vad_service.process_audio(audio_chunk):
                if event == "start":
                    print(
                        f"VAD: Speech started for {current_speaker}. Creating new transcription session."
                    )
                    transcription_buffer = ""
                    transcription_service = TranscriptionService(
                        on_message_callback=on_transcription_message,
                        on_error_callback=on_service_error,
                        on_close_callback=on_service_close,
                        loop=loop,
                    )
                    transcription_service.connect()
                    transcription_service.send_audio(data, STATUS_FIRST_FRAME)
                elif event == "speech":
                    if transcription_service:
                        transcription_service.send_audio(data, STATUS_CONTINUE_FRAME)
                elif event == "end":
                    print("VAD: Speech ended. Closing transcription session.")
                    if transcription_service:
                        transcription_service.send_audio(b"", STATUS_LAST_FRAME)
                        transcription_service = None
    except WebSocketDisconnect:
        print("Transcription client disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred in transcribe endpoint: {e}")
    finally:
        if DEBUG_MODE:
            save_audio_to_wav(audio_frames)

        if transcription_service:
            transcription_service.close()
        print("Cleaned up transcription service.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
