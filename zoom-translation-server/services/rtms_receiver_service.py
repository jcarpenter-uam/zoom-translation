import asyncio
import base64
import json
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.engine import Engine

from .audio_processing_service import AudioProcessingService
from .correction_service import CorrectionService
from .debug_service import (
    log_pipeline_step,
    log_utterance_end,
    log_utterance_start,
    save_audio_to_wav,
)
from .rag_service import RagService
from .soniox_service import SonioxResult, SonioxService


def create_transcribe_router(viewer_manager, DEBUG_MODE, db_engine: Engine):
    router = APIRouter()

    @router.websocket("/ws/transcribe")
    async def websocket_transcribe_endpoint(websocket: WebSocket):
        await websocket.accept()
        log_pipeline_step("SESSION", "Transcription client connected.", detailed=False)

        loop = asyncio.get_running_loop()
        audio_processor = AudioProcessingService()
        transcription_service = None
        correction_service = None

        try:
            ollama_url = os.environ["OLLAMA_URL"]
            log_pipeline_step(
                "SYSTEM",
                f"Ollama Correction Service URL: {ollama_url}",
                detailed=False,
            )

            rag_service = RagService(ollama_url=ollama_url, engine=db_engine)
            log_pipeline_step("SYSTEM", "RAG Service initialized and connected to DB.")

            correction_service = CorrectionService(
                ollama_url=ollama_url,
                viewer_manager=viewer_manager,
                rag_service=rag_service,
            )
        except KeyError:
            log_pipeline_step(
                "SYSTEM",
                "WARNING: The 'OLLAMA_URL' environment variable is not set. Contextual corrections will be disabled.",
                detailed=False,
            )

        current_message_id = None
        is_new_utterance = True
        current_speaker = "Unknown"

        session_debug_dir = None
        if DEBUG_MODE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_debug_dir = os.path.join("debug", timestamp)
            session_raw_audio = []
            session_processed = []
            log_pipeline_step(
                "SESSION",
                "Debug session directory initialized.",
                extra={"path": session_debug_dir},
                detailed=True,
            )

        async def on_service_error(error_message: str):
            log_pipeline_step(
                "SONIOX",
                f"Transcription Error: {error_message}",
                detailed=False,
            )
            await websocket.close(
                code=1011, reason=f"Transcription Error: {error_message}"
            )

        async def on_transcription_message_local(
            result: SonioxResult,
        ):
            """
            Handles all T&T results from the single Soniox service.
            Manages message_id state based on is_final.
            """
            nonlocal current_message_id, is_new_utterance, current_speaker

            if is_new_utterance and not result.is_final:
                current_message_id = str(uuid.uuid4())
                is_new_utterance = False
                log_utterance_start(current_message_id, current_speaker)

            if not current_message_id:
                log_pipeline_step(
                    "SONIOX",
                    "Received result with no active utterance, dropping.",
                    extra={"is_final": result.is_final},
                    detailed=True,
                )
                if result.is_final:
                    is_new_utterance = True
                return

            log_pipeline_step(
                "SONIOX",
                "Received consolidated T&T chunk.",
                speaker=current_speaker,
                extra={
                    "message_id": current_message_id,
                    "is_final": result.is_final,
                    "transcription": result.transcription,
                    "translation": result.translation,
                },
                detailed=True,
            )

            payload_type = "final" if result.is_final else "partial"
            payload = {
                "message_id": current_message_id,
                "transcription": result.transcription,
                "translation": result.translation,
                "speaker": current_speaker,
                "type": payload_type,
                "isfinalize": result.is_final,
            }
            await viewer_manager.broadcast(payload)

            if result.is_final:
                log_utterance_end(current_message_id, current_speaker)

                if (
                    correction_service
                    and result.transcription
                    and result.transcription.strip()
                ):
                    utterance_to_store = {
                        "message_id": current_message_id,
                        "speaker": current_speaker,
                        "transcription": result.transcription,
                        "translation": result.translation,
                    }
                    asyncio.create_task(
                        correction_service.process_final_utterance(utterance_to_store)
                    )

                is_new_utterance = True
                current_message_id = None

        async def on_service_close_local(code: int, reason: str):
            """
            Handles the Soniox service closing unexpectedly.
            """
            log_pipeline_step(
                "SONIOX",
                f"Transcription service closed. Code: {code}, Reason: {reason}",
                detailed=False,
            )

        try:
            transcription_service = SonioxService(
                on_message_callback=on_transcription_message_local,
                on_error_callback=on_service_error,
                on_close_callback=on_service_close_local,
                loop=loop,
            )
            await loop.run_in_executor(None, transcription_service.connect)
            log_pipeline_step(
                "SESSION",
                "Transcription service connected for session.",
                detailed=True,
            )

            while True:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)
                current_speaker = message.get("userName", "Unknown")
                audio_chunk = base64.b64decode(message.get("audio"))
                # log_pipeline_step(
                #     "SESSION",
                #     "Received audio chunk from client.",
                #     extra={
                #         "speaker": current_speaker,
                #         "chunk_bytes": len(audio_chunk),
                #     },
                #     detailed=True,
                # )

                if DEBUG_MODE:
                    session_raw_audio.append(audio_chunk)

                processed_audio = audio_processor.process(audio_chunk)

                if DEBUG_MODE:
                    session_processed.append(processed_audio)

                # BUG: Send processed_audio once new noise filtering is emplemented
                await loop.run_in_executor(
                    None, transcription_service.send_chunk, audio_chunk
                )

        except WebSocketDisconnect:
            log_pipeline_step(
                "SESSION",
                "Transcription client disconnected.",
                detailed=False,
            )
        except Exception as e:
            log_pipeline_step(
                "SESSION",
                f"An unexpected error occurred in transcribe endpoint: {e}",
                detailed=False,
            )
        finally:
            if transcription_service:
                log_pipeline_step(
                    "SESSION",
                    "Client disconnected. Finalizing Soniox stream...",
                    detailed=True,
                )
                await loop.run_in_executor(None, transcription_service.finalize_stream)

            await asyncio.sleep(0.1)

            if correction_service:
                log_pipeline_step(
                    "SESSION",
                    "Running final correction check on remaining utterances.",
                    detailed=False,
                )
                await correction_service.finalize_session()

            if DEBUG_MODE and session_debug_dir:
                log_pipeline_step(
                    "SESSION",
                    "Session ended. Saving debug audio files...",
                    detailed=False,
                )
                save_audio_to_wav(
                    session_raw_audio,
                    session_debug_dir,
                    "raw_audio.wav",
                )
                save_audio_to_wav(
                    session_processed,
                    session_debug_dir,
                    "processed.wav",
                )

            try:
                output_dir = "session_history"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"history_{timestamp}.json"

                os.makedirs(output_dir, exist_ok=True)
                cache_filepath = os.path.join(output_dir, file_name)

                transcript_history = viewer_manager.cache.get_history()
                if transcript_history:
                    with open(cache_filepath, "w", encoding="utf-8") as f:
                        json.dump(transcript_history, f, indent=4, ensure_ascii=False)

                    log_pipeline_step(
                        "SESSION",
                        "Transcript cache saved to file successfully.",
                        extra={
                            "path": cache_filepath,
                            "entries": len(transcript_history),
                        },
                        detailed=True,
                    )

                viewer_manager.cache.clear()

            except Exception as e:
                log_pipeline_step(
                    "SESSION",
                    f"Failed to save history or clear transcript cache: {e}",
                    detailed=False,
                )

    return router
