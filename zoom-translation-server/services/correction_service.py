import asyncio
import json
import os
from collections import deque
from typing import AsyncGenerator

import ollama
from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI

from .debug_service import log_pipeline_step, log_utterance_step
from .rag_service import RagService


class RetranslationService:
    """
    Handles retranslation by making REST API calls to Alibaba's Qwen model
    via the DashScope service in an OpenAI-compatible mode.
    """

    def __init__(self):
        load_dotenv()
        try:
            self.client = AsyncOpenAI(
                api_key=os.environ["ALIBABA_API_KEY"],
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
        except KeyError:
            raise ValueError("The 'ALIBABA_API_KEY' environment variable is not set.")
        log_pipeline_step(
            "RETRANSLATION",
            "Initialized Qwen retranslation client.",
            extra={
                "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            },
            detailed=True,
        )

    async def translate_stream(
        self, text_to_translate: str
    ) -> AsyncGenerator[str, None]:
        """
        Retranslates a block of text using the Qwen model and streams the results.
        """
        user_prompt = (
            "You are a Chinese-to-English translator. Your task is to translate the text in the [TEXT TO TRANSLATE] section. "
            "Your response must contain ONLY the English translation of the [TEXT TO TRANSLATE] and nothing else. Do not include any other text in your response."
        )

        user_prompt += f"\n\n[TEXT TO TRANSLATE]\n{text_to_translate}"

        log_pipeline_step(
            "RETRANSLATION",
            "Submitting text for retranslation.",
            extra={"characters": len(text_to_translate)},
            detailed=False,
        )

        try:
            stream = await self.client.chat.completions.create(
                model="qwen-mt-turbo",
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                extra_body={
                    "translation_options": {
                        "source_lang": "zh",
                        "target_lang": "en",
                    }
                },
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    log_pipeline_step(
                        "RETRANSLATION",
                        "Received retranslation delta from Qwen.",
                        extra={
                            "delta_length": len(content),
                            "has_more": not chunk.choices[0].finish_reason,
                        },
                        detailed=True,
                    )
                    yield content
            log_pipeline_step(
                "RETRANSLATION",
                "Retranslation stream completed successfully.",
                detailed=False,
            )
        except APIError as e:
            error_message = f"[Translation Error: {e.message}]"
            log_pipeline_step(
                "RETRANSLATION",
                f"Alibaba Qwen API error: {e}",
                detailed=False,
            )
            yield error_message
        except Exception as e:
            log_pipeline_step(
                "RETRANSLATION",
                f"An unexpected error occurred during retranslation: {e}",
                detailed=False,
            )
            yield "[Translation Error]"


class CorrectionService:
    """
    Handles transcription correction using a local Ollama model.
    This service is stateful and manages the utterance history.
    """

    def __init__(
        self,
        ollama_url: str,
        viewer_manager,
        rag_service: RagService,
        model: str = "correction",
    ):
        """
        Initializes the service.

        Args:
            ollama_url (str): The base URL for the Ollama API.
            viewer_manager: The WebSocket manager for broadcasting updates.
            rag_service (RagService): The service for logging to the RAG DB.
            model (str): The name of the model to use for corrections.
        """
        log_pipeline_step(
            "CORRECTION",
            f"Initializing Stateful Correction Service with model '{model}' at {ollama_url}...",
            detailed=False,
        )
        self.model = model
        self.client = ollama.AsyncClient(host=ollama_url)
        self.viewer_manager = viewer_manager
        self.retranslation_service = RetranslationService()
        self.rag_service = rag_service
        self.utterance_history = deque(maxlen=5)
        self.CORRECTION_CONTEXT_THRESHOLD = 5

        log_pipeline_step(
            "CORRECTION",
            "Ollama correction client initialized.",
            extra={
                "model": model,
                "host": ollama_url,
                "rag_enabled": True,
            },
            detailed=True,
        )

    async def correct_with_context(
        self, text_to_correct: str, context_history: list[str]
    ) -> dict:
        """
        Sends a transcript to the custom correction model and returns the parsed JSON response.
        """
        prompt_data = {
            "context": " ".join(context_history),
            "target_sentence": text_to_correct,
        }
        prompt = json.dumps(prompt_data, ensure_ascii=False)
        response_content = ""
        log_pipeline_step(
            "CORRECTION",
            f"Sending prompt to Ollama: {prompt}",
            detailed=False,
        )

        try:
            response = await self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            response_content = response["message"]["content"]
            log_pipeline_step(
                "CORRECTION",
                "Received raw correction response.",
                extra={"response_length": len(response_content)},
                detailed=True,
            )

            json_start_index = response_content.find("{")
            json_end_index = response_content.rfind("}")

            if json_start_index != -1:
                if json_end_index != -1 and json_end_index > json_start_index:
                    json_string = response_content[
                        json_start_index : json_end_index + 1
                    ]
                else:
                    json_string = response_content[json_start_index:] + "}"

                response_data = json.loads(json_string)
                log_pipeline_step(
                    "CORRECTION",
                    f"Parsed correction response JSON. '{response_data}'",
                    extra={
                        "is_correction_needed": response_data.get(
                            "is_correction_needed", False
                        ),
                        "has_corrected_sentence": bool(
                            response_data.get("corrected_sentence")
                        ),
                    },
                    detailed=False,
                )
                return response_data
            else:
                raise json.JSONDecodeError(
                    "No JSON object found in response", response_content, 0
                )
        except json.JSONDecodeError:
            log_pipeline_step(
                "CORRECTION",
                f"Error: Could not extract valid JSON from response. RAW_RESPONSE: '{response_content}'",
                extra={"response": response_content},
                detailed=False,
            )
            return {
                "is_correction_needed": False,
                "corrected_sentence": text_to_correct,
                "reasoning": "JSON decode error.",
            }
        except Exception as e:
            log_pipeline_step(
                "CORRECTION",
                f"Error calling Ollama: {e}",
                detailed=False,
            )
            return {
                "is_correction_needed": False,
                "corrected_sentence": text_to_correct,
                "reasoning": "Ollama API error.",
            }

    async def _perform_correction(self, target_utterance: dict):
        """Performs correction logic on a specific target utterance."""
        await self.viewer_manager.broadcast(
            {
                "message_id": target_utterance["message_id"],
                "type": "status_update",
                "correction_status": "checking",
            }
        )

        log_utterance_step(
            "CORRECTION",
            target_utterance["message_id"],
            "Running contextual correction.",
            speaker=target_utterance["speaker"],
            extra={
                "target_transcription": target_utterance["transcription"],
                "history_size": len(self.utterance_history),
            },
        )

        history_as_list = list(self.utterance_history)
        context_list = []
        try:
            target_index = next(
                i
                for i, u in enumerate(history_as_list)
                if u["message_id"] == target_utterance["message_id"]
            )
            context_utterances = history_as_list[target_index + 1 : target_index + 3]
            context_list = [u["transcription"] for u in context_utterances]
        except StopIteration:
            log_utterance_step(
                "CORRECTION",
                target_utterance["message_id"],
                "Target utterance not found in history; sending without context.",
                speaker=target_utterance["speaker"],
            )

        response_data = await self.correct_with_context(
            text_to_correct=target_utterance["transcription"],
            context_history=context_list,
        )

        is_needed = response_data.get("is_correction_needed", False)
        reason = response_data.get("reasoning", "No reason provided.")
        corrected_transcription = response_data.get("corrected_sentence")
        confidence = response_data.get("confidence")
        try:
            correction_confidence = (
                float(confidence) if confidence is not None else None
            )
        except (ValueError, TypeError):
            correction_confidence = None
        if (
            is_needed
            and corrected_transcription
            and corrected_transcription.strip()
            != target_utterance["transcription"].strip()
        ):
            await self.viewer_manager.broadcast(
                {
                    "message_id": target_utterance["message_id"],
                    "type": "status_update",
                    "correction_status": "correcting",
                }
            )
            log_utterance_step(
                "CORRECTION",
                target_utterance["message_id"],
                "Correction is needed and will be applied.",
                speaker=target_utterance["speaker"],
                extra={"reason": reason, "corrected_text": corrected_transcription},
            )

            full_corrected_translation = ""
            async for chunk in self.retranslation_service.translate_stream(
                text_to_translate=corrected_transcription
            ):
                full_corrected_translation = chunk

            asyncio.create_task(
                self.rag_service.log_correction(
                    original_transcription=target_utterance["transcription"],
                    original_translation=target_utterance["translation"],
                    context_history=context_list,
                    corrected_transcription=corrected_transcription,
                    corrected_translation=full_corrected_translation,
                    correction_reason=reason,
                    correction_confidence=correction_confidence,
                    metadata={
                        "message_id": target_utterance["message_id"],
                        "speaker": target_utterance["speaker"],
                        "model": self.model,
                        "correction_applied": True,
                    },
                )
            )

            payload = {
                "message_id": target_utterance["message_id"],
                "transcription": corrected_transcription,
                "translation": full_corrected_translation,
                "speaker": target_utterance["speaker"],
                "type": "correction",
                "isfinalize": True,
            }
            await self.viewer_manager.broadcast(payload)
            log_utterance_step(
                "CORRECTION",
                target_utterance["message_id"],
                "Correction broadcast complete.",
                speaker=target_utterance["speaker"],
                detailed=False,
            )
        else:
            await self.viewer_manager.broadcast(
                {
                    "message_id": target_utterance["message_id"],
                    "type": "status_update",
                    "correction_status": "checked_ok",
                }
            )
            log_reason = (
                reason
                if not is_needed
                else "Model suggested a correction, but it was empty or identical to the original."
            )

            asyncio.create_task(
                self.rag_service.log_correction(
                    original_transcription=target_utterance["transcription"],
                    original_translation=target_utterance["translation"],
                    context_history=context_list,
                    corrected_transcription=target_utterance["transcription"],
                    corrected_translation=target_utterance["translation"],
                    correction_reason=log_reason,
                    correction_confidence=correction_confidence,
                    metadata={
                        "message_id": target_utterance["message_id"],
                        "speaker": target_utterance["speaker"],
                        "model": self.model,
                        "correction_applied": False,
                    },
                )
            )

            log_utterance_step(
                "CORRECTION",
                target_utterance["message_id"],
                "No correction applied.",
                speaker=target_utterance["speaker"],
                extra={"reason": log_reason},
                detailed=False,
            )

    async def _run_contextual_correction(self):
        """Internal helper to check and run correction on the target utterance."""
        if len(self.utterance_history) < self.CORRECTION_CONTEXT_THRESHOLD:
            return

        target_utterance = self.utterance_history[-self.CORRECTION_CONTEXT_THRESHOLD]
        await self._perform_correction(target_utterance)

    async def process_final_utterance(self, utterance: dict):
        """
        Public method to receive a new final utterance, store it,
        and trigger the correction pipeline.
        """
        log_utterance_step(
            "CORRECTION",
            utterance["message_id"],
            "Received final utterance for history.",
            speaker=utterance["speaker"],
            extra={"history_size": len(self.utterance_history)},
            detailed=True,
        )
        self.utterance_history.append(utterance)
        asyncio.create_task(self._run_contextual_correction())

    async def finalize_session(self):
        """
        Public method to be called on session end.
        Processes the last few utterances that didn't get a chance to be corrected.
        """
        num_final_to_check = self.CORRECTION_CONTEXT_THRESHOLD - 1

        if len(self.utterance_history) >= self.CORRECTION_CONTEXT_THRESHOLD:
            final_targets = list(self.utterance_history)[-num_final_to_check:]
            log_pipeline_step(
                "SESSION",
                f"Performing final correction check on last {len(final_targets)} utterance(s).",
                detailed=False,
            )
            for utterance in final_targets:
                await self._perform_correction(utterance)
        elif len(self.utterance_history) > 0:
            log_pipeline_step(
                "SESSION",
                f"Performing final correction check on all {len(self.utterance_history)} utterance(s).",
                detailed=False,
            )
            for utterance in list(self.utterance_history):
                await self._perform_correction(utterance)
        else:
            log_pipeline_step(
                "SESSION",
                f"Not enough history ({len(self.utterance_history)}) for final corrections check.",
                detailed=False,
            )
