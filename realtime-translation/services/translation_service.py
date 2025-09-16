import os
from typing import AsyncGenerator, List

from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI


class QwenTranslationService:
    """
    Handles translation by making REST API calls to Alibaba's Qwen model
    via the DashScope service in an OpenAI-compatible mode.
    """

    def __init__(self):
        load_dotenv()
        try:
            self.client = AsyncOpenAI(
                api_key=os.environ["DASHSCOPE_API_KEY"],
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
        except KeyError:
            raise ValueError("The 'DASHSCOPE_API_KEY' environment variable is not set.")

    async def translate_stream(
        self, text_to_translate: str, context: List[str]
    ) -> AsyncGenerator[str, None]:
        """
        Translates a block of text using the Qwen model and streams the results,
        leveraging previous sentences for context.
        """
        combined_prompt = (
            "Preserve the core meaning and nuance of the original text without adding new information. "
            "Use the provided context to resolve ambiguities."
        )

        if context:
            context_str = " ".join(context)
            combined_prompt += (
                f'\n\nPrevious sentences (for context): "{context_str}"\n\n'
                f'Translate ONLY the following new Chinese sentence: "{text_to_translate}"'
            )
        else:
            combined_prompt += (
                f'\n\nTranslate the following Chinese sentence: "{text_to_translate}"'
            )

        try:
            stream = await self.client.chat.completions.create(
                model="qwen-mt-turbo",
                messages=[
                    {"role": "user", "content": combined_prompt},
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
                    yield content
        except APIError as e:
            error_message = f"[Translation Error: {e.message}]"
            print(f"Alibaba Qwen API error: {e}")
            yield error_message
        except Exception as e:
            print(f"An unexpected error occurred during translation: {e}")
            yield "[Translation Error]"
