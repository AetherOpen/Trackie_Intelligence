# src/trackie/services/llm/openai_service.py

import asyncio
from typing import Any, AsyncGenerator, Dict, List

from .base import LLMService, LLMResponse

class OpenAIService(LLMService):
    """
    Implementação do LLMService para a API OpenAI (GPT-4o).
    (Ainda não implementado)
    """

    def __init__(self, settings: Dict[str, Any]):
        raise NotImplementedError("O serviço OpenAI ainda não foi implementado.")

    async def connect(self, system_prompt: str, tools: List[Dict]) -> None:
        raise NotImplementedError

    async def send_media_stream(self, media_queue: asyncio.Queue) -> None:
        raise NotImplementedError

    async def send_text(self, text: str) -> None:
        raise NotImplementedError

    async def send_tool_response(self, tool_name: str, response_data: Dict) -> None:
        raise NotImplementedError

    async def receive(self) -> AsyncGenerator[LLMResponse, None]:
        if False:
            yield
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError