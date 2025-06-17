# src/trackie/services/llm/base.py

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, NamedTuple, Optional, List
from dataclasses import dataclass

# Estruturas de dados para padronizar a comunicação com a aplicação
@dataclass
class FunctionCall:
    name: str
    args: Dict[str, Any]

@dataclass
class LLMResponse:
    type: str  # "audio", "text", ou "function_call"
    data: Any

class LLMService(ABC):
    """
    Interface abstrata para um serviço de Modelo de Linguagem Grande (LLM).
    Define o contrato que todos os provedores de LLM (Gemini, OpenAI, etc.) devem seguir.
    """

    @abstractmethod
    async def connect(self, system_prompt: str, tools: List[Dict]) -> None:
        """Estabelece a conexão de streaming com o serviço de LLM."""
        pass

    @abstractmethod
    async def send_media_stream(self, media_queue: asyncio.Queue) -> None:
        """Inicia um loop para enviar dados de mídia (áudio/vídeo) de uma fila para o LLM."""
        pass

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Envia uma única mensagem de texto para o LLM."""
        pass

    @abstractmethod
    async def send_tool_response(self, tool_name: str, response_data: Dict) -> None:
        """Envia o resultado da execução de uma ferramenta de volta para o LLM."""
        pass

    @abstractmethod
    async def receive(self) -> AsyncGenerator[LLMResponse, None]:
        """Um gerador assíncrono que produz respostas do LLM."""
        # O 'yield' vazio é necessário para a sintaxe do gerador assíncrono abstrato
        if False:
            yield

    @abstractmethod
    async def close(self) -> None:
        """Fecha a conexão com o serviço de LLM."""
        pass