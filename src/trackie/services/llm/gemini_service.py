# src/trackie/services/llm/gemini_service.py

import asyncio
from typing import Any, AsyncGenerator, Dict, List

from google import genai
from google.genai import types as genai_types
from google.genai.types import Content, Part
from google.protobuf.struct_pb2 import Value

from .base import LLMService, LLMResponse, FunctionCall
from ...utils.logger import get_logger

logger = get_logger(__name__)

class GeminiService(LLMService):
    """Implementação do LLMService para a API Google Gemini LiveConnect."""

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings["llm"]["gemini"]
        self.client = None
        self.session = None
        self.live_connect_config = None

        if not self.settings.get("api_key"):
            raise ValueError("A chave da API do Gemini não foi encontrada nas configurações.")
        
        try:
            self.client = genai.Client(
                api_key=self.settings["api_key"],
                http_options=genai_types.HttpOptions(api_version='v1alpha')
            )
            logger.info("Cliente Gemini inicializado com sucesso.")
        except Exception as e:
            logger.critical(f"ERRO CRÍTICO ao inicializar cliente Gemini: {e}")
            raise

    def _build_config(self, system_prompt: str, tools: List[Dict]) -> None:
        """Constrói o objeto de configuração LiveConnect."""
        # TODO: Carregar as definições de ferramentas do `tool_definitions.json`
        # e convertê-las para o formato `genai_types.Tool`.
        # Por enquanto, usamos a estrutura original.
        gemini_tools = [genai_types.Tool.from_dict(t) for t in tools]

        self.live_connect_config = genai_types.LiveConnectConfig(
            temperature=self.settings.get("temperature", 0.2),
            response_modalities=["audio"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=genai_types.SpeechConfig(
                language_code="pt-BR",
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name="Zephyr")
                )
            ),
            tools=gemini_tools,
            system_instruction=genai_types.Content(
                parts=[Part.from_text(text=system_prompt)],
                role="system"
            )
        )
        logger.info("Configuração LiveConnect do Gemini construída.")

    async def connect(self, system_prompt: str, tools: List[Dict]) -> None:
        self._build_config(system_prompt, tools)
        logger.info(f"Conectando ao modelo Gemini: {self.settings['model']}")
        self.session = await self.client.aio.live.connect(
            model=self.settings['model'],
            config=self.live_connect_config
        )
        logger.info("Sessão Gemini LiveConnect estabelecida.")

    async def send_media_stream(self, media_queue: asyncio.Queue) -> None:
        while True:
            try:
                media_data = await media_queue.get()
                if media_data is None: # Sinal de parada
                    break
                if self.session:
                    await self.session.send(input=media_data)
                media_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro ao enviar mídia para o Gemini: {e}")
                # Adicionar lógica de reconexão se necessário

    async def send_text(self, text: str) -> None:
        if self.session:
            await self.session.send(input=text, end_of_turn=True)

    async def send_tool_response(self, tool_name: str, response_data: Dict) -> None:
        if self.session:
            function_response_content = Content(
                role="tool",
                parts=[Part.from_function_response(
                    name=tool_name,
                    response={"result": Value(string_value=str(response_data.get("result", "")))}
                )]
            )
            await self.session.send(input=function_response_content)

    async def receive(self) -> AsyncGenerator[LLMResponse, None]:
        if not self.session:
            return

        async for response_part in self.session.receive():
            if response_part.data:
                yield LLMResponse(type="audio", data=response_part.data)
            
            if response_part.text:
                yield LLMResponse(type="text", data=response_part.text)

            if getattr(response_part, "function_call", None):
                fc = response_part.function_call
                args = {key: val for key, val in fc.args.items()}
                yield LLMResponse(type="function_call", data=FunctionCall(name=fc.name, args=args))

    async def close(self) -> None:
        if self.session:
            await self.session.close()
            logger.info("Sessão Gemini fechada.")