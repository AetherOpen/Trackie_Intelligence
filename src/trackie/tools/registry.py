# src/trackie/tools/registry.py

import asyncio
import json
from typing import Any, Callable, Coroutine, Dict, List

from .handlers import ToolHandlers
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ToolRegistry:
    """
    Gerencia o registro, as definições e a execução das ferramentas de Function Calling.
    """
    def __init__(
        self,
        handlers: ToolHandlers,
        llm_service: Any, # Para enviar a resposta de volta
        thinking_event: asyncio.Event,
        tool_definitions_path: str
    ):
        self.handlers = handlers
        self.llm_service = llm_service
        self.thinking_event = thinking_event
        self.tool_definitions = self._load_definitions(tool_definitions_path)
        
        # Mapeia o nome da ferramenta para o método handler correspondente
        self.registry: Dict[str, Callable[..., Coroutine]] = {
            "save_known_face": self.handlers.handle_save_known_face,
            "identify_person_in_front": self.handlers.handle_identify_person_in_front,
            "locate_object_and_estimate_distance": self.handlers.handle_locate_object,
            # Adicione outras ferramentas aqui
        }
        logger.info(f"Registro de ferramentas inicializado com: {list(self.registry.keys())}")

    def _load_definitions(self, path: str) -> List[Dict[str, Any]]:
        """Carrega as definições das ferramentas de um arquivo JSON."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                definitions = json.load(f)
                # TODO: Adicionar validação com Pydantic para garantir que o formato está correto.
                logger.info(f"Definições de ferramentas carregadas de: {path}")
                return definitions
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Falha ao carregar definições de ferramentas de {path}: {e}")
            return []

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Retorna a lista de definições de ferramentas para o LLM."""
        # Adiciona as ferramentas padrão que o LLM pode usar
        standard_tools = [
            {"code_execution": {}},
            {"google_search": {}}
        ]
        return standard_tools + self.tool_definitions

    async def execute(self, tool_name: str, tool_args: Dict[str, Any]):
        """
        Executa a ferramenta solicitada e envia o resultado de volta para o LLM.
        """
        handler_coro = self.registry.get(tool_name)
        
        if not handler_coro:
            error_msg = f"Erro: Ferramenta '{tool_name}' não encontrada no registro."
            logger.error(error_msg)
            result = {"result": error_msg}
        else:
            try:
                # Executa o handler da ferramenta com os argumentos fornecidos
                result = await handler_coro(**tool_args)
            except Exception:
                error_msg = f"Ocorreu um erro interno ao executar a ferramenta '{tool_name}'."
                logger.exception(error_msg)
                result = {"result": error_msg}
        
        # Envia o resultado de volta para o LLM
        await self.llm_service.send_tool_response(tool_name, result)
        
        # Libera o evento de "pensando"
        self.thinking_event.clear()
        logger.info(f"Execução da ferramenta '{tool_name}' concluída e resposta enviada.")