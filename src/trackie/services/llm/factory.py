# src/trackie/services/llm/factory.py

from typing import Any, Dict

from .base import LLMService
from .gemini_service import GeminiService
from .openai_service import OpenAIService
from ...utils.logger import get_logger

logger = get_logger(__name__)

def get_llm_service(settings: Dict[str, Any]) -> LLMService:
    """
    Factory para criar uma instância do serviço de LLM com base nas configurações.

    Args:
        settings: O dicionário de configurações da aplicação.

    Returns:
        Uma instância de uma classe que implementa a interface LLMService.
    """
    provider = settings.get("llm", {}).get("provider", "").lower()
    logger.info(f"Criando serviço de LLM para o provedor: '{provider}'")

    if provider == "gemini":
        return GeminiService(settings)
    elif provider == "openai":
        return OpenAIService(settings)
    else:
        raise ValueError(f"Provedor de LLM não suportado ou não especificado: '{provider}'")