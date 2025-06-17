# src/trackie/config/settings.py

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from pydantic import BaseModel, Field, ValidationError

from ..utils.logger import get_logger

logger = get_logger(__name__)

# --- Modelos de Dados Pydantic para Validação ---
# Cada classe aqui corresponde a uma seção do arquivo YAML.

class UserSettings(BaseModel):
    name: str = "Usuário"

class GeminiSettings(BaseModel):
    api_key: str
    model: str = "models/gemini-1.5-flash-latest"
    temperature: float = 0.2

class OpenAISettings(BaseModel):
    api_key: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.7

class LLMSettings(BaseModel):
    provider: str
    system_prompt_path: Path
    gemini: GeminiSettings
    openai: OpenAISettings

class VideoSettings(BaseModel):
    provider: str = "opencv"
    device_index: int = 0
    fps: float = 1.0
    jpeg_quality: int = Field(50, ge=10, le=100) # Validação: entre 10 e 100

class AudioSettings(BaseModel):
    chunk_size: int = 1024
    send_sample_rate: int = 16000
    receive_sample_rate: int = 24000
    channels: int = 1

class VisionSettings(BaseModel):
    yolo_model_path: Path
    confidence_threshold: float = Field(0.45, ge=0.1, le=1.0)
    midas_model_type: str
    deepface_db_path: Path
    deepface_model_name: str
    deepface_detector_backend: str
    deepface_distance_metric: str

class PathSettings(BaseModel):
    data: Path
    tool_definitions: Path
    danger_sound: Path

class AppSettings(BaseModel):
    """O modelo Pydantic principal que engloba todas as configurações."""
    user: UserSettings
    llm: LLMSettings
    video: VideoSettings
    audio: AudioSettings
    vision: VisionSettings
    paths: PathSettings

def _resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Percorre o dicionário de configuração e substitui placeholders de variáveis de ambiente."""
    for key, value in config.items():
        if isinstance(value, dict):
            _resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var_name = value[2:-1]
            env_var_value = os.getenv(env_var_name)
            if not env_var_value:
                raise ValueError(f"Variável de ambiente '{env_var_name}' não definida, mas é necessária para a configuração.")
            config[key] = env_var_value
    return config

def load_settings(config_path: Path = Path("config/settings.yml")) -> AppSettings:
    """
    Carrega, valida e retorna as configurações da aplicação.

    Args:
        config_path: O caminho para o arquivo de configuração YAML.

    Returns:
        Uma instância de AppSettings com todas as configurações validadas.
    """
    if not config_path.exists():
        logger.error(f"Arquivo de configuração não encontrado em: {config_path}")
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    logger.info(f"Carregando configurações de: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Resolve as variáveis de ambiente antes da validação
        resolved_config = _resolve_env_vars(config_data)
        
        # Valida os dados carregados com o modelo Pydantic
        settings = AppSettings(**resolved_config)
        
        logger.info("Configurações carregadas e validadas com sucesso.")
        return settings

    except yaml.YAMLError as e:
        logger.critical(f"Erro de sintaxe no arquivo de configuração YAML: {e}")
        raise
    except ValidationError as e:
        logger.critical(f"Erro de validação nas configurações: {e}")
        raise
    except Exception as e:
        logger.critical(f"Erro inesperado ao carregar as configurações: {e}")
        raise

# Instância global das configurações para ser importada por outros módulos
# O carregamento acontece uma única vez quando este módulo é importado.
try:
    # Define o diretório base do projeto (assumindo que src está na raiz)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    settings = load_settings(BASE_DIR / "config/settings.yml")
except (FileNotFoundError, ValueError) as e:
    logger.critical(f"Não foi possível carregar as configurações. Encerrando. Erro: {e}")
    # Em uma aplicação real, você poderia ter um fallback ou encerrar o programa.
    settings = None 