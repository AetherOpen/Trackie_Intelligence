# trackie_app/external_apis.py
import os
import pyaudio
from google import genai
from google.genai import types as genai_types # Renomeado para evitar conflito
from dotenv import load_dotenv

from .logger_config import get_logger
from .app_config import BASE_DIR # Importa BASE_DIR para o caminho do .env

logger = get_logger(__name__)

# --- Inicialização do PyAudio ---
PYAUDIO_INSTANCE = None
PYAUDIO_FORMAT = None
try:
    PYAUDIO_INSTANCE = pyaudio.PyAudio()
    PYAUDIO_FORMAT = pyaudio.paInt16 # Definindo o formato aqui
    logger.info("PyAudio inicializado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao inicializar PyAudio: {e}. O áudio não funcionará.")
    PYAUDIO_INSTANCE = None
    PYAUDIO_FORMAT = None # Garante que é None se falhar

# --- Configuração do Cliente Gemini ---
GEMINI_API_KEY = "AIzaSyC98B7lUqRWEGW4cmMHBz_dgDCwujslt10"
if not GEMINI_API_KEY:
    try:
        dotenv_path = os.path.join(BASE_DIR, '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
            if GEMINI_API_KEY:
                 logger.info(f"Chave API carregada de: {dotenv_path}")
            else:
                logger.warning(f"Chave GEMINI_API_KEY não encontrada em {dotenv_path} após carregar.")
        else:
            logger.info(f"Arquivo .env não encontrado em: {dotenv_path}")
    except ImportError:
        logger.info("Biblioteca python-dotenv não instalada. Não é possível carregar .env.")
    except Exception as e_env:
        logger.error(f"Erro ao carregar .env: {e_env}")

if not GEMINI_API_KEY:
    logger.warning("AVISO: Chave da API Gemini não encontrada nas variáveis de ambiente ou .env.")
    # O código original tinha um fallback para uma chave hardcoded, que foi removido por segurança.
    # Se a chave não for encontrada, o cliente Gemini falhará ao inicializar.

GEMINI_CLIENT = None
if GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=genai_types.HttpOptions(api_version='v1alpha')
        )
        logger.info("Cliente Gemini inicializado.")
    except Exception as e_client:
        logger.error(f"ERRO CRÍTICO ao inicializar cliente Gemini: {e_client}")
        logger.info("Verifique a API Key e a conexão.")
        GEMINI_CLIENT = None
else:
    logger.error("ERRO CRÍTICO: Chave da API Gemini não fornecida. Cliente Gemini não pode ser inicializado.")
    GEMINI_CLIENT = None