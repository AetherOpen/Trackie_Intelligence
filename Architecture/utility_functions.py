# trackie_app/utility_functions.py
from playsound import playsound
from .logger_config import get_logger # Import relativo

logger = get_logger(__name__)

def play_wav_file_sync(filepath):
    """Reproduz um arquivo WAV de forma síncrona."""
    try:
        logger.info(f"Reproduzindo arquivo de áudio: {filepath}") # Adicionado filepath ao log
        playsound(filepath)
        logger.info(f"Reprodução de {filepath} concluída.") # Adicionado filepath ao log
    except Exception as e:
        logger.error(f"Erro ao reproduzir o arquivo WAV {filepath}: {e}") # Adicionado filepath ao log

# Outras funções utilitárias puras podem ser adicionadas aqui no futuro.