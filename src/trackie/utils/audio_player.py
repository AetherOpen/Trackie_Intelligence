# src/trackie/utils/audio_player.py

from pathlib import Path
from playsound import playsound

from .logger import get_logger

logger = get_logger(__name__)

def play_wav_file(filepath: Path):
    """
    Reproduz um arquivo de áudio de forma síncrona (bloqueante).

    Args:
        filepath: O caminho para o arquivo de áudio.
    """
    if not filepath.exists():
        logger.error(f"Arquivo de áudio não encontrado para reprodução: {filepath}")
        return

    try:
        logger.info(f"Reproduzindo arquivo de áudio: {filepath}")
        playsound(str(filepath)) # playsound espera uma string
        logger.info("Reprodução de áudio concluída.")
    except Exception as e:
        # A biblioteca playsound pode lançar exceções variadas dependendo do sistema operacional
        logger.error(f"Erro ao reproduzir o arquivo de áudio {filepath}: {e}")
