# src/trackie/utils/logger.py

import logging
import sys

# Nível de log padrão. Pode ser sobrescrito por uma configuração no futuro.
LOG_LEVEL = logging.INFO

def setup_logging():
    """
    Configura o formato e o nível do logger raiz.
    Esta função deve ser chamada uma única vez, no início da aplicação.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout, # Garante que os logs vão para a saída padrão
    )
    
    # Reduz o "ruído" de bibliotecas muito verbosas
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google_auth_httplib2").setLevel(logging.WARNING)
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Sistema de logging configurado.")

def get_logger(name: str) -> logging.Logger:
    """
    Retorna uma instância de logger com o nome especificado.

    Args:
        name: O nome do logger, geralmente __name__ do módulo que o chama.

    Returns:
        Uma instância de logging.Logger.
    """
    return logging.getLogger(name)