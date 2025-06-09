# trackie_app/logger_config.py
import logging
import sys # Adicionado para o print de sys.path

# --- Configuração do sys.path ---
# Esta parte é específica para o ambiente de desenvolvimento e pode precisar de ajustes.
# Idealmente, a estrutura do projeto e o PYTHONPATH lidariam com isso.
dependency_path = r"C:\TrackieIntelligence\WorkTools\WorkingTools" # Mantido como no original

if dependency_path not in sys.path:
    sys.path.insert(0, dependency_path)
    print(f"'{dependency_path}' adicionado ao sys.path para importação de módulos.") # Ajustado para mostrar o caminho
else:
    print(f"'{dependency_path}' já está no sys.path.") # Ajustado para mostrar o caminho
# --- Fim da configuração do sys.path ---


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

def get_logger(name):
    """Retorna uma instância de logger com o nome especificado."""
    return logging.getLogger(name)