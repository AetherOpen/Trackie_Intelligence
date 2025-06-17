# src/trackie/services/vision/object_detector.py

from typing import Any, Dict, List
import numpy as np
from ultralytics import YOLO

from ....utils.logger import get_logger

logger = get_logger(__name__)

class ObjectDetector:
    """
    Encapsula o modelo de detecção de objetos YOLO.
    """
    def __init__(self, settings: Dict[str, Any]):
        """
        Inicializa o detector de objetos carregando o modelo YOLO.

        Args:
            settings: Dicionário de configuração contendo os caminhos e parâmetros do YOLO.
        """
        vision_settings = settings.get("vision", {})
        model_path = vision_settings.get("yolo_model_path")
        self.confidence_threshold = vision_settings.get("confidence_threshold", 0.40)
        
        if not model_path:
            raise ValueError("Caminho do modelo YOLO não especificado nas configurações.")

        try:
            logger.info(f"Carregando modelo YOLO de: {model_path}")
            self.model = YOLO(model_path)
            # Realiza uma predição "a frio" para aquecer o modelo
            self.model.predict(np.zeros((640, 480, 3), dtype=np.uint8), verbose=False)
            logger.info("Modelo YOLO carregado e aquecido com sucesso.")
        except Exception as e:
            logger.critical(f"Falha crítica ao carregar o modelo YOLO: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Any]:
        """
        Realiza a detecção de objetos em um único frame.

        Args:
            frame: O frame da imagem em formato NumPy array (BGR).

        Returns:
            Uma lista de resultados de detecção brutos da biblioteca YOLO.
        """
        try:
            # YOLO espera imagens em RGB, mas a biblioteca ultralytics lida com a conversão.
            results = self.model.predict(
                frame, 
                verbose=False, 
                conf=self.confidence_threshold
            )
            return results
        except Exception as e:
            logger.error(f"Erro durante a inferência do YOLO: {e}")
            return []