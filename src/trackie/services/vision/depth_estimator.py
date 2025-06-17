# src/trackie/services/vision/depth_estimator.py

import os
from typing import Any, Dict, Optional
import torch
import cv2
import numpy as np

from ....utils.logger import get_logger

logger = get_logger(__name__)

class DepthEstimator:
    """
    Encapsula o modelo de estimativa de profundidade MiDaS.
    """
    def __init__(self, settings: Dict[str, Any]):
        """
        Inicializa o estimador de profundidade carregando o modelo MiDaS.

        Args:
            settings: Dicionário de configuração contendo os parâmetros do MiDaS.
        """
        vision_settings = settings.get("vision", {})
        self.model_type = vision_settings.get("midas_model_type", "DPT_SwinV2_L_384")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.transform = None

        try:
            logger.info(f"Carregando modelo MiDaS ({self.model_type}) para o dispositivo: {self.device}")
            
            # Define um diretório de cache local para os modelos
            cache_dir = os.path.join(settings['paths']['data'], "models", "midas_cache")
            os.makedirs(cache_dir, exist_ok=True)
            torch.hub.set_dir(cache_dir)

            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            
            transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = transforms_hub.dpt_transform if "dpt" in self.model_type.lower() else transforms_hub.small_transform
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Modelo MiDaS carregado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo MiDaS: {e}. A estimativa de profundidade será desabilitada.")
            self.model = None # Garante que o serviço falhe graciosamente

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estima o mapa de profundidade de um único frame.

        Args:
            frame: O frame da imagem em formato NumPy array (BGR).

        Returns:
            Um mapa de profundidade como um NumPy array, ou None em caso de falha.
        """
        if not self.model or not self.transform:
            return None

        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            return prediction.cpu().numpy()
        except Exception as e:
            logger.error(f"Erro durante a inferência do MiDaS: {e}")
            return None