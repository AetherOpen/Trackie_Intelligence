# src/trackie/services/vision/face_recognizer.py

import os
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd

from ....utils.logger import get_logger

logger = get_logger(__name__)

# Importa DeepFace dinamicamente para lidar com erros de instalação
try:
    from deepface import DeepFace
except ImportError:
    logger.error("A biblioteca 'deepface' não está instalada. O reconhecimento facial está desabilitado.")
    DeepFace = None

class FaceRecognizer:
    """
    Encapsula as funcionalidades de reconhecimento e gerenciamento de rostos
    usando a biblioteca DeepFace.
    """
    def __init__(self, settings: Dict[str, Any]):
        if not DeepFace:
            raise ImportError("DeepFace não pôde ser importado. O serviço não pode ser inicializado.")

        vision_settings = settings.get("vision", {})
        self.db_path = vision_settings.get("deepface_db_path")
        self.model_name = vision_settings.get("deepface_model_name", "VGG-Face")
        self.detector_backend = vision_settings.get("deepface_detector_backend", "opencv")
        self.distance_metric = vision_settings.get("deepface_distance_metric", "cosine")

        self._prepare_database()

    def _prepare_database(self):
        """Garante que o diretório do banco de dados exista e pré-carrega os modelos."""
        logger.info("Preparando o serviço de reconhecimento facial...")
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            logger.info(f"Diretório de banco de dados de rostos criado em: {self.db_path}")
        
        try:
            logger.info("Pré-carregando modelos DeepFace...")
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(img_path=dummy_frame, actions=['emotion'], enforce_detection=False)
            logger.info("Modelos DeepFace pré-carregados.")
        except Exception as e:
            logger.warning(f"AVISO: Erro ao pré-carregar modelos DeepFace: {e}")

    def identify_faces(self, frame: np.ndarray) -> List[pd.DataFrame]:
        """
        Busca por rostos conhecidos em um frame.

        Args:
            frame: O frame da imagem em formato NumPy array (BGR).

        Returns:
            Uma lista de DataFrames do Pandas, onde cada DataFrame contém as
            correspondências para um rosto detectado no frame. Retorna lista vazia se nada for encontrado.
        """
        try:
            dfs = DeepFace.find(
                img_path=frame,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=True,
                silent=True
            )
            return dfs
        except ValueError: # Erro comum quando nenhum rosto é detectado no frame de entrada
            logger.debug("Nenhum rosto detectado no frame para identificação.")
            return []
        except Exception as e:
            logger.error(f"Erro durante a identificação de rostos com DeepFace: {e}")
            return []

    def save_face(self, frame: np.ndarray, person_name: str) -> bool:
        """
        Extrai o rosto mais proeminente de um frame e o salva no banco de dados.

        Args:
            frame: O frame da imagem em formato NumPy array (BGR).
            person_name: O nome a ser associado ao rosto.

        Returns:
            True se o rosto foi salvo com sucesso, False caso contrário.
        """
        try:
            # Extrai o rosto (DeepFace lança ValueError se não encontrar)
            extracted_faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            # Pega o primeiro rosto (geralmente o maior)
            face_data = extracted_faces[0]['facial_area']
            x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
            face_image_cropped = frame[y:y+h, x:x+w]

            # Cria o diretório para a pessoa se não existir
            person_dir = os.path.join(self.db_path, person_name)
            os.makedirs(person_dir, exist_ok=True)

            # Salva a imagem do rosto
            timestamp = int(datetime.now().timestamp())
            image_path = os.path.join(person_dir, f"{person_name}_{timestamp}.jpg")
            
            success = cv2.imwrite(image_path, face_image_cropped)
            if success:
                logger.info(f"Rosto de '{person_name}' salvo em: {image_path}")
                # Força a reconstrução do cache de representações
                self._clear_representations_cache()
                return True
            else:
                logger.error(f"Falha ao salvar a imagem do rosto em {image_path}")
                return False

        except ValueError:
            logger.warning(f"Nenhum rosto detectado no frame para salvar para '{person_name}'.")
            return False
        except Exception as e:
            logger.error(f"Erro ao salvar rosto para '{person_name}': {e}")
            return False

    def _clear_representations_cache(self):
        """Remove o arquivo .pkl de representações para forçar a recriação."""
        pkl_path = os.path.join(self.db_path, f"representations_{self.model_name.lower()}.pkl")
        if os.path.exists(pkl_path):
            try:
                os.remove(pkl_path)
                logger.info("Cache de representações do DeepFace limpo para atualização.")
            except OSError as e:
                logger.warning(f"Não foi possível remover o cache de representações: {e}")