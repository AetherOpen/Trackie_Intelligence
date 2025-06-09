# trackie_app/audio_loop_refactored.py
import os
import sys
import asyncio
import base64
import io
import json
import traceback
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union


# Bibliotecas de Terceiros
import cv2
from PIL import Image
import mss
import pandas as pd # Importação mantida, com checagens onde DeepFace.find é usado
from google import genai
from google.genai import types as genai_types
from google.genai.types import Content, Part, GenerateContentConfig, LiveConnectConfig, Modality # Explicitamente importado
from google.genai import errors as genai_errors
from google.protobuf.struct_pb2 import Value
import numpy as np
import torch

# Imports de módulos locais
from .logger_config import get_logger # Supondo que este módulo exista e funcione
from .app_config import (
    DEFAULT_MODE, TRCKUSER, DANGER_SOUND_PATH, YOLO_CONFIDENCE_THRESHOLD,
    YOLO_CLASS_MAP, DANGER_CLASSES, DB_PATH, DEEPFACE_DETECTOR_BACKEND,
    DEEPFACE_DISTANCE_METRIC, DEEPFACE_MODEL_NAME, METERS_PER_STEP,
    AUDIO_CHANNELS, AUDIO_SEND_SAMPLE_RATE, AUDIO_CHUNK_SIZE, CONFIG_PATH,
    GEMINI_MODEL_NAME, AUDIO_RECEIVE_SAMPLE_RATE
)
from .external_apis import PYAUDIO_INSTANCE, PYAUDIO_FORMAT, GEMINI_CLIENT # Supondo que este módulo exista e funcione
from .gemini_settings import GEMINI_LIVE_CONNECT_CONFIG, GEMINI_TOOLS # Supondo que este módulo exista e funcione
from .utility_functions import play_wav_file_sync # Supondo que este módulo exista e funcione
from .models import ( # Supondo que este módulo exista e funcione
    load_yolo_model, preload_deepface_models, load_midas_model, ensure_deepface_db_path
)

# Importar DeepFace dinamicamente ou condicionalmente se for um problema
try:
    from deepface import DeepFace # Movido para importação direta
except ImportError:
    DeepFace = None # Permite que o programa funcione parcialmente se DeepFace não estiver instalado
    get_logger(__name__).error("DeepFace não pôde ser importado. Funcionalidades de reconhecimento facial estarão desabilitadas.")


logger = get_logger(__name__)
from .audio_loop import AudioLoop
print("audioloop importada para function calling")


class Function_Calling:


    # --- Funções de "Tool" (Executadas localmente a pedido do Gemini) ---

    def _handle_save_known_face(self, person_name: str) -> str:
        """
        Salva o rosto da pessoa atualmente visível na câmera com o nome fornecido.
        Esta função é BLOQUEANTE e deve ser chamada com `asyncio.to_thread`.

        Args:
            person_name (str): O nome da pessoa para associar ao rosto salvo.

        Returns:
            str: Uma mensagem indicando o sucesso ou falha da operação.
        """
        if not DeepFace:
            logger.error("[DeepFace Tool] DeepFace não está disponível. Não é possível salvar rosto.")
            return f"Desculpe, {self.trckuser}, a funcionalidade de reconhecimento facial não está disponível no momento."

        logger.info(f"[DeepFace Tool] Executando _handle_save_known_face para '{person_name}'.")
        start_time = time.time()
        
        frame_to_process: Optional[np.ndarray] = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy() # Trabalha com uma cópia

        if frame_to_process is None:
            logger.warning("[DeepFace Tool] Nenhum frame de câmera disponível para salvar rosto.")
            return f"{self.trckuser}, não consigo ver nada no momento para salvar o rosto de {person_name}."

        # Sanitiza o nome da pessoa para criar um nome de diretório/arquivo seguro
        safe_person_name_dir = "".join(c if c.isalnum() or c in [' '] else '_' for c in person_name).strip().replace(" ", "_")
        if not safe_person_name_dir: safe_person_name_dir = "desconhecido_face" # Evita nome vazio
        
        person_image_dir = os.path.join(DB_PATH, safe_person_name_dir)

        try:
            if not os.path.exists(person_image_dir):
                os.makedirs(person_image_dir)
                logger.info(f"[DeepFace Tool] Diretório criado para '{person_name}': {person_image_dir}")

            # Extrai rostos do frame. DeepFace.extract_faces é bloqueante.
            # `enforce_detection=True` (padrão) já lança ValueError se nenhum rosto for detectado.
            # No entanto, o retorno pode ser uma lista vazia se `enforce_detection=False`.
            # Vamos manter `enforce_detection=True` (ou omitir, pois é padrão) e tratar ValueError.
            detected_faces = DeepFace.extract_faces(
                img_path=frame_to_process,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=True, 
                align=True
            )

            if not detected_faces or not isinstance(detected_faces, list) or 'facial_area' not in detected_faces[0]:
                # Este bloco pode ser redundante se enforce_detection=True sempre lança erro, mas é uma segurança.
                logger.warning(f"[DeepFace Tool] Nenhum rosto detectado ou formato de resultado inesperado para '{person_name}'.")
                return f"{self.trckuser}, não consegui detectar um rosto claro para {person_name}."

            # Pega a primeira face detectada (a maior, geralmente)
            face_data = detected_faces[0]['facial_area'] 
            x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
            
            # Adiciona uma pequena margem e garante que as coordenadas estão dentro dos limites do frame
            margin = 10 
            y1 = max(0, y - margin)
            y2 = min(frame_to_process.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame_to_process.shape[1], x + w + margin)
            
            face_image_cropped = frame_to_process[y1:y2, x1:x2]

            if face_image_cropped.size == 0:
                logger.warning(f"[DeepFace Tool] Erro ao recortar rosto para '{person_name}' (imagem resultante vazia).")
                return f"{self.trckuser}, houve um erro ao processar o rosto de {person_name}."

            timestamp = int(time.time())
            safe_file_name_base = "".join(c if c.isalnum() else '_' for c in person_name).strip()
            if not safe_file_name_base: safe_file_name_base = "rosto"
            # Nome de arquivo único para evitar sobrescrever
            image_file_name = f"{safe_file_name_base.lower()}_{timestamp}.jpg" 
            full_image_path = os.path.join(person_image_dir, image_file_name)

            save_success = cv2.imwrite(full_image_path, face_image_cropped)
            if not save_success:
                logger.error(f"[DeepFace Tool] Falha ao salvar imagem do rosto em '{full_image_path}'.")
                return f"{self.trckuser}, ocorreu um erro ao salvar a imagem do rosto de {person_name}."

            # Remove o arquivo de representações em cache para forçar a recriação pelo DeepFace.find
            # O nome do arquivo .pkl depende do modelo, ex: "representations_vgg_face.pkl"
            # É mais seguro deixar o DeepFace gerenciar seu cache, mas se a remoção for necessária:
            model_name_safe = DEEPFACE_MODEL_NAME.replace('-', '_').lower() # Ex: 'vgg_face'
            representations_pkl_path = os.path.join(DB_PATH, f"representations_{model_name_safe}.pkl")
            if os.path.exists(representations_pkl_path):
                try:
                    os.remove(representations_pkl_path)
                    logger.info(f"[DeepFace Tool] Cache de representações '{representations_pkl_path}' removido para atualização.")
                except OSError:
                    logger.warning(f"[DeepFace Tool] Falha ao remover cache de representações '{representations_pkl_path}'. Pode ser recriado automaticamente.")
            
            duration = time.time() - start_time
            logger.info(f"[DeepFace Tool] Rosto de '{person_name}' salvo em '{full_image_path}'. Duração: {duration:.2f}s.")
            return f"{self.trckuser}, o rosto de {person_name} foi salvo com sucesso."

        except ValueError as ve: # Comumente lançado por DeepFace se nenhum rosto é detectado com enforce_detection=True
            logger.warning(f"[DeepFace Tool] Nenhum rosto detectado (ValueError) para '{person_name}': {ve}")
            return f"{self.trckuser}, não consegui detectar um rosto claro para salvar para {person_name}."
        except Exception:
            logger.exception(f"[DeepFace Tool] Erro inesperado ao salvar rosto para '{person_name}'.")
            return f"{self.trckuser}, ocorreu um erro inesperado ao tentar salvar o rosto de {person_name}."

    def _handle_identify_person_in_front(self) -> str:
        """
        Tenta identificar a pessoa atualmente visível na câmera usando DeepFace.
        Esta função é BLOQUEANTE e deve ser chamada com `asyncio.to_thread`.

        Returns:
            str: Uma mensagem descrevendo a pessoa identificada ou indicando falha.
        """
        if not DeepFace:
            logger.error("[DeepFace Tool] DeepFace não está disponível. Não é possível identificar pessoa.")
            return f"Desculpe, {self.trckuser}, a funcionalidade de reconhecimento facial não está disponível no momento."
        if pd is None:
            logger.error("[DeepFace Tool] Biblioteca 'pandas' não está disponível, necessária para DeepFace.find. Identificação desabilitada.")
            return f"Erro interno: {self.trckuser}, uma dependência para identificação de rostos (pandas) está faltando."

        logger.info("[DeepFace Tool] Executando _handle_identify_person_in_front.")
        start_time = time.time()

        frame_to_process: Optional[np.ndarray] = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            logger.warning("[DeepFace Tool] Nenhum frame de câmera disponível para identificar pessoa.")
            return f"{self.trckuser}, não consigo ver nada no momento para identificar alguém."

        try:
            # DeepFace.find é bloqueante. Retorna uma lista de DataFrames.
            # Normalmente, se um rosto é detectado na img_path, a lista conterá um DataFrame.
            # Se múltiplos rostos forem detectados na img_path, pode retornar múltiplos DataFrames
            # (dependendo da versão e comportamento exato do backend). Vamos focar no primeiro.
            dfs_results = DeepFace.find(
                img_path=frame_to_process,
                db_path=DB_PATH,
                model_name=DEEPFACE_MODEL_NAME,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                distance_metric=DEEPFACE_DISTANCE_METRIC,
                enforce_detection=True, # Garante que um rosto seja detectado na imagem de entrada
                align=True,
                silent=True # Suprime logs do DeepFace
            )
            
            # Verifica se dfs_results é uma lista e não está vazia, e se o primeiro item é um DataFrame não vazio
            if not dfs_results or not isinstance(dfs_results, list) or \
               not isinstance(dfs_results[0], pd.DataFrame) or dfs_results[0].empty:
                logger.info("[DeepFace Tool] Nenhuma correspondência encontrada ou rosto não detectado claramente na imagem de entrada.")
                return f"{self.trckuser}, não consegui reconhecer ninguém conhecido ou não detectei um rosto claro."

            df_matches = dfs_results[0] # Pega o DataFrame para o primeiro (e geralmente único) rosto detectado na img_path

            # O nome da coluna de distância pode variar ligeiramente (ex: 'VGG-Face_cosine')
            # Tenta encontrar a coluna de distância correta
            distance_col_name = f"{DEEPFACE_MODEL_NAME}_{DEEPFACE_DISTANCE_METRIC}" # Ex: "VGG-Face_cosine"
            if distance_col_name not in df_matches.columns:
                # Fallback para nomes comuns ou que contenham a métrica
                if 'distance' in df_matches.columns:
                    distance_col_name = 'distance'
                else:
                    found_col = None
                    for col in df_matches.columns:
                        if DEEPFACE_DISTANCE_METRIC in col.lower():
                            found_col = col
                            break
                    if not found_col:
                        logger.error(f"[DeepFace Tool] Coluna de distância ('{distance_col_name}' ou similar) não encontrada. Colunas disponíveis: {df_matches.columns.tolist()}")
                        return f"{self.trckuser}, ocorreu um erro ao processar o resultado da identificação."
                    distance_col_name = found_col
            
            df_matches = df_matches.sort_values(by=distance_col_name, ascending=True)
            best_match = df_matches.iloc[0]
            
            # Extrai o nome da pessoa do caminho da identidade (ex: DB_PATH/john_doe/img1.jpg -> john_doe)
            identity_path = best_match['identity']
            person_name = os.path.basename(os.path.dirname(str(identity_path))) # Garante que identity_path é string
            distance = best_match[distance_col_name]

            logger.info(f"[DeepFace Tool] Pessoa potencialmente identificada: '{person_name}' (Distância: {distance:.4f})")

            # Limiares de distância são cruciais e dependem do modelo e da métrica.
            # Estes são exemplos e podem precisar de ajuste fino.
            # Fonte comum para limiares: Documentação do DeepFace ou seus repositórios.
            thresholds = {
                'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86},
                'Facenet': {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 1.10}, # Euclidean para Facenet é maior
                'Facenet512': {'cosine': 0.30, 'euclidean': 23.56, 'euclidean_l2': 1.04}, # Similarmente
                'ArcFace': {'cosine': 0.68, 'euclidean': 4.15, 'euclidean_l2': 1.13},
                'Dlib': {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6}, # Dlib tem distâncias menores
                'SFace': {'cosine': 0.593, 'euclidean': 10.734, 'euclidean_l2': 1.055},
                # Adicione outros modelos e métricas conforme necessário
            }
            recognition_threshold = thresholds.get(DEEPFACE_MODEL_NAME, {}).get(DEEPFACE_DISTANCE_METRIC)
            
            if recognition_threshold is None:
                logger.warning(f"[DeepFace Tool] Limiar de reconhecimento não definido para {DEEPFACE_MODEL_NAME}/{DEEPFACE_DISTANCE_METRIC}. Usando um padrão genérico (0.5 para cosine, 1.0 para L2).")
                recognition_threshold = 0.5 if DEEPFACE_DISTANCE_METRIC == 'cosine' else 1.0


            duration = time.time() - start_time
            logger.info(f"[DeepFace Tool] Identificação concluída em {duration:.2f}s.")

            if distance <= recognition_threshold:
                return f"{self.trckuser}, a pessoa na sua frente parece ser {person_name.replace('_', ' ')}."
            else:
                logger.info(f"[DeepFace Tool] Distância {distance:.4f} > limiar ({recognition_threshold}). Não reconhecido com confiança.")
                return f"{self.trckuser}, detectei um rosto, mas não tenho certeza de quem é."

        except ValueError as ve: # Comumente de enforce_detection=True se nenhum rosto na img_path
            logger.warning(f"[DeepFace Tool] Nenhum rosto detectado na imagem de entrada (ValueError): {ve}")
            return f"{self.trckuser}, não detectei um rosto claro para identificar."
        except Exception:
            logger.exception("[DeepFace Tool] Erro inesperado ao identificar pessoa.")
            return f"{self.trckuser}, ocorreu um erro inesperado ao tentar identificar a pessoa."

    def _run_midas_inference(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Executa a inferência MiDaS em um frame para estimar a profundidade.
        Esta função é BLOQUEANTE e deve ser chamada com `asyncio.to_thread`.

        Args:
            frame_bgr (np.ndarray): O frame de entrada em formato BGR.

        Returns:
            Optional[np.ndarray]: O mapa de profundidade normalizado ou None em caso de falha.
        """
        if not self.midas_model or not self.midas_transform or not self.midas_device:
            logger.warning("[MiDaS Tool] Modelo MiDaS, transformador ou dispositivo não carregado. Não é possível estimar profundidade.")
            return None
        
        try:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_batch = self.midas_transform(img_rgb).to(self.midas_device)

            with torch.no_grad():
                prediction = self.midas_model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2], # (altura, largura)
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth_map = prediction.cpu().numpy()
            return depth_map
        except Exception:
            logger.exception("[MiDaS Tool] Erro durante a inferência MiDaS.")
            return None

    def _find_best_yolo_match(self, object_type_query: str, yolo_results_current_frame: List[Any]) -> Optional[Tuple[Dict[str, int], float, str]]:
        """
        Encontra a melhor correspondência YOLO para um tipo de objeto nos resultados atuais.
        Esta função é relativamente rápida, mas acessa `self.yolo_model.names`.

        Args:
            object_type_query (str): O tipo de objeto a ser procurado (ex: "copo", "celular").
            yolo_results_current_frame (List[Any]): Os resultados da predição YOLO para o frame atual.

        Returns:
            Optional[Tuple[Dict[str, int], float, str]]:
                - Bounding box (x1, y1, x2, y2) da melhor correspondência.
                - Confiança da melhor correspondência.
                - Nome da classe detectada.
                Retorna None se nenhuma correspondência for encontrada ou se o modelo YOLO não estiver disponível.
        """
        if not self.yolo_model or not yolo_results_current_frame:
            # logger.debug(f"[YOLO Match] Modelo YOLO ou resultados não disponíveis para encontrar '{object_type_query}'.")
            return None

        best_match_info: Optional[Tuple[Dict[str, int], float, str]] = None
        highest_confidence = -1.0
        
        # Mapeia o tipo de objeto consultado para as classes YOLO reais
        # Ex: "celular" pode mapear para "cell phone"
        target_yolo_class_names = YOLO_CLASS_MAP.get(object_type_query.lower(), [object_type_query.lower()])
        # logger.debug(f"[YOLO Match] Procurando por classes YOLO: {target_yolo_class_names} para query '{object_type_query}'.")

        for result_item in yolo_results_current_frame: # Iterar sobre cada resultado (geralmente um por imagem)
            if not hasattr(result_item, 'boxes') or not result_item.boxes:
                continue

            for box in result_item.boxes:
                if not (hasattr(box, 'cls') and hasattr(box, 'conf') and hasattr(box, 'xyxy') and \
                        box.cls.nelement() > 0 and box.conf.nelement() > 0 and box.xyxy.nelement() >= 4):
                    continue # Box inválido ou sem informações suficientes

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id < len(self.yolo_model.names):
                    detected_class_name = self.yolo_model.names[class_id]
                else:
                    # logger.warning(f"[YOLO Match] ID de classe {class_id} fora do intervalo de nomes do modelo YOLO.")
                    continue # ID de classe inválido

                if detected_class_name in target_yolo_class_names:
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        coords = list(map(int, box.xyxy[0]))
                        bbox = {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3]}
                        best_match_info = (bbox, confidence, detected_class_name)
                        # logger.debug(f"[YOLO Match] Novo melhor match para '{object_type_query}': {detected_class_name} ({confidence:.2f}) at {bbox}")
        
        # if best_match_info:
        #     logger.info(f"[YOLO Match] Melhor correspondência para '{object_type_query}': {best_match_info[2]} com conf {best_match_info[1]:.2f}.")
        # else:
        #     logger.info(f"[YOLO Match] Nenhuma correspondência encontrada para '{object_type_query}' nas classes {target_yolo_class_names}.")
        return best_match_info

    def _estimate_direction_from_bbox(self, bbox: Dict[str, int], frame_width: int) -> str:
        """
        Estima a direção de um objeto (esquerda, centro, direita) com base em sua bounding box.

        Args:
            bbox (Dict[str, int]): Bounding box do objeto {'x1', 'y1', 'x2', 'y2'}.
            frame_width (int): Largura do frame da câmera.

        Returns:
            str: Descrição da direção (ex: "à sua esquerda").
        """
        if frame_width == 0: return "em uma direção indeterminada" # Evita divisão por zero
        
        box_center_x = (bbox['x1'] + bbox['x2']) / 2.0
        
        # Divide o frame em três zonas verticais para direção
        one_third_width = frame_width / 3.0
        
        if box_center_x < one_third_width:
            return "à sua esquerda"
        elif box_center_x > (frame_width - one_third_width):
            return "à sua direita"
        else:
            return "à sua frente"

    def _check_if_object_is_on_surface(self, target_bbox: Dict[str, int], yolo_results_current_frame: List[Any]) -> bool:
        """
        Verifica se o objeto (target_bbox) parece estar sobre uma superfície (mesa, bancada, etc.)
        detectada pelo YOLO no mesmo frame.

        Args:
            target_bbox (Dict[str, int]): Bounding box do objeto de interesse.
            yolo_results_current_frame (List[Any]): Resultados YOLO do frame atual.

        Returns:
            bool: True se o objeto parece estar sobre uma superfície, False caso contrário.
        """
        if not self.yolo_model or not yolo_results_current_frame:
            return False

        # Nomes de classes YOLO que representam superfícies
        surface_class_keys = ["mesa", "mesa de jantar", "bancada", "prateleira", "escrivaninha", "cama"] # Adicionado "desk", "bed"
        surface_yolo_target_names = []
        for key in surface_class_keys:
            surface_yolo_target_names.extend(YOLO_CLASS_MAP.get(key, [])) 
        surface_yolo_target_names = list(set(surface_yolo_target_names)) # Remove duplicatas

        if not surface_yolo_target_names:
            # logger.debug("[Surface Check] Nenhuma classe YOLO de superfície definida no YOLO_CLASS_MAP.")
            return False

        target_bottom_y = target_bbox['y2']
        target_center_x = (target_bbox['x1'] + target_bbox['x2']) / 2.0

        for result_item in yolo_results_current_frame:
            if not hasattr(result_item, 'boxes') or not result_item.boxes:
                continue
            for box in result_item.boxes:
                if not (hasattr(box, 'cls') and hasattr(box, 'xyxy') and \
                        box.cls.nelement() > 0 and box.xyxy.nelement() >= 4):
                    continue

                class_id = int(box.cls[0])
                if class_id >= len(self.yolo_model.names): continue
                
                detected_class_name = self.yolo_model.names[class_id]

                if detected_class_name in surface_yolo_target_names:
                    s_coords = list(map(int, box.xyxy[0]))
                    s_x1, s_y1, s_x2, s_y2 = s_coords[0], s_coords[1], s_coords[2], s_coords[3]

                    # Lógica de alinhamento:
                    # 1. Objeto horizontalmente sobre a superfície? (centro do objeto dentro da largura da superfície)
                    is_horizontally_aligned = (s_x1 < target_center_x < s_x2)
                    
                    # 2. Objeto verticalmente próximo ao topo da superfície?
                    #    (base do objeto está perto do topo da superfície)
                    #    y_tolerance_pixels define o quão "próximo" é aceitável.
                    #    A base do objeto (target_bottom_y) deve estar um pouco acima ou sobre o topo da superfície (s_y1).
                    y_tolerance_pixels = 30 # Ajustável
                    is_vertically_aligned = (s_y1 - y_tolerance_pixels) < target_bottom_y < (s_y1 + y_tolerance_pixels * 1.5)
                                           # (permite que o objeto esteja um pouco "dentro" da superfície ou flutuando um pouco acima)

                    if is_horizontally_aligned and is_vertically_aligned:
                        # logger.debug(f"[Surface Check] Objeto em {target_bbox} parece estar na superfície '{detected_class_name}' em {s_coords}.")
                        return True
        return False

    def _handle_find_object_and_estimate_distance(self, object_description: str, object_type: str) -> str:
        """
        Localiza um objeto na visão da câmera, estima sua distância e direção.
        Esta função é BLOQUEANTE e deve ser chamada com `asyncio.to_thread`.

        Args:
            object_description (str): Descrição fornecida pelo usuário (ex: "meu celular azul").
            object_type (str): O tipo de objeto principal extraído pelo Gemini (ex: "celular").

        Returns:
            str: Uma mensagem para o usuário sobre a localização do objeto.
        """
        logger.info(f"[Find Object Tool] Executando para '{object_description}' (tipo: '{object_type}').")
        start_time = time.time()

        current_frame_bgr: Optional[np.ndarray] = None
        yolo_results_for_frame: Optional[List[Any]] = None
        frame_height, frame_width = 0, 0

        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                current_frame_bgr = self.latest_bgr_frame.copy()
                yolo_results_for_frame = self.latest_yolo_results # Pode ser None se YOLO falhou ou não rodou ainda
                if current_frame_bgr is not None: # Checagem adicional de segurança
                    frame_height, frame_width, _ = current_frame_bgr.shape
        
        if current_frame_bgr is None or frame_width == 0 or frame_height == 0:
            logger.warning("[Find Object Tool] Nenhum frame de câmera válido disponível.")
            return f"{self.trckuser}, não estou enxergando nada no momento para localizar o {object_description}."

        if not yolo_results_for_frame:
            logger.warning("[Find Object Tool] Nenhum resultado YOLO disponível para o frame atual. Tentando rodar YOLO sob demanda.")
            # Tenta rodar YOLO se não houver resultados (pode acontecer se get_frames estiver lento ou YOLO desabilitado)
            if self.yolo_model:
                try:
                    frame_rgb_temp = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)
                    yolo_results_for_frame = self.yolo_model.predict(frame_rgb_temp, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                    logger.info("[Find Object Tool] YOLO executado sob demanda.")
                except Exception:
                    logger.exception("[Find Object Tool] Falha ao executar YOLO sob demanda.")
                    yolo_results_for_frame = None # Garante que é None
            if not yolo_results_for_frame: # Se ainda não há resultados
                 return f"{self.trckuser}, não consegui processar a imagem a tempo para encontrar o {object_description}."


        # Tenta encontrar o objeto usando o object_type fornecido pelo Gemini
        best_yolo_match = self._find_best_yolo_match(object_type, yolo_results_for_frame)

        # Fallback: se não encontrou com object_type, tenta com a última palavra da descrição
        if not best_yolo_match and object_description:
            last_word_in_description = object_description.split(" ")[-1].lower()
            if last_word_in_description != object_type.lower(): # Evita busca redundante
                logger.info(f"[Find Object Tool] Nenhum '{object_type}' encontrado. Tentando fallback com a última palavra: '{last_word_in_description}'.")
                best_yolo_match = self._find_best_yolo_match(last_word_in_description, yolo_results_for_frame)
        
        if not best_yolo_match:
            logger.info(f"[Find Object Tool] Objeto '{object_description}' (tipo: '{object_type}') não encontrado via YOLO.")
            return f"{self.trckuser}, não consegui encontrar um(a) {object_description} na imagem."
            
        target_bbox, confidence, detected_class_name = best_yolo_match
        logger.info(f"[Find Object Tool] Melhor correspondência YOLO: Classe '{detected_class_name}', Conf: {confidence:.2f}, BBox: {target_bbox}")

        # Estimativas de direção e superfície
        direction_str = self._estimate_direction_from_bbox(target_bbox, frame_width)
        is_on_surface = self._check_if_object_is_on_surface(target_bbox, yolo_results_for_frame)
        surface_msg_part = "sobre uma superfície (como uma mesa ou prateleira)" if is_on_surface else ""

        # Estimativa de distância com MiDaS
        distance_steps_str = ""
        if self.midas_model and current_frame_bgr is not None: # current_frame_bgr deve existir aqui
            logger.info("[Find Object Tool] Executando MiDaS para estimativa de profundidade...")
            depth_map = self._run_midas_inference(current_frame_bgr) # Bloqueante
            
            if depth_map is not None:
                try:
                    # Calcula o centro da bounding box do objeto
                    obj_center_x = int((target_bbox['x1'] + target_bbox['x2']) / 2)
                    obj_center_y = int((target_bbox['y1'] + target_bbox['y2']) / 2)
                    
                    # Garante que as coordenadas estão dentro dos limites do mapa de profundidade
                    obj_center_y = max(0, min(obj_center_y, depth_map.shape[0] - 1))
                    obj_center_x = max(0, min(obj_center_x, depth_map.shape[1] - 1))
                    
                    depth_value_at_center = depth_map[obj_center_y, obj_center_x]

                    # Heurística para converter valor de profundidade MiDaS (inversa) para metros
                    # Estes valores são altamente empíricos e dependem do treinamento do MiDaS e da cena.
                    # MiDaS geralmente produz profundidade relativa (valores maiores = mais perto).
                    # Precisaria de calibração para conversão precisa.
                    # A heurística original parecia invertida (maior depth_value = mais longe).
                    # Vamos assumir que um valor de profundidade maior significa MAIS PERTO.
                    # E que METERS_PER_STEP é, por exemplo, 0.5 metros.
                    estimated_meters = -1.0
                    if depth_value_at_center > 1e-6: # Valor válido
                        # Exemplo de heurística (precisa de ajuste e calibração sérios):
                        # Supondo que depth_value_at_center varia de ~10 (longe) a ~300 (perto)
                        if depth_value_at_center > 250: estimated_meters = np.random.uniform(0.3, 1.0) # Muito perto
                        elif depth_value_at_center > 150: estimated_meters = np.random.uniform(1.0, 2.5) # Perto
                        elif depth_value_at_center > 75: estimated_meters = np.random.uniform(2.5, 5.0)  # Médio
                        elif depth_value_at_center > 25: estimated_meters = np.random.uniform(5.0, 10.0) # Longe
                        else: estimated_meters = np.random.uniform(10.0, 15.0) # Muito longe
                        
                        estimated_meters = max(0.3, min(estimated_meters, 20.0)) # Limita
                        num_steps = max(1, round(estimated_meters / METERS_PER_STEP))
                        distance_steps_str = f"a aproximadamente {num_steps} passo{'s' if num_steps > 1 else ''}"
                        logger.info(f"[Find Object Tool] Profundidade MiDaS no centro ({obj_center_x},{obj_center_y}): {depth_value_at_center:.4f}. Metros Estimados (heurístico): {estimated_meters:.2f}. Passos: {num_steps}.")
                    else:
                        logger.warning("[Find Object Tool] Valor de profundidade MiDaS inválido ou muito baixo no centro do objeto.")
                except IndexError:
                    logger.error(f"[Find Object Tool] Coordenadas ({obj_center_x},{obj_center_y}) fora dos limites do mapa de profundidade ({depth_map.shape}).")
                except Exception:
                    logger.exception("[Find Object Tool] Erro ao processar profundidade MiDaS.")
            else:
                logger.warning("[Find Object Tool] Falha ao gerar mapa de profundidade MiDaS.")
        else:
            logger.warning("[Find Object Tool] MiDaS não disponível. Não é possível estimar distância com profundidade.")

        # Monta a resposta para o usuário
        response_parts = [f"{self.trckuser}, o {object_description}"]
        if surface_msg_part: response_parts.append(surface_msg_part)
        if distance_steps_str: response_parts.append(distance_steps_str)
        response_parts.append(direction_str)
        
        # Concatena as partes da resposta de forma mais natural
        if len(response_parts) == 2: # Apenas nome e direção
            result_message = f"{response_parts[0]} está {response_parts[1]}."
        elif len(response_parts) > 2:
            # Ex: "o celular azul, sobre uma superfície, a aprox 2 passos, à sua frente."
            # Junta os atributos com vírgula, e o último com "e" ou diretamente.
            # Simplificado: junta com vírgulas e termina com a direção.
            main_part = response_parts[0] # "o celular azul"
            attributes = response_parts[1:-1] # ["sobre uma superfície", "a aprox 2 passos"]
            direction = response_parts[-1] # "à sua frente"
            if attributes:
                result_message = f"{main_part} está {', '.join(attributes)}, {direction}."
            else: # Só nome e direção
                result_message = f"{main_part} está {direction}."
        else: # Fallback muito básico
            result_message = f"{self.trckuser}, encontrei o {object_description} {direction_str}."

        duration = time.time() - start_time
        logger.info(f"[Find Object Tool] Concluído em {duration:.2f}s. Resposta: {result_message}")
        return result_message


    #oooters

    async def _execute_function_call(self, function_name: str, args: Dict[str, Any]) -> None:
        """
        Executa uma função local (tool) solicitada pelo Gemini.
        """
        logger.info(f"Processando Function Call: '{function_name}' com args: {args}")
        self.thinking_event.set() # Sinaliza que o sistema está ocupado com uma tarefa local

        result_message_from_tool: Optional[str] = None
        tool_executed_successfully = False

        # Lógica especial para 'save_known_face' se o nome não for fornecido nos args
        if function_name == "save_known_face" and not args.get("person_name"):
            logger.info(f"[Function Call] '{function_name}' chamado sem 'person_name'. Solicitando ao usuário.")
            self.awaiting_name_for_save_face = True
            self.pending_function_call_name = function_name # Guarda o nome da FC original
            
            # Pede ao Gemini (ou diretamente ao usuário via TTS) para fornecer o nome
            # Esta mensagem será falada pelo Gemini se a sessão estiver configurada para TTS.
            prompt_for_name = f"{self.trckuser}, qual o nome da pessoa que você gostaria de salvar?"
            if self.gemini_session:
                try:
                    # Envia a pergunta como um novo turno para o Gemini
                    await self.gemini_session.send(input=prompt_for_name, end_of_turn=True)
                    logger.info(f"Solicitação de nome para '{function_name}' enviada ao Gemini.")
                except Exception:
                    logger.exception(f"Erro ao enviar solicitação de nome para '{function_name}' ao Gemini.")
                    # Se falhar, reseta o estado para evitar ficar preso
                    self.awaiting_name_for_save_face = False
                    self.pending_function_call_name = None
                    result_message_from_tool = "Não consegui pedir o nome da pessoa." # Informa o Gemini sobre a falha
            else:
                logger.warning("Sessão Gemini inativa. Não é possível solicitar nome para save_known_face.")
                result_message_from_tool = "Sessão inativa, não pude pedir o nome."
            
            # Neste ponto, a execução da FC original é adiada. O nome virá em uma próxima mensagem do usuário.
            # O `thinking_event` será limpo quando o nome for recebido e a função real for chamada.
            # Não enviamos FunctionResponse aqui; esperamos a resposta do usuário.
            # O `thinking_event` é limpo aqui porque a "ação" imediata (pedir nome) terminou.
            # Ele será setado novamente quando o nome for recebido e a função real for chamada.
            if not result_message_from_tool: # Se o pedido de nome foi enviado com sucesso
                 self.thinking_event.clear() # Permite que o Gemini responda
                 return # Sai para aguardar o nome
            # Se houve erro ao pedir nome, `result_message_from_tool` terá uma mensagem de erro.
            # Prossegue para enviar essa falha como FunctionResponse.

        elif self.awaiting_name_for_save_face and self.pending_function_call_name:
            # Este bloco é chamado quando o Gemini envia o nome fornecido pelo usuário
            # (que foi capturado como texto em _process_gemini_responses e agora está nos args da NOVA FC implícita)
            # ou se o usuário digitou o nome e o Gemini o encaminhou.
            # No código original, o nome era pego do `response_part.text` diretamente.
            # Aqui, vamos assumir que o Gemini re-emite uma FC ou o texto é tratado antes.
            # Para simplificar e alinhar com o fluxo original: se `awaiting_name_for_save_face` é true,
            # e `function_name` é o texto que o usuário falou (o nome), então processamos.
            # Esta lógica precisa ser ajustada se o Gemini sempre re-emitir uma FC com o nome.
            # O código original parecia pegar o `response_part.text` como o nome.
            # Vamos adaptar: se `awaiting_name_for_save_face` é true, o `function_name` aqui
            # é na verdade o nome da pessoa que o Gemini transcreveu.
            
            user_provided_name = args.get("captured_name_from_user_speech") # Supõe que o Gemini envie assim
            if not user_provided_name and isinstance(args.get("person_name"), str) : # Ou se Gemini re-emitiu a FC com o nome
                user_provided_name = args.get("person_name")

            # Se o nome ainda não foi capturado (ex: Gemini não enviou como arg),
            # e a FC atual é um texto simples (que seria o nome), usamos isso.
            # Esta parte é uma adaptação da lógica original onde `response_part.text` era usado.
            # Se `function_name` não for uma das tools conhecidas, e estamos esperando um nome,
            # então `function_name` pode ser o nome.
            # Esta lógica é um pouco frágil e depende de como o Gemini responde.
            # A melhor abordagem seria o Gemini reenviar a FC `save_known_face` com o argumento `person_name` preenchido.
            # Por ora, vamos manter a lógica de que o nome vem como texto e é passado para a função.
            # O `_process_gemini_responses` deve ter tratado isso e chamado `_execute_actual_tool_with_name`.
            # Esta função `_execute_function_call` não deveria ser chamada com `function_name` sendo o nome da pessoa.
            # Portanto, o fluxo de `awaiting_name_for_save_face` deve ser tratado em `_process_gemini_responses`
            # ANTES de chamar `_execute_function_call`.

            # REVISÃO DE FLUXO:
            # 1. Gemini chama `save_known_face` sem nome.
            # 2. `_execute_function_call` seta `awaiting_name_for_save_face=True`, `pending_function_call_name="save_known_face"`, pede nome. Retorna.
            # 3. Usuário fala o nome. Gemini transcreve.
            # 4. `_process_gemini_responses` recebe o texto (nome).
            # 5. Se `awaiting_name_for_save_face` é True:
            #    - Pega o texto como `user_provided_name`.
            #    - Chama `_execute_actual_tool_with_name(self.pending_function_call_name, user_provided_name)`.
            #    - Envia FunctionResponse. Reseta flags.
            # Esta função `_execute_function_call` só deve lidar com FCs diretas.
            # A lógica de `awaiting_name` foi movida para `_handle_pending_name_submission` chamada de `_process_gemini_responses`.
            pass # Este `elif` não deve ser alcançado se a lógica de `_process_gemini_responses` estiver correta.


        # Feedback de voz pré-execução da ferramenta (opcional, mas bom para UX)
        # Removido para simplificar, pois o Gemini pode fornecer esse feedback.
        # Se necessário, pode ser adicionado aqui, enviando uma mensagem para o Gemini.

        # Executa a função da ferramenta apropriada (bloqueante, em thread)
        # Verifica se a função requer modo câmera e se está ativo
        vision_tools = ["save_known_face", "identify_person_in_front", "find_object_and_estimate_distance"]
        if self.video_mode != "camera" and function_name in vision_tools:
            logger.warning(f"[Function Call] Ferramenta '{function_name}' requer modo câmera, mas modo atual é '{self.video_mode}'.")
            result_message_from_tool = f"Desculpe, {self.trckuser}, a função '{function_name}' só está disponível quando a câmera está ativa."
        else:
            try:
                logger.info(f"[Function Call] Executando ferramenta '{function_name}' em thread...")
                if function_name == "save_known_face":
                    person_name_arg = args.get("person_name")
                    if person_name_arg and isinstance(person_name_arg, str):
                        result_message_from_tool = await asyncio.to_thread(self._handle_save_known_face, person_name_arg)
                        tool_executed_successfully = True
                    else:
                        # Este caso (nome faltando) deve ser tratado pelo fluxo `awaiting_name_for_save_face` acima.
                        # Se chegar aqui, é um erro de lógica ou Gemini não seguiu o fluxo.
                        logger.error(f"ERRO LÓGICO: Tentativa de chamar _handle_save_known_face sem 'person_name' fora do fluxo de espera.")
                        result_message_from_tool = "Erro interno: nome não fornecido corretamente para salvar rosto."
                
                elif function_name == "identify_person_in_front":
                    result_message_from_tool = await asyncio.to_thread(self._handle_identify_person_in_front)
                    tool_executed_successfully = True

                elif function_name == "find_object_and_estimate_distance":
                    desc_arg = args.get("object_description")
                    obj_type_arg = args.get("object_type")
                    if desc_arg and isinstance(desc_arg, str) and obj_type_arg and isinstance(obj_type_arg, str):
                        if not self.midas_model and not YOLO_CLASS_MAP: # Se componentes visuais chave faltam
                             result_message_from_tool = f"{self.trckuser}, desculpe, o módulo de localização de objetos não está totalmente funcional."
                        else:
                            result_message_from_tool = await asyncio.to_thread(
                                self._handle_find_object_and_estimate_distance, desc_arg, obj_type_arg
                            )
                            tool_executed_successfully = True
                    else:
                        logger.error(f"Argumentos faltando ou inválidos para '{function_name}': desc='{desc_arg}', type='{obj_type_arg}'")
                        result_message_from_tool = "Descrição ou tipo do objeto não fornecido corretamente para localização."
                else:
                    logger.warning(f"Recebida chamada para função desconhecida ou não mapeada: '{function_name}'")
                    result_message_from_tool = f"Função '{function_name}' desconhecida ou não implementada."
            
            except Exception: # Captura erros da execução da ferramenta
                logger.exception(f"Erro ao executar handler para ferramenta '{function_name}'.")
                result_message_from_tool = f"Ocorreu um erro interno ao processar a função {function_name}."

        # Envia o resultado da função de volta para o Gemini
        if result_message_from_tool is not None and self.gemini_session:
            logger.info(f"[Function Call] Resultado da ferramenta '{function_name}': '{result_message_from_tool}'")
            try:
                function_response_content = Content(
                    role="tool", # Papel correto para respostas de função
                    parts=[Part.from_function_response(
                        name=function_name, # Nome da função original que foi chamada
                        response={"result": Value(string_value=str(result_message_from_tool))} # Resultado como string
                    )]
                )
                await self.gemini_session.send(input=function_response_content) # Não deve ter end_of_turn=True
                logger.info(f"FunctionResponse para '{function_name}' enviado para Gemini.")
            except Exception:
                logger.exception(f"Erro ao enviar FunctionResponse para '{function_name}' ao Gemini.")
                # Se o envio da FunctionResponse falhar, o Gemini pode ficar esperando.
                # Pode ser necessário um tratamento mais robusto aqui, como tentar fechar e reabrir a sessão.
        elif not self.gemini_session:
            logger.warning(f"Sessão Gemini inativa. Não foi possível enviar resultado da função '{function_name}'.")

        if self.thinking_event.is_set():
            self.thinking_event.clear()
        logger.info(f"Processamento da Function Call '{function_name}' concluído.")


    async def _handle_pending_name_submission(self, user_provided_name: str) -> None:
        """
        Chamado quando o nome para uma função pendente (ex: save_known_face) é fornecido.
        Executa a função real e envia a FunctionResponse.
        """
        if not self.awaiting_name_for_save_face or not self.pending_function_call_name:
            logger.warning("_handle_pending_name_submission chamado sem estado de espera ativo.")
            return

        original_function_name = self.pending_function_call_name
        logger.info(f"Nome '{user_provided_name}' recebido para função pendente '{original_function_name}'. Processando...")
        self.thinking_event.set()

        # Limpa o estado de espera ANTES de chamar a função, para evitar loops se falhar
        self.awaiting_name_for_save_face = False
        self.pending_function_call_name = None
        result_message_fc: Optional[str] = None

        try:
            if original_function_name == "save_known_face":
                result_message_fc = await asyncio.to_thread(self._handle_save_known_face, user_provided_name)
            else:
                logger.error(f"Lógica de nome pendente não implementada para função: {original_function_name}")
                result_message_fc = f"Não sei como usar o nome '{user_provided_name}' para '{original_function_name}'."
        except Exception:
            logger.exception(f"Erro ao executar '{original_function_name}' com nome '{user_provided_name}'.")
            result_message_fc = f"Ocorreu um erro ao tentar '{original_function_name}' para '{user_provided_name}'."

        if result_message_fc is not None and self.gemini_session:
            logger.info(f"Resultado da função (com nome pendente) '{original_function_name}': '{result_message_fc}'")
            try:
                await self.gemini_session.send(
                    input=Content(
                        role="tool",
                        parts=[Part.from_function_response(
                            name=original_function_name,
                            response={"result": Value(string_value=result_message_fc)}
                        )]
                    )
                )
                logger.info(f"FunctionResponse (com nome pendente) para '{original_function_name}' enviado.")
            except Exception:
                logger.exception(f"Erro ao enviar FunctionResponse (com nome pendente) para '{original_function_name}'.")
        elif not self.gemini_session:
            logger.warning(f"Sessão inativa. Não foi possível enviar resultado da função (com nome pendente) '{original_function_name}'.")

        if self.thinking_event.is_set():
            self.thinking_event.clear()
        logger.info(f"Processamento de nome pendente para '{original_function_name}' concluído.")


