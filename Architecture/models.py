# trackie_app/models.py
import os
import traceback # Mantido para logs de erro detalhados
import torch
# import torchvision # Não usado diretamente no carregamento de MiDaS via torch.hub
# import timm # Não usado diretamente no carregamento de MiDaS via torch.hub
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
from .app_config import BASE_DIR # Importa BASE_DIR para o caminho do .env

from .app_config import (
    YOLO_MODEL_PATH, DB_PATH, DEEPFACE_MODEL_NAME,
    MIDAS_MODEL_TYPE
)
from .logger_config import get_logger

logger = get_logger(__name__)

def load_yolo_model():
    """Carrega o modelo YOLO."""
    yolo_model = None
    try:
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.error(f"ERRO CRÍTICO: Modelo YOLO '{YOLO_MODEL_PATH}' Não encontrado.")
            logger.info("Verifique o caminho em BASE_DIR e YOLO_MODEL_PATH ou baixe o modelo.")
            return None # Retorna None se o modelo não for encontrado

        yolo_model = YOLO(YOLO_MODEL_PATH)
        logger.info(f"Modelo YOLO '{YOLO_MODEL_PATH}' carregado.")
    except FileNotFoundError: # Esta exceção é redundante devido à verificação os.path.exists
        logger.error(f"ERRO: Modelo YOLO não encontrado em '{YOLO_MODEL_PATH}'. YOLO desabilitado.")
        yolo_model = None
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo YOLO: {e}. YOLO desabilitado.")
        traceback.print_exc()
        yolo_model = None
    return yolo_model

def ensure_deepface_db_path():
    """Garante que o diretório do banco de dados DeepFace exista."""
    if not os.path.exists(DB_PATH):
        try:
            os.makedirs(DB_PATH)
            logger.info(f"Diretório DeepFace DB criado em: {DB_PATH}")
        except Exception as e:
            logger.error(f"Erro ao criar diretório {DB_PATH}: {e}")

def preload_deepface_models():
    """Pré-carrega os modelos DeepFace para evitar atrasos na primeira utilização."""
    try:
        logger.info("Pré-carregando modelos DeepFace...")
        ensure_deepface_db_path() # Garante que o DB_PATH exista
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # A análise com enforce_detection=False e uma ação leve é suficiente para carregar.
        DeepFace.analyze(img_path=dummy_frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        logger.info("Modelos DeepFace pré-carregados.")
    except Exception as e:
        logger.warning(f"AVISO: Erro ao pré-carregar modelos DeepFace: {e}.")
        # traceback.print_exc() # Descomente se precisar depurar o pré-carregamento

import os
import torch

def load_midas_model():
    """Carrega o modelo MiDaS e suas transformações."""
    midas_model_type = "DPT_SwinV2_L_384"  # Define o tipo do modelo
    midas_transform = None
    midas_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        logger.info(f"Carregando modelo MiDaS ({midas_model_type}) para {midas_device}...")
        
        # Salvar o diretório de cache original do torch.hub
        original_cache_dir = torch.hub.get_dir()
        
        # Definir o novo diretório de cache para baixar os pesos
        cache_dir = os.path.join(BASE_DIR, "WorkTools", "dpt_swin2_large384")
        os.makedirs(cache_dir, exist_ok=True)  # Criar o diretório se não existir
        torch.hub.set_dir(cache_dir)  # Alterar o diretório de cache temporariamente
        
        # Carregar o modelo MiDaS
        midas_model = torch.hub.load("intel-isl/MiDaS", midas_model_type)
        
        # Carregar as transformações
        midas_transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms")  # Renomeado para evitar conflito
        if midas_model_type == "MiDaS_small":
            midas_transform = midas_transforms_hub.small_transform
        else:
            midas_transform = midas_transforms_hub.dpt_transform  # Usa dpt_transform para dpt_swin2_large_384
        
        # Restaurar o diretório de cache original
        torch.hub.set_dir(original_cache_dir)
        
        # Mover o modelo para o dispositivo apropriado e ativar modo de avaliação
        midas_model.to(midas_device)
        midas_model.eval()
        logger.info("Modelo MiDaS carregado.")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo MiDaS: {e}. Estimativa de profundidade desabilitada.")
        midas_model = None
        midas_transform = None
    return midas_model, midas_transform, midas_device
