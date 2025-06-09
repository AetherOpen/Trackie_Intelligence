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
class AudioLoop:
    """
    Gerencia o loop principal do assistente multimodal, com foco em robustez,
    modularidade interna e eficiência aprimoradas.

    Esta classe orquestra a captura de áudio e vídeo, processamento de dados,
    interação com a API Gemini e execução de funções locais baseadas em
    comandos do modelo.
    """

    def __init__(self, video_mode: str = DEFAULT_MODE, show_preview: bool = False):
        """
        Inicializa a instância AudioLoopRefactored.

        Args:
            video_mode (str): O modo de operação de vídeo ("camera", "screen", ou outro).
            show_preview (bool): Se True e video_mode for "camera", exibe uma janela de preview.
        """
        logger.info(f"Inicializando AudioLoopRefactored com video_mode='{video_mode}', show_preview={show_preview}")
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.trckuser: str = cfg.get("trckuser", TRCKUSER)
            logger.info(f"Nome do usuário TRCKUSER carregado do config: '{self.trckuser}'")
        except FileNotFoundError:
            logger.warning(f"Arquivo de configuração '{CONFIG_PATH}' não encontrado. Usando TRCKUSER padrão: '{TRCKUSER}'.")
            self.trckuser = TRCKUSER
        except json.JSONDecodeError:
            logger.error(f"Erro ao decodificar JSON do arquivo '{CONFIG_PATH}'. Usando TRCKUSER padrão: '{TRCKUSER}'.")
            self.trckuser = TRCKUSER
        except Exception as e:
            logger.exception(f"Erro inesperado ao ler '{CONFIG_PATH}'. Usando TRCKUSER padrão: '{TRCKUSER}'.")
            self.trckuser = TRCKUSER

        self.video_mode: str = video_mode
        self.show_preview: bool = show_preview if video_mode == "camera" else False
        
        # Filas para comunicação entre tarefas assíncronas
        self.audio_input_gemini_queue: Optional[asyncio.Queue[bytes]] = None # Áudio do Gemini para playback
        self.multimedia_output_gemini_queue: Optional[asyncio.Queue[Dict[str, Any]]] = None # Áudio/Vídeo do usuário para Gemini
        self.command_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=50) # Para comandos internos, se necessário

        # Eventos de sincronização
        self.thinking_event: asyncio.Event = asyncio.Event() # Sinaliza quando o sistema está processando uma função local
        self.stop_event: asyncio.Event = asyncio.Event() # Sinaliza para todas as tarefas pararem

        # Estado da sessão e modelos
        self.gemini_session: Optional[genai_types.AsyncLiveSession] = None
        self.yolo_model: Optional[Any] = None # Ultralytics YOLO model
        self.midas_model: Optional[torch.nn.Module] = None
        self.midas_transform: Optional[Any] = None
        self.midas_device: Optional[torch.device] = None

        # Estado da interface e dados
        self.preview_window_active: bool = False
        self.frame_lock: threading.Lock = threading.Lock() # Protege acesso a latest_bgr_frame e latest_yolo_results
        self.latest_bgr_frame: Optional[np.ndarray] = None
        self.latest_yolo_results: Optional[List[Any]] = None # Resultados brutos do YOLO
        
        self.awaiting_name_for_save_face: bool = False # Flag para o fluxo de salvar rosto
        self.pending_function_call_name: Optional[str] = None # Nome da função pendente de nome

        self._initialize_models()

    def _initialize_models(self) -> None:
        """Carrega e pré-inicializa os modelos de IA."""
        logger.info("Inicializando modelos de IA...")
        if self.video_mode == "camera":
            try:
                self.yolo_model = load_yolo_model()
                logger.info("Modelo YOLO carregado com sucesso.")
            except Exception:
                logger.exception("Falha ao carregar modelo YOLO. Funcionalidades de detecção de objetos podem ser afetadas.")
                self.yolo_model = None # Garante que é None se falhar

        if DeepFace: # Só tenta carregar DeepFace se a importação foi bem-sucedida
            try:
                ensure_deepface_db_path()
                preload_deepface_models()
                logger.info("Modelos DeepFace pré-carregados e DB path assegurado.")
            except Exception:
                logger.exception("Falha ao pré-carregar modelos DeepFace. Funcionalidades de reconhecimento facial podem ser afetadas.")
        else:
            logger.warning("DeepFace não está disponível. Funções de reconhecimento facial serão desabilitadas.")

        try:
            self.midas_model, self.midas_transform, self.midas_device = load_midas_model()
            logger.info("Modelo MiDaS carregado com sucesso.")
        except Exception:
            logger.exception("Falha ao carregar modelo MiDaS. Estimativa de profundidade pode ser afetada.")
            self.midas_model, self.midas_transform, self.midas_device = None, None, None
        logger.info("Inicialização de modelos concluída.")

    async def send_text_to_gemini(self) -> None:
        """
        Lê input de texto do console, trata comandos de debug locais ('q', 'p')
        e envia o texto para a sessão Gemini.
        """
        logger.info(f"Pronto para receber comandos de texto. Digite 'q' para sair, 'p' para salvar rosto (debug).")
        while not self.stop_event.is_set():
            try:
                text_input = await asyncio.to_thread(input, f"{self.trckuser} message > ")

                # Limpa a fila de saída multimídia se houver nova entrada de texto,
                # para priorizar a nova interação.
                if self.multimedia_output_gemini_queue:
                    while not self.multimedia_output_gemini_queue.empty():
                        try:
                            self.multimedia_output_gemini_queue.get_nowait()
                            self.multimedia_output_gemini_queue.task_done()
                        except asyncio.QueueEmpty:
                            break # Fila já estava vazia
                        except Exception:
                            logger.exception("Erro ao limpar multimedia_output_gemini_queue antes de enviar texto.")
                            break # Evita loop infinito em caso de erro inesperado na fila
                    logger.debug("Fila de saída multimídia limpa antes de enviar novo texto.")

                if text_input.lower() == "q":
                    logger.info("Comando 'q' recebido. Sinalizando parada para todas as tarefas.")
                    self.stop_event.set()
                    break
                elif text_input.lower() == "p": # Comando de debug para salvar rosto
                    logger.info("[DEBUG] Comando 'p' recebido. Tentando salvar rosto como 'pedro_debug'.")
                    if self.video_mode == "camera":
                        if DeepFace:
                            try:
                                # Esta é uma chamada síncrona, executada em thread para não bloquear o asyncio
                                result = await asyncio.to_thread(self._handle_save_known_face, "pedro_debug")
                                logger.info(f"[DEBUG] Resultado do salvamento de rosto (pedro_debug): {result}")
                            except Exception:
                                logger.exception("[DEBUG] Erro ao tentar salvar rosto 'pedro_debug' diretamente.")
                        else:
                            logger.warning("[DEBUG] DeepFace não disponível. Não é possível salvar rosto.")
                    else:
                        logger.info("[DEBUG] Salvar rosto (comando 'p') só funciona no modo câmera.")
                    continue # Volta para o input sem enviar 'p' para o Gemini

                if self.gemini_session:
                    logger.info(f"Enviando texto para Gemini: '{text_input}'")
                    # Envia "." se o input for vazio, como no código original, mas idealmente deveria tratar isso.
                    await self.gemini_session.send(input=text_input or ".", end_of_turn=True)
                else:
                    if not self.stop_event.is_set(): # Evita log excessivo durante o desligamento
                        logger.warning("Sessão Gemini não está ativa. Não é possível enviar mensagem de texto.")
                        await asyncio.sleep(0.5) # Aguarda um pouco antes de permitir nova tentativa

            except asyncio.CancelledError:
                logger.info("Tarefa send_text_to_gemini cancelada.")
                break
            except Exception:
                logger.exception("Erro inesperado em send_text_to_gemini.")
                # Verifica se o erro indica que a sessão foi fechada
                # error_str_upper = str(e).upper()
                # if "LIVESESSION CLOSED" in error_str_upper or "LIVESESSION NOT CONNECTED" in error_str_upper:
                #     logger.info("Erro em send_text_to_gemini indica sessão fechada. Sinalizando parada.")
                #     self.stop_event.set() # A lógica de reconexão no `run` cuidará disso.
                # break # Sai do loop em caso de erro grave na sessão
                await asyncio.sleep(1) # Pausa antes de tentar ler novo input
        logger.info("Tarefa send_text_to_gemini finalizada.")

    def _process_camera_frame(self, cap: cv2.VideoCapture) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Captura um frame da câmera, realiza detecção YOLO, atualiza o preview (se ativo)
        e prepara o frame para envio.

        Args:
            cap (cv2.VideoCapture): O objeto de captura de vídeo.

        Returns:
            Tuple[Optional[Dict[str, Any]], List[str]]:
                - Um dicionário com os dados da imagem para envio (mime_type, data) ou None se falhar.
                - Uma lista de nomes de classes de perigo detectadas pelo YOLO.
        """
        ret, frame_bgr = cap.read()
        if not ret:
            logger.warning("Falha ao ler frame da câmera.")
            with self.frame_lock:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
            return None, []

        # Faz uma cópia para processamento e armazenamento
        # frame_bgr já é um novo buffer da câmera, mas copiar garante isolamento se for modificar muito
        current_frame_copy = frame_bgr.copy() 
        
        yolo_alerts: List[str] = []
        yolo_results_for_this_frame: Optional[List[Any]] = None
        display_frame_for_preview: Optional[np.ndarray] = None

        if self.yolo_model:
            frame_rgb = cv2.cvtColor(current_frame_copy, cv2.COLOR_BGR2RGB)
            try:
                results = self.yolo_model.predict(frame_rgb, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                yolo_results_for_this_frame = results # Armazena os resultados brutos

                if self.show_preview:
                    # Cria uma cópia separada para desenhar, para não afetar current_frame_copy
                    display_frame_for_preview = current_frame_copy.copy()

                for result_item in results:
                    for box in result_item.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        class_name_yolo = self.yolo_model.names[cls_id]
                        conf = float(box.conf[0])

                        if display_frame_for_preview is not None:
                            label = f"{class_name_yolo}: {conf:.2f}"
                            cv2.rectangle(display_frame_for_preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame_for_preview, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Verifica se a classe detectada está em alguma lista de perigo
                        is_dangerous = any(class_name_yolo in danger_list for danger_list in DANGER_CLASSES.values())
                        if is_dangerous and conf >= YOLO_CONFIDENCE_THRESHOLD:
                            yolo_alerts.append(class_name_yolo)
            except Exception:
                logger.exception("Erro durante a inferência YOLO.")
                yolo_results_for_this_frame = None
        elif self.show_preview: # Se não há modelo YOLO mas o preview está ativo
             display_frame_for_preview = current_frame_copy.copy()


        with self.frame_lock:
            self.latest_bgr_frame = current_frame_copy # Armazena o frame BGR original (copiado)
            self.latest_yolo_results = yolo_results_for_this_frame

        if self.show_preview and display_frame_for_preview is not None:
            try:
                cv2.imshow("Trackie YOLO Preview", display_frame_for_preview)
                cv2.waitKey(1) # Essencial para o OpenCV processar eventos da GUI
                self.preview_window_active = True
            except cv2.error as e_cv:
                # Erros comuns relacionados à ausência de um servidor X ou bibliotecas GUI
                if "DISPLAY" in str(e_cv).upper() or "GTK" in str(e_cv).upper() or \
                   "QT" in str(e_cv).upper() or "COULD NOT CONNECT TO DISPLAY" in str(e_cv).upper() or \
                   "plugin \"xcb\"" in str(e_cv).lower(): # Adicionado xcb
                    logger.warning("--------------------------------------------------------------------")
                    logger.warning("AVISO: Não foi possível mostrar a janela de preview da câmera.")
                    logger.warning("Verifique se um ambiente gráfico (X11, Wayland com XWayland) está disponível.")
                    logger.warning("Desabilitando feedback visual para esta sessão.")
                    logger.warning("--------------------------------------------------------------------")
                    self.show_preview = False # Desabilita para futuras tentativas nesta sessão
                    self.preview_window_active = False
                    # Tenta fechar qualquer janela que possa ter sido criada parcialmente
                    try: cv2.destroyAllWindows()
                    except Exception: pass
                else:
                    logger.exception("Erro inesperado no OpenCV ao tentar mostrar preview.")
                    self.show_preview = False
                    self.preview_window_active = False
            except Exception: # Captura qualquer outra exceção
                logger.exception("Erro geral ao tentar mostrar preview da câmera.")
                self.show_preview = False
                self.preview_window_active = False
        
        image_part_for_gemini: Optional[Dict[str, Any]] = None
        try:
            # Converte o frame (RGB, se já convertido para YOLO, ou BGR para RGB) para JPEG
            # Usa current_frame_copy que é BGR, para garantir consistência
            frame_to_encode_rgb = cv2.cvtColor(current_frame_copy, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(frame_to_encode_rgb)
            img.thumbnail((1024, 1024)) # Redimensiona mantendo a proporção
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg", quality=50) # Qualidade ajustável
            image_io.seek(0)
            image_part_for_gemini = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_io.read()).decode('utf-8')
            }
        except Exception:
            logger.exception("Erro ao converter frame da câmera para JPEG para envio.")
            return None, list(set(yolo_alerts)) # Retorna alertas mesmo se a imagem falhar

        return image_part_for_gemini, list(set(yolo_alerts)) # Remove duplicatas dos alertas

    async def stream_camera_frames(self) -> None:
        """
        Loop principal para capturar frames da câmera, processá-los e enviá-los
        para a fila de saída multimídia. Também envia alertas YOLO.
        """
        logger.info("Iniciando stream_camera_frames...")
        cap = None
        try:
            # Tenta abrir a câmera. cv2.VideoCapture é bloqueante.
            cap = await asyncio.to_thread(cv2.VideoCapture, 0) # TODO: Tornar o índice da câmera configurável
            
            if not cap.isOpened():
                logger.critical("Erro crítico: Não foi possível abrir a câmera. stream_camera_frames será encerrado.")
                with self.frame_lock: # Garante que o estado reflita a falha
                    self.latest_bgr_frame = None
                    self.latest_yolo_results = None
                self.stop_event.set() # Sinaliza para outras tarefas pararem se a câmera é essencial
                return

            # Configuração de FPS (pode não ser respeitada por todas as câmeras/drivers)
            target_fps = 1.0 # FPS desejado para processamento
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"FPS solicitado para câmera: {target_fps}, FPS real reportado: {actual_fps if actual_fps > 0 else 'Não disponível/Configurável'}")

            # Calcula o intervalo de sleep. Se actual_fps for válido e próximo, usa-o.
            # Senão, usa target_fps, com limites para evitar busy-waiting ou sleeps muito longos.
            sleep_interval = 1.0 / target_fps 
            if 0 < actual_fps < (target_fps * 5): # Se FPS real é razoável
                 sleep_interval = 1.0 / actual_fps
            sleep_interval = max(0.05, min(sleep_interval, 2.0))
            logger.info(f"Intervalo de processamento de frame: {sleep_interval:.3f}s (Alvo: {1.0/target_fps:.3f}s)")

            while not self.stop_event.is_set():
                if not cap.isOpened(): # Checagem adicional dentro do loop
                    logger.error("Câmera desconectada ou fechada inesperadamente durante o loop. Encerrando stream_camera_frames.")
                    self.stop_event.set()
                    break

                # _process_camera_frame é síncrono e intensivo em CPU, então roda em thread
                image_part, yolo_alerts = await asyncio.to_thread(self._process_camera_frame, cap)

                frame_was_successfully_read: bool
                with self.frame_lock:
                    frame_was_successfully_read = self.latest_bgr_frame is not None

                if not frame_was_successfully_read:
                    if not cap.isOpened(): # Se a câmera fechou E a leitura falhou
                        logger.error("Leitura do frame falhou e câmera está fechada. Encerrando stream_camera_frames.")
                        self.stop_event.set()
                        break
                    else: # Falha temporária na leitura, câmera ainda aberta
                        logger.warning("Falha temporária na leitura do frame da câmera. Tentando novamente...")
                        await asyncio.sleep(0.5) # Pausa antes de tentar novamente
                        continue
                
                # Envia a imagem para a fila de saída para o Gemini
                if image_part and self.multimedia_output_gemini_queue:
                    try:
                        if self.multimedia_output_gemini_queue.full():
                            # Descarta o mais antigo para dar espaço ao novo
                            discarded_item = await self.multimedia_output_gemini_queue.get()
                            self.multimedia_output_gemini_queue.task_done()
                            logger.debug(f"Fila de saída multimídia cheia. Frame descartado: {str(discarded_item)[:50]}...")
                        self.multimedia_output_gemini_queue.put_nowait(image_part)
                    except asyncio.QueueFull: # Deve ser raro devido à checagem .full() e descarte
                        logger.warning("Fila de saída multimídia ainda cheia após tentativa de descarte. Frame perdido.")
                    except Exception:
                        logger.exception("Erro inesperado ao colocar frame na multimedia_output_gemini_queue.")
                
                # Envia alertas YOLO para o Gemini (se houver sessão ativa)
                if yolo_alerts and self.gemini_session:
                    for alert_class_name in yolo_alerts:
                        try:
                            # TODO: Considerar tocar o som de perigo de forma assíncrona se necessário
                            # play_wav_file_sync(DANGER_SOUND_PATH) # Bloqueante, pode ser problemático aqui
                            alert_msg = f"ALERTA DE PERIGO (YOLO): Trackie, avise {self.trckuser} URGENTEMENTE que um(a) '{alert_class_name.upper()}' foi detectado!"
                            logger.info(f"Enviando alerta YOLO para Gemini: {alert_msg}")
                            await self.gemini_session.send(input=alert_msg, end_of_turn=True)
                        except Exception: # genai_errors.LiveSessionClosedError, etc.
                            logger.exception(f"Erro ao enviar alerta YOLO para '{alert_class_name}'.")
                            # Se a sessão estiver fechada, o loop principal de `run` deve tratar a reconexão.
                            # Não setar stop_event aqui diretamente por um alerta falho, a menos que seja erro de sessão.
                            # error_str = str(e_alert).upper()
                            # if "LIVESESSION CLOSED" in error_str or "LIVESESSION NOT CONNECTED" in error_str:
                            #    self.stop_event.set() # Sinaliza para reconexão
                            #    break # Sai do loop de alertas
                    if self.stop_event.is_set(): break # Se um erro de sessão ocorreu ao enviar alerta

                await asyncio.sleep(sleep_interval) # Controla a taxa de captura

        except asyncio.CancelledError:
            logger.info("Tarefa stream_camera_frames cancelada.")
        except Exception:
            logger.exception("Erro crítico em stream_camera_frames. Sinalizando parada.")
            self.stop_event.set()
        finally:
            logger.info("Finalizando stream_camera_frames...")
            if cap and cap.isOpened():
                cap.release()
                logger.info("Câmera liberada.")
            with self.frame_lock: # Limpa o último frame ao finalizar
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
            
            if self.preview_window_active:
                try:
                    cv2.destroyWindow("Trackie YOLO Preview") # Tenta fechar a janela específica
                    logger.info("Janela de preview 'Trackie YOLO Preview' fechada.")
                except Exception:
                    try: # Fallback para fechar todas as janelas OpenCV
                        cv2.destroyAllWindows()
                        logger.info("Todas as janelas OpenCV foram solicitadas a fechar.")
                    except Exception:
                        logger.warning("AVISO: Erro ao tentar fechar janelas de preview do OpenCV no finally de stream_camera_frames.")
            self.preview_window_active = False
            logger.info("stream_camera_frames concluído.")

    async def send_multimedia_realtime(self) -> None:
        """
        Consome dados da fila `multimedia_output_gemini_queue` (frames de vídeo/tela, áudio do microfone)
        e os envia para a sessão Gemini.
        """
        logger.info("send_multimedia_realtime pronto para enviar dados multimídia para Gemini...")
        try:
            while not self.stop_event.is_set():
                if self.thinking_event.is_set(): # Pausa o envio se uma função local está em execução
                    await asyncio.sleep(0.05)
                    continue

                if not self.multimedia_output_gemini_queue:
                    logger.debug("Fila de saída multimídia não inicializada. Aguardando...")
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Espera por um item na fila com timeout para não bloquear indefinidamente
                    media_data_item = await asyncio.wait_for(self.multimedia_output_gemini_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue # Sem itens na fila, tenta novamente
                except asyncio.QueueEmpty: # Redundante com TimeoutError, mas para clareza
                    continue
                except Exception: # Outros erros da fila
                    logger.exception("Erro ao obter item da multimedia_output_gemini_queue.")
                    await asyncio.sleep(0.1)
                    continue
                
                if not self.gemini_session:
                    logger.warning("Sessão Gemini não ativa em send_multimedia_realtime. Descartando item.")
                    if self.multimedia_output_gemini_queue: self.multimedia_output_gemini_queue.task_done()
                    await asyncio.sleep(0.5) # Aguarda antes de tentar processar próximo item
                    continue
                
                try:
                    # Verifica se o item é um dicionário com 'data' e 'mime_type' (imagem, áudio PCM)
                    # ou uma string (texto, embora o envio de texto principal seja por send_text_to_gemini)
                    if isinstance(media_data_item, dict) and "data" in media_data_item and "mime_type" in media_data_item:
                        # logger.debug(f"Enviando {media_data_item['mime_type']} para Gemini...")
                        await self.gemini_session.send(input=media_data_item, end_of_turn=True) # end_of_turn=True para áudio/imagem
                    elif isinstance(media_data_item, str): # Caso algum texto seja enfileirado aqui
                        logger.info(f"Enviando texto via send_multimedia_realtime (tratando como turno completo): '{media_data_item}'")
                        await self.gemini_session.send(input=media_data_item, end_of_turn=True)
                    else:
                        logger.warning(f"Tipo de mensagem desconhecido em multimedia_output_gemini_queue: {type(media_data_item)}")
                    
                    if self.multimedia_output_gemini_queue: self.multimedia_output_gemini_queue.task_done()

                except Exception as e_send: # genai_errors.LiveSessionError e outros
                    logger.error(f"Erro ao enviar dados multimídia para Gemini: {type(e_send).__name__} - {e_send}")
                    if self.multimedia_output_gemini_queue: self.multimedia_output_gemini_queue.task_done() # Garante task_done
                    
                    error_str_upper = str(e_send).upper()
                    # Condições que indicam problemas sérios de sessão/conexão
                    if any(err_key in error_str_upper for err_key in [
                        "LIVESESSION CLOSED", "LIVESESSION NOT CONNECTED", "DEADLINE EXCEEDED",
                        "RST_STREAM", "UNAVAILABLE", "GOAWAY", "INTERNALERROR"
                    ]):
                        logger.info("Erro de envio indica sessão Gemini fechada/perdida. Sinalizando parada para reconexão.")
                        # Não seta stop_event aqui, deixa o `run` tratar a reconexão.
                        # Se a sessão fechar, o `run` vai cair no except e tentar reconectar.
                        # Se o `send` falhar repetidamente, pode indicar que a sessão está ruim.
                        # O `run` loop é o principal ponto de controle para o estado da sessão.
                        await asyncio.sleep(1.0) # Pausa para evitar spam de envios falhos
                        # A sessão pode ser fechada pelo `run` loop se o erro for persistente.
                    else: # Outros erros de envio
                        # traceback.print_exc() # O logger.error acima já deve capturar isso se for exception
                        await asyncio.sleep(0.5) # Pausa antes de tentar o próximo item

        except asyncio.CancelledError:
            logger.info("Tarefa send_multimedia_realtime cancelada.")
        except Exception:
            logger.exception("Erro fatal em send_multimedia_realtime. Sinalizando parada.")
            self.stop_event.set() # Erro crítico, melhor parar tudo
        finally:
            logger.info("send_multimedia_realtime finalizado.")

    async def stream_microphone_audio(self) -> None:
        """
        Captura áudio do microfone e o envia para a fila `multimedia_output_gemini_queue`.
        """
        if not PYAUDIO_INSTANCE or not PYAUDIO_FORMAT:
            logger.error("PyAudio não inicializado corretamente. Tarefa stream_microphone_audio não pode iniciar.")
            return

        audio_stream = None
        try:
            logger.info("Configurando stream de áudio de entrada (microfone)...")
            mic_info = await asyncio.to_thread(PYAUDIO_INSTANCE.get_default_input_device_info)
            logger.info(f"Usando microfone: {mic_info['name']} (Taxa: {mic_info['defaultSampleRate']} Hz, Canais: {mic_info['maxInputChannels']})")
            
            # Abre o stream de forma síncrona em uma thread separada
            audio_stream = await asyncio.to_thread(
                PYAUDIO_INSTANCE.open,
                format=PYAUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=AUDIO_CHUNK_SIZE
            )
            logger.info("Escutando áudio do microfone...")

            while not self.stop_event.is_set():
                if self.thinking_event.is_set(): # Pausa a captura de áudio se o sistema está "pensando"
                    await asyncio.sleep(0.05)
                    continue

                if not audio_stream or not audio_stream.is_active():
                    logger.warning("Stream de áudio de entrada (microfone) não está ativo. Encerrando stream_microphone_audio.")
                    self.stop_event.set() # Pode ser um problema sério com o áudio
                    break
                
                try:
                    # Leitura do áudio é bloqueante, então roda em thread
                    audio_data_chunk = await asyncio.to_thread(
                        audio_stream.read, AUDIO_CHUNK_SIZE, exception_on_overflow=False
                    )
                    
                    if self.multimedia_output_gemini_queue:
                        audio_payload = {"data": audio_data_chunk, "mime_type": "audio/pcm"}
                        try:
                            if self.multimedia_output_gemini_queue.full():
                                discarded = await self.multimedia_output_gemini_queue.get()
                                self.multimedia_output_gemini_queue.task_done()
                                # logger.debug(f"Fila de saída multimídia cheia (áudio). Chunk descartado.")
                            self.multimedia_output_gemini_queue.put_nowait(audio_payload)
                        except asyncio.QueueFull:
                            # logger.warning("Fila de saída multimídia (áudio) ainda cheia. Chunk de áudio perdido.")
                            pass # Evita log excessivo
                        except Exception:
                            logger.exception("Erro inesperado ao colocar chunk de áudio na multimedia_output_gemini_queue.")
                
                except OSError as e_os:
                    if e_os.errno == -9988 or "Stream closed" in str(e_os) or "Input overflowed" in str(e_os).lower():
                        logger.warning(f"Stream de áudio (microfone) fechado ou com overflow (OSError: {e_os}). Encerrando stream_microphone_audio.")
                        self.stop_event.set() # Sinaliza problema com áudio
                        break
                    else: # Outro OSError
                        logger.exception("Erro de OS não tratado ao ler do stream de áudio (microfone).")
                        self.stop_event.set()
                        break
                except Exception: # Outras exceções durante a leitura
                    logger.exception("Erro durante a leitura do áudio em stream_microphone_audio.")
                    self.stop_event.set() 
                    break
        
        except asyncio.CancelledError:
            logger.info("Tarefa stream_microphone_audio cancelada.")
        except Exception: # Erros na configuração do stream, etc.
            logger.exception("Erro crítico em stream_microphone_audio (ex: configuração do stream). Sinalizando parada.")
            self.stop_event.set()
        finally:
            logger.info("Finalizando stream_microphone_audio...")
            if audio_stream:
                try:
                    if audio_stream.is_active():
                        audio_stream.stop_stream()
                    audio_stream.close()
                    logger.info("Stream de áudio de entrada (microfone) fechado.")
                except Exception:
                    logger.exception("Erro ao fechar stream de áudio de entrada (microfone).")
            logger.info("stream_microphone_audio concluído.")

    # --- Processamento de Respostas do Gemini e Chamadas de Função ---
    async def _process_gemini_responses(self) -> None:
        """
        Recebe e processa respostas da sessão Gemini, incluindo texto, áudio para playback,
        e FunctionCalls para executar ferramentas locais.
        """
        logger.info("_process_gemini_responses pronto para receber dados do Gemini...")
        try:
            if not self.gemini_session:
                logger.critical("Sessão Gemini não estabelecida em _process_gemini_responses. Encerrando tarefa.")
                self.stop_event.set() # Sinaliza para outras tarefas pararem
                return

            # Loop para processar continuamente as respostas da sessão Gemini
            while not self.stop_event.is_set():
                if not self.gemini_session: # Checagem adicional dentro do loop
                    logger.error("Sessão Gemini tornou-se indisponível durante _process_gemini_responses. Sinalizando parada.")
                    self.stop_event.set()
                    break
                
                try:
                    # Itera sobre as partes da resposta do Gemini.
                    # session.receive() é um gerador assíncrono.
                    current_turn_text_parts: List[str] = [] # Acumula texto para este turno
                    current_turn_has_function_call = False

                    async for response_part in self.gemini_session.receive():
                        if self.stop_event.is_set(): break # Verifica antes de processar

                        # Parte 1: Lidar com áudio para playback
                        if response_part.data and self.audio_input_gemini_queue: # Áudio PCM do Gemini
                            try:
                                self.audio_input_gemini_queue.put_nowait(response_part.data)
                            except asyncio.QueueFull:
                                logger.warning("Fila de áudio de entrada (do Gemini para playback) cheia. Áudio descartado.")
                            # Áudio não impede o processamento de texto ou function calls no mesmo turno.

                        # Parte 2: Lidar com Function Calls
                        if getattr(response_part, "function_call", None):
                            current_turn_has_function_call = True
                            fc = response_part.function_call
                            function_name = fc.name
                            args_dict = {key: val for key, val in fc.args.items()} # Converte para dict Python
                            logger.info(f"\n[Gemini Function Call] Recebido: '{function_name}', Args: {args_dict}")

                            await self._execute_function_call(function_name, args_dict)
                            # Após executar uma function call e enviar a FunctionResponse,
                            # o Gemini geralmente envia uma nova resposta (texto/áudio) baseada no resultado.
                            # Portanto, continuamos no loop de `receive()` para pegar essa resposta.
                            # O `thinking_event` é limpo dentro de `_execute_function_call`.
                            # Limpa texto acumulado, pois a FC inicia um novo "sub-turno".
                            if current_turn_text_parts:
                                accumulated_text_so_far = "".join(current_turn_text_parts).strip()
                                if accumulated_text_so_far: # Imprime o que foi dito ANTES da FC
                                     print(f"\n[Gemini Texto ANTES de FC]: {accumulated_text_so_far}", end="", flush=True)
                                current_turn_text_parts = []


                        # Parte 3: Lidar com texto
                        if response_part.text:
                            current_turn_text_parts.append(response_part.text)
                            # Imprime o texto incrementalmente
                            print(response_part.text, end="", flush=True) 
                        
                        # Parte 4: Fim do turno do Gemini (se não houver mais partes ou se for uma FC)
                        # O Gemini SDK lida com o conceito de "fim de turno" internamente ao enviar
                        # FunctionResponse ou quando o gerador `receive()` termina para um turno.
                        # Se `response_part.is_end_of_turn` (ou similar) estivesse disponível, seria útil.
                        # Por ora, o loop `async for` trata cada parte.
                        # Se uma FC foi processada, o texto anterior já foi impresso.

                    # Fim do loop `async for response_part`, significa que o Gemini completou seu turno atual (ou a sessão fechou)
                    if self.stop_event.is_set(): break

                    if current_turn_text_parts: # Se houve texto e não foi interrompido por FC
                        final_text_this_turn = "".join(current_turn_text_parts).strip()
                        if final_text_this_turn and not final_text_this_turn.endswith('\n'):
                            print() # Adiciona nova linha se o Gemini não enviou
                        current_turn_text_parts = [] # Reseta para o próximo turno

                    # Pequena pausa para não sobrecarregar o loop se não houver dados imediatos
                    if not current_turn_has_function_call and not response_part.text and not response_part.data: # type: ignore
                         await asyncio.sleep(0.01)


                except genai_errors.LiveSessionClosedError:
                    logger.warning("Sessão Gemini fechada (LiveSessionClosedError) enquanto recebia. Tentando reconectar via loop principal.")
                    # Não seta stop_event, deixa o `run` tratar.
                    await asyncio.sleep(1) # Pausa antes que o `run` loop tente reconectar
                    break # Sai do loop interno de `receive` para permitir que `run` reconecte
                except genai_errors.DeadlineExceededError:
                    logger.warning("Timeout (DeadlineExceededError) ao receber da sessão Gemini. Pode ser problema de rede ou do servidor.")
                    await asyncio.sleep(1) # Pausa antes de tentar continuar ou reconectar
                    # Não quebra o loop aqui, pode ser temporário. Se persistir, o `run` deve pegar.
                except Exception: # Outros erros durante o processamento da resposta
                    logger.exception("Erro durante o recebimento ou processamento de resposta do Gemini.")
                    # error_str_upper = str(e_inner_loop).upper()
                    # if any(err_key in error_str_upper for err_key in ["LIVESESSION", "DEADLINE", "RST_STREAM", "UNAVAILABLE"]):
                    #     logger.info("Erro indica que a sessão Gemini foi perdida. Sinalizando para reconexão.")
                    #     # Não seta stop_event, deixa o `run` tratar.
                    #     await asyncio.sleep(1)
                    #     break # Sai para reconexão
                    # else: # Erro não relacionado à sessão, pode ser lógica interna
                    #     traceback.print_exc() # Já logado pelo logger.exception
                    await asyncio.sleep(0.5) # Pausa antes de tentar continuar

            if self.stop_event.is_set():
                logger.info("Loop de _process_gemini_responses interrompido pelo stop_event.")

        except asyncio.CancelledError:
            logger.info("Tarefa _process_gemini_responses cancelada.")
        except Exception: # Erro crítico na configuração da tarefa ou loop externo
            logger.exception("Erro crítico em _process_gemini_responses. Sinalizando parada.")
            self.stop_event.set()
        finally:
            logger.info("_process_gemini_responses finalizado.")
            self.awaiting_name_for_save_face = False # Garante que a flag é resetada
            self.pending_function_call_name = None
            if self.thinking_event.is_set():
                self.thinking_event.clear() # Garante que o evento de "pensando" é limpo


    # --- Playback de Áudio do Gemini ---
    async def play_audio_from_gemini(self) -> None:
        """
        Consome chunks de áudio da fila `audio_input_gemini_queue` (enviados pelo Gemini)
        e os reproduz usando PyAudio.
        """
        if not PYAUDIO_INSTANCE or not PYAUDIO_FORMAT:
            logger.error("PyAudio não inicializado corretamente. Tarefa play_audio_from_gemini não pode iniciar.")
            return

        audio_output_stream = None
        try:
            logger.info("Configurando stream de áudio de saída (playback)...")
            # Tenta obter informações do dispositivo de saída padrão para logging
            try:
                out_device_info = await asyncio.to_thread(PYAUDIO_INSTANCE.get_default_output_device_info)
                logger.info(f"Usando dispositivo de saída de áudio: {out_device_info['name']} @ {out_device_info['defaultSampleRate']} Hz (esperado: {AUDIO_RECEIVE_SAMPLE_RATE} Hz)")
            except Exception:
                logger.warning(f"Não foi possível obter informações do dispositivo de saída padrão. Usando taxa padrão: {AUDIO_RECEIVE_SAMPLE_RATE} Hz.")

            audio_output_stream = await asyncio.to_thread(
                PYAUDIO_INSTANCE.open,
                format=PYAUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_RECEIVE_SAMPLE_RATE, # Taxa que o Gemini envia
                output=True
            )
            logger.info("Player de áudio (para respostas Gemini) pronto.")

            while not self.stop_event.is_set():
                if not self.audio_input_gemini_queue:
                    logger.debug("Fila de áudio de entrada (Gemini) não inicializada. Aguardando...")
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Interrompe o áudio atual se houver nova entrada de texto do usuário (experimental)
                    # Isso pode ser complexo de acertar (ex: se o usuário está respondendo ao Gemini)
                    # if self.multimedia_output_gemini_queue and not self.multimedia_output_gemini_queue.empty() and \
                    #    audio_output_stream and audio_output_stream.is_active():
                    #     logger.info("Nova entrada de usuário detectada, interrompendo áudio do Gemini.")
                    #     audio_output_stream.stop_stream()
                    #     # Limpa a fila de áudio pendente do Gemini
                    #     while not self.audio_input_gemini_queue.empty():
                    #         self.audio_input_gemini_queue.get_nowait()
                    #         self.audio_input_gemini_queue.task_done()
                    #     await asyncio.sleep(0.05) # Pequena pausa
                    #     if not audio_output_stream.is_active():
                    #         audio_output_stream.start_stream() # Reinicia para o próximo áudio

                    audio_chunk_to_play = await asyncio.wait_for(self.audio_input_gemini_queue.get(), timeout=0.5)
                    
                    if audio_chunk_to_play is None: # Sinal de encerramento da fila (enviado no cleanup)
                        logger.info("Recebido sinal de encerramento (None) para play_audio_from_gemini.")
                        break 

                    if audio_output_stream and audio_output_stream.is_active():
                        await asyncio.to_thread(audio_output_stream.write, audio_chunk_to_play)
                    else:
                        logger.warning("Stream de áudio para playback (Gemini) não está ativo. Descartando chunk de áudio.")
                    
                    if self.audio_input_gemini_queue: self.audio_input_gemini_queue.task_done()

                except asyncio.TimeoutError:
                    continue # Sem áudio na fila, tenta novamente
                except asyncio.QueueEmpty: # Redundante
                    continue
                except OSError as e_os_play:
                    if "Stream closed" in str(e_os_play): # Checagem mais robusta
                        logger.warning(f"Stream de playback (Gemini) fechado (OSError: {e_os_play}). Encerrando play_audio_from_gemini.")
                        break 
                    else:
                        logger.exception("Erro de OS não tratado ao reproduzir áudio do Gemini.")
                        break # Erro sério, encerra a tarefa
                except Exception: # Outros erros
                    logger.exception("Erro ao reproduzir áudio do Gemini (interno).")
                    # Se o stream fechar por outro motivo, pode ser pego aqui
                    if "Stream closed" in str(sys.exc_info()[1]): 
                        logger.warning("Stream de playback (Gemini) fechado (Exception). Encerrando play_audio_from_gemini.")
                        break
                    # Não quebra por erros genéricos, a menos que sejam repetitivos ou fatais.
                    await asyncio.sleep(0.1)


        except asyncio.CancelledError:
            logger.info("Tarefa play_audio_from_gemini cancelada.")
        except Exception: # Erros na configuração do stream, etc.
            logger.exception("Erro crítico em play_audio_from_gemini (ex: configuração do stream).")
            # Não seta stop_event aqui, pois o áudio pode não ser crítico para o resto.
        finally:
            logger.info("Finalizando play_audio_from_gemini...")
            if audio_output_stream:
                try:
                    if audio_output_stream.is_active():
                        audio_output_stream.stop_stream()
                    audio_output_stream.close()
                    logger.info("Stream de áudio de saída (playback Gemini) fechado.")
                except Exception:
                    logger.exception("Erro ao fechar stream de áudio de saída (playback Gemini).")
            logger.info("play_audio_from_gemini concluído.")


    # --- Loop Principal de Execução e Gerenciamento de Sessão ---
    async def run(self) -> None:
        """
        O loop principal que gerencia a conexão com o Gemini, inicializa
        e supervisiona todas as tarefas assíncronas (captura, envio, recebimento, playback).
        """
        logger.info("Iniciando AudioLoopRefactored.run()...")
        max_connection_retries = 3
        retry_delay_base_seconds = 2.0
        connection_attempt = 0

        while connection_attempt <= max_connection_retries and not self.stop_event.is_set():
            try:
                if connection_attempt > 0: # Se é uma tentativa de reconexão
                    retry_delay = retry_delay_base_seconds * (2 ** (connection_attempt -1)) # Backoff exponencial
                    logger.info(f"Tentativa de reconexão {connection_attempt}/{max_connection_retries} à sessão Gemini após {retry_delay:.1f}s...")
                    await asyncio.sleep(retry_delay)
                
                # Limpa estado da sessão anterior antes de tentar uma nova
                if self.gemini_session:
                    try: await self.gemini_session.close()
                    except Exception: pass
                self.gemini_session = None
                self.audio_input_gemini_queue = None # Áudio do Gemini para playback
                self.multimedia_output_gemini_queue = None # Áudio/Vídeo do usuário para Gemini
                self.awaiting_name_for_save_face = False
                self.pending_function_call_name = None
                if self.thinking_event.is_set(): self.thinking_event.clear()

                # Verifica se os componentes essenciais do Gemini estão disponíveis
                if not GEMINI_CLIENT:
                    logger.critical("Cliente Gemini (GEMINI_CLIENT) não inicializado. Não é possível conectar. Encerrando.")
                    self.stop_event.set()
                    break
                if not GEMINI_LIVE_CONNECT_CONFIG:
                    logger.critical("Configuração LiveConnect do Gemini (GEMINI_LIVE_CONNECT_CONFIG) não definida. Não é possível conectar. Encerrando.")
                    self.stop_event.set()
                    break
                
                logger.info(f"Tentando conectar ao Gemini (Modelo: {GEMINI_MODEL_NAME}, Tentativa {connection_attempt + 1})...")

                async with GEMINI_CLIENT.aio.live.connect(
                    model=GEMINI_MODEL_NAME, 
                    config=GEMINI_LIVE_CONNECT_CONFIG
                ) as session:
                    self.gemini_session = session
                    session_id_str = 'N/A'
                    if hasattr(session, 'session_id'): session_id_str = session.session_id
                    elif hasattr(session, '_session_id'): session_id_str = session._session_id # type: ignore
                    logger.info(f"Sessão Gemini LiveConnect estabelecida (ID: {session_id_str}). Tentativa {connection_attempt + 1} bem-sucedida.")
                    connection_attempt = 0 # Reseta contador de tentativas após sucesso

                    # Inicializa as filas de comunicação para esta sessão
                    self.audio_input_gemini_queue = asyncio.Queue() # Para áudio do Gemini para playback
                    self.multimedia_output_gemini_queue = asyncio.Queue(maxsize=150) # Para áudio/vídeo do usuário para Gemini

                    # Grupo de tarefas para gerenciar todas as corrotinas da sessão
                    async with asyncio.TaskGroup() as tg:
                        logger.info("Iniciando tarefas da sessão Gemini...")
                        
                        # Tarefa para enviar texto do console para Gemini
                        tg.create_task(self.send_text_to_gemini(), name="send_text_to_gemini_task")
                        
                        # Tarefa para enviar dados multimídia (áudio do mic, vídeo/tela) para Gemini
                        tg.create_task(self.send_multimedia_realtime(), name="send_multimedia_realtime_task")
                        
                        # Tarefa para capturar áudio do microfone (se PyAudio disponível)
                        if PYAUDIO_INSTANCE:
                            tg.create_task(self.stream_microphone_audio(), name="stream_microphone_audio_task")
                        else:
                            logger.warning("PyAudio não disponível. Captura de áudio do microfone desabilitada.")

                        # Tarefas de captura de vídeo/tela baseadas no modo
                        if self.video_mode == "camera":
                            tg.create_task(self.stream_camera_frames(), name="stream_camera_frames_task")
                        elif self.video_mode == "screen":
                            tg.create_task(self.stream_screen_frames(), name="stream_screen_frames_task")
                        
                        # Tarefa para processar respostas do Gemini (texto, áudio para playback, function calls)
                        tg.create_task(self._process_gemini_responses(), name="process_gemini_responses_task")
                        
                        # Tarefa para reproduzir áudio recebido do Gemini (se PyAudio disponível)
                        if PYAUDIO_INSTANCE:
                            tg.create_task(self.play_audio_from_gemini(), name="play_audio_from_gemini_task")
                        
                        logger.info("Todas as tarefas da sessão Gemini foram iniciadas. Aguardando conclusão ou sinal de parada...")
                    
                    # Se o TaskGroup finalizar sem self.stop_event.is_set(),
                    # pode indicar que a sessão Gemini terminou ou uma tarefa crítica falhou.
                    logger.info("TaskGroup da sessão Gemini finalizado.")
                    if not self.stop_event.is_set():
                        logger.warning("Sessão Gemini ou uma de suas tarefas principais terminou inesperadamente. Tentando reconectar...")
                        connection_attempt += 1 
                    else: # stop_event foi setado, encerra o loop de conexão
                        logger.info("Sinal de parada detectado após TaskGroup. Encerrando loop de conexão.")
                        break 
            
            except asyncio.CancelledError:
                logger.info("Loop principal (run) cancelado. Encerrando.")
                self.stop_event.set() # Garante que todas as outras partes saibam
                break
            except ExceptionGroup as eg: # Erros originados dentro do TaskGroup
                logger.error(f"Erro(s) no TaskGroup da sessão Gemini (Tentativa {connection_attempt + 1}):")
                for i, exc in enumerate(eg.exceptions):
                    logger.error(f"  Erro {i+1} no TaskGroup: {type(exc).__name__} - {exc}")
                    # traceback.print_exception(type(exc), exc, exc.__traceback__) # Log mais detalhado se necessário
                # self.stop_event.set() # Não necessariamente, pode ser um erro recuperável pela reconexão
                connection_attempt += 1
                self.gemini_session = None # Garante que a sessão é considerada perdida
            except (genai_errors.LiveSessionError, genai_errors.GoogleAPIError, Exception) as e_conn: # Erros de conexão ou outros
                logger.error(f"Erro ao conectar ou erro inesperado no loop run (Tentativa {connection_attempt + 1}): {type(e_conn).__name__} - {e_conn}")
                # traceback.print_exc() # O logger.error acima já deve capturar isso se for exception
                
                # Verifica se o erro é fatal ou se vale a pena tentar reconectar
                error_str_upper = str(e_conn).upper()
                is_auth_error = "AUTHENTICATION" in error_str_upper or "PERMISSION_DENIED" in error_str_upper
                is_critical_config_error = "API_KEY" in error_str_upper # Exemplo

                if is_auth_error or is_critical_config_error:
                    logger.critical(f"Erro crítico de autenticação ou configuração: {e_conn}. Encerrando.")
                    self.stop_event.set()
                    break # Não tenta reconectar em erros de autenticação/configuração

                connection_attempt += 1
                self.gemini_session = None # Garante que a sessão é considerada perdida
                if connection_attempt > max_connection_retries:
                    logger.error("Máximo de tentativas de reconexão ({max_connection_retries}) atingido após erro. Encerrando.")
                    self.stop_event.set()
                    break
        
        # Fim do loop while de conexão
        if not self.stop_event.is_set() and connection_attempt > max_connection_retries:
            logger.critical("Não foi possível estabelecer ou manter a conexão com Gemini após múltiplas tentativas. Encerrando.")
            self.stop_event.set()

        await self._cleanup_resources()
        logger.info("AudioLoopRefactored.run() concluído e todos os recursos limpos.")

    async def _cleanup_resources(self) -> None:
        """Realiza a limpeza final de todos os recursos."""
        logger.info("Iniciando limpeza final de recursos em AudioLoopRefactored...")
        self.stop_event.set() # Garante que todas as tarefas sejam sinalizadas para parar

        if self.gemini_session:
            try:
                logger.info("Fechando sessão Gemini LiveConnect ativa...")
                await self.gemini_session.close()
                logger.info("Sessão Gemini LiveConnect fechada.")
            except Exception:
                logger.exception("Erro ao fechar sessão Gemini LiveConnect na limpeza final.")
            finally:
                self.gemini_session = None

        # Sinaliza para a fila de playback de áudio que não haverá mais itens
        if self.audio_input_gemini_queue:
            try:
                self.audio_input_gemini_queue.put_nowait(None) # Envia None como sentinela
            except asyncio.QueueFull:
                logger.warning("Fila de áudio de entrada (Gemini playback) cheia. Não foi possível enviar sentinela de parada.")
            except Exception: # Outros erros
                logger.exception("Erro ao enviar sentinela para audio_input_gemini_queue durante limpeza.")

        # Fecha janelas OpenCV se estiverem ativas
        if self.preview_window_active:
            logger.info("Fechando janelas OpenCV (se houver)...")
            try:
                cv2.destroyAllWindows() # Tenta fechar todas as janelas
                logger.info("Janelas OpenCV foram solicitadas a fechar na limpeza final.")
            except Exception: # cv2.error ou outros
                logger.warning("AVISO: Erro ao tentar fechar janelas de preview do OpenCV na limpeza final.")
            finally:
                self.preview_window_active = False
        
        # Termina PyAudio (deve ser chamado apenas uma vez)
        if PYAUDIO_INSTANCE:
            try:
                logger.info("Terminando instância PyAudio...")
                # PyAudio.terminate() é síncrono
                await asyncio.to_thread(PYAUDIO_INSTANCE.terminate)
                logger.info("Recursos de PyAudio liberados.")
            except Exception:
                logger.exception("Erro ao terminar instância PyAudio.")
        
        # Limpa as filas principais (opcional, pois as tarefas consumidoras devem parar)
        # if self.multimedia_output_gemini_queue:
        #     while not self.multimedia_output_gemini_queue.empty():
        #         try: self.multimedia_output_gemini_queue.get_nowait()
        #         except: break
        #     logger.debug("Fila multimedia_output_gemini_queue esvaziada na limpeza.")

        logger.info("Limpeza de recursos de AudioLoopRefactored concluída.")
