# src/trackie/services/video/camera.py

import asyncio
import base64
import io
from typing import Any, Dict, Optional
import cv2
import numpy as np
from PIL import Image

from ....utils.logger import get_logger

logger = get_logger(__name__)

class CameraService:
    """
    Gerencia a captura e o processamento de frames da câmera.
    """
    def __init__(
        self,
        settings: Dict[str, Any],
        # TODO: Substituir 'Any' por classes de interface quando forem criadas
        object_detector: Optional[Any],
        preview_window: Optional[Any],
        shared_state: Any  # A instância da Application para acesso ao estado compartilhado
    ):
        """
        Inicializa o serviço de câmera.

        Args:
            settings: Dicionário de configuração.
            object_detector: O serviço de detecção de objetos (ex: YOLO).
            preview_window: O serviço para exibir a janela de preview.
            shared_state: A instância da Application para atualizar o estado global.
        """
        video_settings = settings.get("video", {})
        self.camera_index = video_settings.get("device_index", 0)
        self.fps = video_settings.get("fps", 1.0)
        self.image_quality = video_settings.get("jpeg_quality", 50)
        
        self.object_detector = object_detector
        self.preview_window = preview_window
        self.shared_state = shared_state # Para `latest_bgr_frame` e `latest_yolo_results`
        
        self.video_capture = None
        logger.info("Serviço de Câmera inicializado.")

    async def stream(self, media_queue: asyncio.Queue, stop_event: asyncio.Event):
        """
        Inicia o streaming de frames da câmera, processando-os e enfileirando-os.

        Args:
            media_queue: A fila para onde os dados da imagem serão enviados.
            stop_event: O evento que sinaliza a parada do streaming.
        """
        try:
            logger.info(f"Abrindo dispositivo de câmera no índice {self.camera_index}...")
            self.video_capture = await asyncio.to_thread(cv2.VideoCapture, self.camera_index)
            
            if not self.video_capture.isOpened():
                logger.critical("Erro crítico: Não foi possível abrir a câmera.")
                stop_event.set()
                return

            sleep_interval = 1.0 / self.fps
            logger.info(f"Câmera iniciada. Processando a ~{self.fps:.1f} FPS (intervalo de {sleep_interval:.2f}s).")

            while not stop_event.is_set():
                # O processamento do frame (que inclui a inferência do modelo) é bloqueante
                processed_data = await asyncio.to_thread(self._capture_and_process_frame)

                if processed_data:
                    image_payload, alerts = processed_data
                    
                    # Enfileira a imagem para o LLM
                    if image_payload and not media_queue.full():
                        media_queue.put_nowait(image_payload)
                    
                    # Enfileira alertas de perigo como mensagens de texto de alta prioridade
                    if alerts:
                        user_name = self.shared_state.trckuser # Acessa o nome do usuário do estado compartilhado
                        for alert_class in alerts:
                            alert_msg = f"ALERTA DE PERIGO: Avise {user_name} URGENTEMENTE que um(a) '{alert_class.upper()}' foi detectado!"
                            if not media_queue.full():
                                media_queue.put_nowait(alert_msg)

                await asyncio.sleep(sleep_interval)

        except asyncio.CancelledError:
            logger.info("Streaming da câmera cancelado.")
        except Exception:
            logger.exception("Erro crítico no serviço de câmera. Sinalizando parada.")
            stop_event.set()
        finally:
            self.stop()
            logger.info("Streaming da câmera finalizado.")

    def _capture_and_process_frame(self) -> Optional[tuple]:
        """
        Captura um único frame, o processa e o prepara para envio.
        Este é um método síncrono projetado para ser executado em uma thread.
        """
        if not self.video_capture:
            return None

        ret, frame_bgr = self.video_capture.read()
        if not ret:
            logger.warning("Falha ao ler frame da câmera.")
            return None

        yolo_results = None
        alerts = []

        # 1. Detecção de Objetos
        if self.object_detector:
            yolo_results = self.object_detector.detect(frame_bgr)
            # TODO: A lógica para extrair `alerts` dos `yolo_results`
            # será movida para uma função em `processing/calculations.py`
            # Por enquanto, a lógica pode ficar aqui.

        # 2. Atualiza o estado compartilhado da aplicação (thread-safe)
        with self.shared_state.frame_lock:
            self.shared_state.latest_bgr_frame = frame_bgr.copy()
            self.shared_state.latest_yolo_results = yolo_results

        # 3. Atualiza a janela de preview, se existir
        if self.preview_window:
            self.preview_window.update(frame_bgr, yolo_results)

        # 4. Prepara a imagem para envio ao LLM
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((1024, 1024)) # Redimensiona
            
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg", quality=self.image_quality)
            image_io.seek(0)
            
            image_payload = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_io.read()).decode('utf-8')
            }
            return image_payload, alerts
        except Exception as e:
            logger.error(f"Erro ao codificar frame para envio: {e}")
            return None, alerts

    def stop(self):
        """Libera o recurso da câmera."""
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
            logger.info("Dispositivo de câmera liberado.")