# src/trackie/ui/preview.py

import asyncio
from typing import Any, List, Optional
import cv2
import numpy as np

from ...utils.logger import get_logger

logger = get_logger(__name__)

class PreviewWindow:
    """
    Gerencia a janela de preview do OpenCV para exibir os frames da câmera
    e as detecções do YOLO.
    """
    def __init__(self, window_name: str = "Trackie Preview"):
        self.window_name = window_name
        self.is_active = False
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: Optional[List[Any]] = None
        self._update_event = asyncio.Event()

    def update(self, frame: np.ndarray, detections: Optional[List[Any]]):
        """
        Recebe um novo frame e detecções para serem exibidos.
        Este método é chamado por outro serviço (ex: CameraService).
        """
        self._latest_frame = frame.copy()
        self._latest_detections = detections
        self._update_event.set() # Sinaliza para o loop de renderização que há um novo frame

    def _draw_detections(self, frame: np.ndarray, detections: List[Any]) -> np.ndarray:
        """
        Desenha as bounding boxes e os labels das detecções no frame.
        """
        if not detections:
            return frame

        # TODO: Obter os nomes das classes do modelo YOLO de forma mais elegante
        # Por enquanto, assumimos que o primeiro resultado tem os nomes.
        yolo_model_names = detections[0].names if detections else []

        for result in detections:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                class_name = yolo_model_names[cls_id] if cls_id < len(yolo_model_names) else "unknown"
                label = f"{class_name}: {conf:.2f}"
                
                # Desenha o retângulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Desenha o texto
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    async def run(self, stop_event: asyncio.Event):
        """
        O loop principal que gerencia a renderização da janela.
        Deve ser executado como uma tarefa asyncio.
        """
        logger.info("Serviço de Preview UI iniciado.")
        self.is_active = True

        while not stop_event.is_set():
            try:
                # Espera pelo sinal de que um novo frame está pronto
                await asyncio.wait_for(self._update_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                # Se não houver novos frames, apenas continua o loop para checar o stop_event
                # e manter a janela responsiva.
                pass

            if self._latest_frame is not None:
                display_frame = self._latest_frame
                if self._latest_detections:
                    display_frame = self._draw_detections(display_frame, self._latest_detections)
                
                cv2.imshow(self.window_name, display_frame)
            
            # Limpa o evento para esperar pelo próximo frame
            self._update_event.clear()

            # cv2.waitKey(1) é crucial para que o OpenCV processe os eventos da GUI
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Tecla 'q' pressionada na janela de preview. Sinalizando parada.")
                stop_event.set()
                break
        
        self.destroy()
        logger.info("Serviço de Preview UI finalizado.")

    def destroy(self):
        """Fecha a janela do OpenCV."""
        if self.is_active:
            try:
                cv2.destroyWindow(self.window_name)
                self.is_active = False
                logger.info(f"Janela de preview '{self.window_name}' fechada.")
            except Exception as e:
                logger.warning(f"Erro ao fechar a janela de preview: {e}")