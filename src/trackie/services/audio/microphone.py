# src/trackie/services/audio/microphone.py

import asyncio
from typing import Any, Dict
import pyaudio

from ....utils.logger import get_logger

logger = get_logger(__name__)

class MicrophoneService:
    """
    Gerencia a captura de áudio do microfone.
    """
    def __init__(self, settings: Dict[str, Any]):
        """
        Inicializa o serviço de microfone.

        Args:
            settings: Dicionário de configuração contendo os parâmetros de áudio.
        """
        audio_settings = settings.get("audio", {})
        self.chunk_size = audio_settings.get("chunk_size", 1024)
        self.sample_rate = audio_settings.get("send_sample_rate", 16000)
        self.channels = audio_settings.get("channels", 1)
        self.format = pyaudio.paInt16  # Formato padrão para a maioria das aplicações

        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        logger.info("Serviço de Microfone inicializado.")

    async def stream(self, media_queue: asyncio.Queue, stop_event: asyncio.Event):
        """
        Inicia o streaming de áudio do microfone para uma fila.

        Args:
            media_queue: A fila para onde os chunks de áudio serão enviados.
            stop_event: O evento que sinaliza a parada do streaming.
        """
        try:
            device_info = await asyncio.to_thread(self.pyaudio_instance.get_default_input_device_info)
            logger.info(f"Usando microfone padrão: {device_info['name']}")

            self.stream = await asyncio.to_thread(
                self.pyaudio_instance.open,
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_info["index"]
            )
            logger.info("Microfone está escutando...")

            while not stop_event.is_set():
                try:
                    # A leitura é bloqueante, então a executamos em uma thread separada
                    audio_chunk = await asyncio.to_thread(
                        self.stream.read, self.chunk_size, exception_on_overflow=False
                    )
                    
                    # Cria o payload padronizado para a fila de mídia
                    media_payload = {"data": audio_chunk, "mime_type": "audio/pcm"}
                    
                    # Descarta chunks antigos se a fila estiver cheia para manter a baixa latência
                    if media_queue.full():
                        await media_queue.get()
                        media_queue.task_done()
                    
                    media_queue.put_nowait(media_payload)

                except asyncio.QueueFull:
                    logger.warning("Fila de mídia cheia. Chunk de áudio do microfone descartado.")
                    continue
                except OSError as e:
                    logger.error(f"Erro de OS no stream do microfone: {e}")
                    break

        except asyncio.CancelledError:
            logger.info("Streaming do microfone cancelado.")
        except Exception:
            logger.exception("Erro crítico no serviço de microfone. Sinalizando parada.")
            stop_event.set()
        finally:
            self.stop()
            logger.info("Streaming do microfone finalizado.")

    def stop(self):
        """Para e fecha o stream de áudio e termina a instância do PyAudio."""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            logger.info("Stream do microfone fechado.")
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            logger.info("Instância PyAudio do microfone terminada.")