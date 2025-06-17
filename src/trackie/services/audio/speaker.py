# src/trackie/services/audio/speaker.py

import asyncio
from typing import Any, Dict
import pyaudio

from ....utils.logger import get_logger

logger = get_logger(__name__)

class SpeakerService:
    """
    Gerencia a reprodução de áudio nos alto-falantes.
    """
    def __init__(self, settings: Dict[str, Any]):
        """
        Inicializa o serviço de alto-falante.

        Args:
            settings: Dicionário de configuração contendo os parâmetros de áudio.
        """
        audio_settings = settings.get("audio", {})
        self.sample_rate = audio_settings.get("receive_sample_rate", 24000)
        self.channels = audio_settings.get("channels", 1)
        self.format = pyaudio.paInt16

        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        logger.info("Serviço de Alto-falante inicializado.")

    async def stream(self, audio_queue: asyncio.Queue, stop_event: asyncio.Event):
        """
        Inicia o loop que consome áudio de uma fila e o reproduz.

        Args:
            audio_queue: A fila de onde os chunks de áudio serão consumidos.
            stop_event: O evento que sinaliza a parada da reprodução.
        """
        try:
            self.stream = await asyncio.to_thread(
                self.pyaudio_instance.open,
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            logger.info("Alto-falante pronto para reproduzir áudio.")

            while not stop_event.is_set():
                try:
                    # Espera por um chunk de áudio com um timeout
                    audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)

                    if audio_chunk is None: # Sinal de término
                        logger.info("Sinal de término recebido pelo speaker.")
                        break
                    
                    if self.stream and self.stream.is_active():
                        # A escrita é bloqueante, então a executamos em uma thread
                        await asyncio.to_thread(self.stream.write, audio_chunk)
                    
                    audio_queue.task_done()

                except asyncio.TimeoutError:
                    continue # Sem áudio na fila, continua o loop
                except asyncio.CancelledError:
                    logger.info("Streaming do alto-falante cancelado.")
                    break
                except Exception as e:
                    logger.error(f"Erro ao reproduzir áudio: {e}")
                    await asyncio.sleep(0.1)

        except Exception:
            logger.exception("Erro crítico no serviço de alto-falante.")
        finally:
            self.stop()
            logger.info("Streaming do alto-falante finalizado.")

    def stop(self):
        """Para e fecha o stream de áudio e termina a instância do PyAudio."""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            logger.info("Stream do alto-falante fechado.")
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            logger.info("Instância PyAudio do alto-falante terminada.")