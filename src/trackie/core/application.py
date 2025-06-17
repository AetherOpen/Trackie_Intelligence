# src/trackie/core/application.py

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, Optional, List

from ..utils.logger import get_logger

logger = get_logger(__name__)

class Application:
    """
    Orquestra o fluxo principal da aplicação Trackie, gerenciando os serviços
    de áudio, vídeo, IA e a interação entre eles.
    """
    def __init__(
        self,
        llm_service: Any,
        camera_service: Optional[Any],
        microphone_service: Any,
        speaker_service: Any,
        tool_registry: Any,
        preview_window: Optional[Any],
        settings: Dict[str, Any]
    ):
        """
        Inicializa a aplicação com todos os serviços necessários.
        """
        logger.info("Inicializando a classe Application com serviços injetados...")
        # Serviços injetados
        self.llm_service = llm_service
        self.camera_service = camera_service
        self.microphone_service = microphone_service
        self.speaker_service = speaker_service
        self.tool_registry = tool_registry
        self.preview_window = preview_window
        self.settings = settings
        self.trckuser = settings.get("user", {}).get("name", "usuário")

        # Filas para comunicação
        self.media_to_llm_queue = asyncio.Queue(maxsize=150)
        self.audio_from_llm_queue = asyncio.Queue()

        # Eventos de sincronização
        self.stop_event = asyncio.Event()
        # O ToolRegistry agora gerencia o thinking_event, que é passado para os serviços
        self.thinking_event = self.tool_registry.thinking_event

        # --- Estado Compartilhado ---
        # Este estado é necessário para as ferramentas que precisam do frame mais recente.
        # O acesso é protegido por um lock para garantir a segurança entre threads.
        self.frame_lock = threading.Lock()
        self.latest_bgr_frame: Optional[bytes] = None
        self.latest_yolo_results: Optional[List[Any]] = None

    async def _process_llm_responses(self):
        """Consome as respostas do LLM e as encaminha."""
        logger.info("Processador de respostas do LLM iniciado.")
        try:
            async for response in self.llm_service.receive():
                if self.stop_event.is_set():
                    break

                if response.type == "audio" and response.data:
                    await self.audio_from_llm_queue.put(response.data)
                
                elif response.type == "text" and response.data:
                    # Imprime a resposta de texto do LLM no console
                    print(response.data, end="", flush=True)

                elif response.type == "function_call":
                    # O ToolRegistry lida com a execução e o envio da resposta.
                    # A execução é feita em uma nova tarefa para não bloquear este loop.
                    logger.info(f"Recebida solicitação para ferramenta: {response.data.name}")
                    asyncio.create_task(
                        self.tool_registry.execute(response.data.name, response.data.args)
                    )
        except asyncio.CancelledError:
            logger.info("Processador de respostas do LLM cancelado.")
        except Exception:
            logger.exception("Erro crítico no processador de respostas do LLM. Sinalizando parada.")
            self.stop_event.set()

    async def run(self):
        """O loop principal que inicia e supervisiona todas as tarefas da aplicação."""
        logger.info("Application.run() iniciado.")
        try:
            # Carrega o prompt do sistema do arquivo especificado nas configurações
            prompt_path = Path(self.settings["llm"]["system_prompt_path"])
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().replace("{TRCKUSER}", self.trckuser)

            await self.llm_service.connect(
                system_prompt=system_prompt,
                tools=self.tool_registry.get_definitions()
            )

            async with asyncio.TaskGroup() as tg:
                logger.info("Iniciando o grupo de tarefas principal da aplicação...")
                
                tg.create_task(self._process_llm_responses())
                tg.create_task(self.llm_service.send_media_stream(self.media_to_llm_queue, self.thinking_event))
                tg.create_task(self.microphone_service.stream(self.media_to_llm_queue, self.stop_event, self.thinking_event))
                tg.create_task(self.speaker_service.stream(self.audio_from_llm_queue, self.stop_event))

                if self.camera_service:
                    tg.create_task(self.camera_service.stream(self.media_to_llm_queue, self.stop_event))
                
                if self.preview_window:
                    tg.create_task(self.preview_window.run(self.stop_event))

                logger.info("Todas as tarefas foram iniciadas. Trackie está operacional.")

        except Exception:
            logger.exception("Erro fatal no loop de execução da aplicação.")
        finally:
            logger.info("Iniciando o processo de limpeza da aplicação...")
            self.stop_event.set()
            await self._cleanup_resources()
            logger.info("Aplicação finalizada.")

    async def _cleanup_resources(self):
        """Realiza a limpeza final de todos os recursos e serviços."""
        logger.info("Limpando recursos da aplicação...")
        
        # Envia um sinal de término para a fila do speaker para que a tarefa possa sair
        if self.audio_from_llm_queue:
            await self.audio_from_llm_queue.put(None)
        
        # Fecha a conexão com o LLM
        if self.llm_service:
            await self.llm_service.close()
        
        # Destrói a janela de UI
        if self.preview_window:
            self.preview_window.destroy()
            
        logger.info("Limpeza de recursos concluída.")
