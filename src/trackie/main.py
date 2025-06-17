# src/trackie/main.py

import asyncio
import argparse
import traceback
from pathlib import Path

# Importa o objeto de configurações e os modelos Pydantic
from .config.settings import settings, AppSettings

# Importa as classes e fábricas de todos os nossos módulos
from .core.application import Application
from .services.llm.factory import get_llm_service
from .services.vision.object_detector import ObjectDetector
from .services.vision.depth_estimator import DepthEstimator
from .services.vision.face_recognizer import FaceRecognizer
from .services.audio.microphone import MicrophoneService
from .services.audio.speaker import SpeakerService
from .services.video.camera import CameraService
from .ui.preview import PreviewWindow
from .tools.handlers import ToolHandlers
from .tools.registry import ToolRegistry
from .utils.logger import setup_logging, get_logger

def main():
    """Ponto de entrada principal que monta e executa a aplicação Trackie."""
    # 1. Configuração Inicial
    setup_logging()
    logger = get_logger(__name__)

    if not settings:
        logger.critical("As configurações não puderam ser carregadas. Verifique o arquivo config/settings.yml e as variáveis de ambiente.")
        return

    # 2. Argumentos da Linha de Comando
    parser = argparse.ArgumentParser(description="Trackie - Assistente IA visual e auditivo.")
    parser.add_argument(
        "--mode", type=str, default="camera", choices=["camera", "screen", "none"],
        help="Modo de operação para entrada de vídeo ('camera', 'screen', 'none')."
    )
    parser.add_argument(
        "--show_preview", action="store_true",
        help="Mostra janela com preview da câmera e detecções."
    )
    args = parser.parse_args()

    # --- 3. Montagem dos Serviços (Injeção de Dependência) ---
    app_instance = None
    try:
        logger.info("Iniciando a montagem dos serviços da aplicação...")
        
        # Converte settings para dict para facilitar a passagem
        settings_dict = settings.dict()

        # Serviços de IA e Hardware
        llm_service = get_llm_service(settings_dict)
        object_detector = ObjectDetector(settings_dict)
        depth_estimator = DepthEstimator(settings_dict)
        face_recognizer = FaceRecognizer(settings_dict)
        microphone_service = MicrophoneService(settings_dict)
        speaker_service = SpeakerService(settings_dict)

        # Serviços opcionais (UI e Vídeo)
        preview_service = PreviewWindow() if args.show_preview and args.mode == "camera" else None
        
        # O CameraService precisa de uma referência ao preview para atualizá-lo
        camera_service = None
        if args.mode == "camera":
            camera_service = CameraService(settings_dict, object_detector, preview_service)

        # Montagem das Ferramentas (Tools)
        tool_handlers = ToolHandlers(
            face_recognizer=face_recognizer,
            object_detector=object_detector,
            depth_estimator=depth_estimator,
            settings=settings_dict
        )
        tool_registry = ToolRegistry(
            handlers=tool_handlers,
            llm_service=llm_service,
            tool_definitions_path=settings.paths.tool_definitions
        )

        # 4. Criação da Instância Principal da Aplicação
        logger.info("Todos os serviços montados. Criando a instância principal da aplicação.")
        app_instance = Application(
            llm_service=llm_service,
            camera_service=camera_service,
            microphone_service=microphone_service,
            speaker_service=speaker_service,
            tool_registry=tool_registry,
            preview_window=preview_service,
            settings=settings_dict
        )
        
        # Injeta a referência da aplicação nos serviços que precisam de estado compartilhado
        if camera_service:
            camera_service.shared_state = app_instance
        tool_handlers.shared_state = app_instance

        # 5. Execução da Aplicação
        logger.info(f"Iniciando Trackie no modo: {args.mode}")
        asyncio.run(app_instance.run())

    except (ValueError, FileNotFoundError) as e:
        logger.critical(f"Erro de configuração ou arquivo não encontrado durante a inicialização: {e}")
    except KeyboardInterrupt:
        logger.info("\nInterrupção pelo teclado recebida. Encerrando...")
    except Exception:
        logger.critical("Erro fatal e inesperado no nível principal da aplicação.")
        traceback.print_exc()
    finally:
        if app_instance and app_instance.stop_event and not app_instance.stop_event.is_set():
            logger.info("Sinalizando parada para as tarefas...")
            app_instance.stop_event.set()
        logger.info("Aplicação finalizada.")

if __name__ == "__main__":
    main()