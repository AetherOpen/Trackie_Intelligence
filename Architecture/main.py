# trackie_app/main.py
import asyncio
import argparse
import os # Para verificar YOLO_MODEL_PATH
import traceback # Para o bloco finally do main

# Imports de módulos locais
# Primeiro, garanta que logger_config seja importado para configurar sys.path se necessário
from . import logger_config # Executa o código em logger_config.py
logger = logger_config.get_logger(__name__) # Obtém o logger configurado

from .app_config import DEFAULT_MODE, YOLO_MODEL_PATH, SYSTEM_INSTRUCTION_TEXT
from .external_apis import PYAUDIO_INSTANCE, GEMINI_CLIENT
from .audio_loop import AudioLoop
from .function_call import Function_Calling

# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trackie - Assistente IA visual e auditivo.")
    parser.add_argument(
        "--mode", type=str, default=DEFAULT_MODE, choices=["camera", "screen", "none"],
        help="Modo de operação para entrada de vídeo/imagem ('camera', 'screen', 'none')."
    )
    parser.add_argument(
        "--show_preview", action="store_true",
        help="Mostra janela com preview da câmera e detecções YOLO (apenas no modo 'camera')."
    )
    args = parser.parse_args()

    show_actual_preview = False
    if args.mode == "camera" and args.show_preview:
        show_actual_preview = True
        logger.info("Feedback visual da câmera (preview) ATIVADO.")
    elif args.mode != "camera" and args.show_preview:
        logger.info("Aviso: --show_preview só tem efeito com --mode camera. Ignorando.")
    else:
        logger.info("Feedback visual da câmera (preview) DESATIVADO.")

    # Verifica se o modelo YOLO existe se o modo for camera
    if args.mode == "camera":
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.critical(f"ERRO CRÍTICO: Modelo YOLO '{YOLO_MODEL_PATH}' Não encontrado.") # Mudado para critical
            logger.info("Verifique o caminho em app_config.py (BASE_DIR e YOLO_MODEL_PATH) ou baixe o modelo.")
            exit(1)

    if not PYAUDIO_INSTANCE:
        logger.critical("ERRO CRÍTICO: PyAudio não pôde ser inicializado.") # Mudado para critical
        logger.info("Verifique a instalação do PyAudio e suas dependências (como PortAudio).")
        logger.info("O programa não pode funcionar sem áudio. Encerrando.")
        exit(1)

    if not GEMINI_CLIENT:
        logger.critical("ERRO CRÍTICO: Cliente Gemini não pôde ser inicializado (verifique API Key/conexão). Encerrando.") # Mudado para critical
        exit(1)

    # Verifica se o arquivo de prompt foi carregado (SYSTEM_INSTRUCTION_TEXT deve existir)
    # A verificação de "Você é um assistente prestativo." é para o caso de o arquivo estar vazio ou com o default.
    if not SYSTEM_INSTRUCTION_TEXT or SYSTEM_INSTRUCTION_TEXT == "Você é um assistente prestativo.":
        logger.warning("AVISO: Falha ao carregar a instrução do sistema do arquivo ou arquivo não encontrado/vazio. Usando prompt padrão.")
        # exit(1) # Descomente para sair se o prompt for essencial

    main_loop_instance = None # Renomeado para evitar conflito
    try:
        logger.info(f"Iniciando Trackie no modo: {args.mode}")
        main_loop_instance = AudioLoop(video_mode=args.mode, show_preview=show_actual_preview)
        asyncio.run(main_loop_instance.run())

    except KeyboardInterrupt:
        logger.info("\nInterrupção pelo teclado recebida (Ctrl+C). Encerrando...")
        if main_loop_instance:
            logger.info("Sinalizando parada para as tarefas...")
            main_loop_instance.stop_event.set()
    except Exception as e:
        logger.critical(f"Erro inesperado e não tratado no bloco __main__: {type(e).__name__}: {e}") # Mudado para critical
        traceback.print_exc()
        if main_loop_instance:
            logger.info("Sinalizando parada devido a erro inesperado...")
            main_loop_instance.stop_event.set()
    finally:
        logger.info("Bloco __main__ finalizado.")
        # A limpeza de PyAudio é feita dentro de AudioLoop.run()
        logger.info("Programa completamente finalizado.")