import os
import sys 
import asyncio
import base64
import io
import json
import logging


dependency_path = r"C:\TrackieIntelligence\WorkTools\WorkingTools"


if dependency_path not in sys.path:
    sys.path.insert(0, dependency_path)
    print(f"'{dependency_path}' adicionado ao sys.path para importação de módulos.")
else:
    print(f"'{dependency_path}' já está no sys.path.")
# --- Fim da configuração do sys.path ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
import traceback
import time
import argparse
import threading
from typing import Dict, Any, Optional, List, Tuple
from playsound import playsound


# Bibliotecas de Terceiros
import cv2
import pyaudio
from PIL import Image
import mss
import pandas as pd
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration
from google.genai.types import Content, Part
from google.genai.types import GenerateContentConfig
from google.genai import errors
from google.genai.types import LiveConnectConfig, Modality
from google.protobuf.struct_pb2 import Value, Struct
from ultralytics import YOLO
import numpy as np
from deepface import DeepFace
import torch
import torchvision
import timm


# --- Constantes ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "camera"
BaseDir = "C:\TrackieIntelligence"
DANGER_SOUND_PATH = os.path.join(BaseDir, "WorkTools", "SoundBibTrackie", "Trackiedanger.wav")

# --- Inicialização do PyAudio ---
try:
    pya = pyaudio.PyAudio()
except Exception as e:
    logger.error(f"Erro ao inicializar PyAudio: {e}. O áudio não funcionará.")
    pya = None

def play_wav_file_sync(filepath):
    """Reproduz um arquivo WAV de forma síncrona."""
    try:
        logger.info(f"Reproduzindo arquivo de áudio: {filepath}")
        playsound(filepath)
        logger.info(f"Reprodução de {filepath} concluída.")
    except Exception as e:
        logger.error(f"Erro ao reproduzir o arquivo WAV {filepath}: {e}")     

# --- Caminho para o arquivo de prompt ---
SYSTEM_INSTRUCTION_PATH = os.path.join(BaseDir,"UserSettings", "Prompt's", "TrckItcs.txt")
CONFIG_PATH = os.path.join(BaseDir, "UserSettings", "trckconfig.json")
#SNOWBOY_MODEL_PATH = os.path.join(BaseDir, "WorkTools", "trackie.pmdl")

# Carregar o JSON e extrair CFG'S
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    trckuser = config_data.get('trckuser', 'Usuário Padrão')  # Valor padrão se trckuser não existir
except FileNotFoundError:
    logging.error(f"Arquivo de configuração não encontrado: {CONFIG_PATH}")
    trckuser = 'Usuário Padrão'  # Valor padrão em caso de erro
except json.JSONDecodeError as e:
    logging.error(f"Erro ao decodificar JSON em {CONFIG_PATH}: {e}")
    trckuser = 'Usuário Padrão'
except Exception as e:
    logging.error(f"Erro ao carregar configuração de {CONFIG_PATH}: {e}")
    trckuser = 'Usuário Padrão'

# YOLO
YOLO_MODEL_PATH = os.path.join(BaseDir,"WorkTools", "yolov8n.pt")
DANGER_CLASSES = {
    # (Dicionário DANGER_CLASSES inalterado - omitido para brevidade)
    'faca':             ['knife'],
    'tesoura':          ['scissors'],
    'barbeador':        ['razor'],
    'serra':            ['saw'],
    'machado':          ['axe'],
    'machadinha':       ['hatchet'],
    'arma_de_fogo':     ['gun'],
    'pistola':          ['pistol'],
    'rifle':            ['rifle'],
    'espingarda':       ['shotgun'],
    'revólver':         ['revolver'],
    'bomba':            ['bomb'],
    'granada':          ['grenade'],
    'fogo':             ['fire'],
    'chama':            ['flame'],
    'fumaça':           ['smoke'],
    'isqueiro':         ['lighter'],
    'fósforos':         ['matches'],
    'fogão':            ['stove'],
    'superfície_quente':['hot surface'],
    'vela':             ['candle'],
    'queimador':        ['burner'],
    'fio_energizado':   ['live_wire'],
    'tomada_elétrica':  ['electric_outlet'],
    'bateria':          ['battery'],
    'vidro_quebrado':   ['broken_glass'],
    'estilhaço':        ['shard'],
    'agulha':           ['needle'],
    'seringa':         ['syringe'],
    'martelo':          ['hammer'],
    'chave_de_fenda':   ['wrench'],
    'furadeira':        ['drill'],
    'motosserra':       ['chainsaw'],
    'carro':            ['car'],
    'motocicleta':      ['motorcycle'],
    'bicicleta':        ['bicycle'],
    'caminhão':         ['truck'],
    'ônibus':           ['bus'],
    'urso':             ['bear'],
    'cobra':            ['snake'],
    'aranha':           ['spider'],
    'jacaré':           ['alligator'],
    'penhasco':         ['cliff'],
    'buraco':           ['hole'],
    'escada':           ['stairs'],
}
YOLO_CONFIDENCE_THRESHOLD = 0.40
YOLO_CLASS_MAP = {
    # (Dicionário YOLO_CLASS_MAP inalterado - omitido para brevidade)
    "pessoa":                     ["person"],
    "gato":                       ["cat"],
    "cachorro":                   ["dog"],
    "coelho":                     ["rabbit"],
    "urso":                       ["bear"],
    "elefante":                   ["elephant"],
    "zebra":                      ["zebra"],
    "girafa":                     ["giraffe"],
    "vaca":                       ["cow"],
    "cavalo":                     ["horse"],
    "ovelha":                     ["sheep"],
    "macaco":                     ["monkey"],
    "bicicleta":                  ["bicycle"],
    "moto":                       ["motorcycle"],
    "carro":                      ["car"],
    "ônibus":                     ["bus"],
    "trem":                       ["train"],
    "caminhão":                   ["truck"],
    "avião":                      ["airplane"],
    "barco":                      ["boat"],
    "skate":                      ["skateboard"],
    "prancha de surf":            ["surfboard"],
    "tênis":                      ["tennis racket"],
    "mesa de jantar":             ["dining table"],
    "mesa":                       ["table", "desk", "dining table"],
    "cadeira":                    ["chair"],
    "sofá":                       ["couch", "sofa"],
    "cama":                       ["bed"],
    "vaso de planta":             ["potted plant"],
    "banheiro":                   ["toilet"],
    "televisão":                  ["tv", "tvmonitor"],
    "abajur":                     ["lamp"],
    "espelho":                    ["mirror"],
    "laptop":                     ["laptop"],
    "computador":                 ["computer", "desktop computer", "tv"],
    "teclado":                    ["keyboard"],
    "mouse":                      ["mouse"],
    "controle remoto":            ["remote"],
    "celular":                    ["cell phone"],
    "micro-ondas":                ["microwave"],
    "forno":                      ["oven"],
    "torradeira":                 ["toaster"],
    "geladeira":                  ["refrigerator"],
    "caixa de som":               ["speaker"],
    "câmera":                     ["camera"],
    "garrafa":                    ["bottle"],
    "copo":                       ["cup"],
    "taça de vinho":              ["wine glass"],
    "taça":                       ["wine glass", "cup"],
    "prato":                      ["plate", "dish"],
    "tigela":                     ["bowl"],
    "garfo":                      ["fork"],
    "faca":                       ["knife"],
    "colher":                     ["spoon"],
    "panela":                     ["pan", "pot"],
    "frigideira":                 ["skillet", "frying pan"],
    "martelo":                    ["hammer"],
    "chave inglesa":              ["wrench"],
    "furadeira":                  ["drill"],
    "parafusadeira":              ["drill"],
    "serra":                      ["saw"],
    "roçadeira":                  ["brush cutter"],
    "alicate":                    ["pliers"],
    "chave de fenda":             ["screwdriver"],
    "lanterna":                   ["flashlight"],
    "fita métrica":               ["tape measure"],
    "mochila":                    ["backpack"],
    "bolsa":                      ["handbag", "purse", "bag"],
    "carteira":                   ["wallet"],
    "óculos":                     ["glasses", "eyeglasses"],
    "relógio":                    ["clock", "watch"],
    "chinelo":                    ["sandal", "flip-flop"],
    "sapato":                     ["shoe"],
    "sanduíche":                  ["sandwich"],
    "hambúrguer":                 ["hamburger"],
    "banana":                     ["banana"],
    "maçã":                       ["apple"],
    "laranja":                    ["orange"],
    "bolo":                       ["cake"],
    "rosquinha":                  ["donut"],
    "pizza":                      ["pizza"],
    "cachorro-quente":            ["hot dog"],
    "escova de dentes":           ["toothbrush"],
    "secador de cabelo":          ["hair drier", "hair dryer"],
    "cotonete":                   ["cotton swab"],
    "sacola plástica":            ["plastic bag"],
    "livro":                      ["book"],
    "vaso":                       ["vase"],
    "bola":                       ["sports ball", "ball"],
    "bexiga":                     ["balloon"],
    "pipa":                       ["kite"],
    "luva":                       ["glove"],
    "skis":                       ["skis"],
    "snowboard":                  ["snowboard"],
    "tesoura":                    ["scissors"],
}

# DeepFace
DB_PATH = os.path.join(BaseDir,"UserSettings", "known_faces")
DEEPFACE_MODEL_NAME = 'VGG-Face'
DEEPFACE_DETECTOR_BACKEND = 'opencv'
DEEPFACE_DISTANCE_METRIC = 'cosine'

# MiDaS
MIDAS_MODEL_TYPE = "MiDaS_small"
METERS_PER_STEP = 0.7

# --- Configuração do Cliente Gemini ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    try:
        from dotenv import load_dotenv
        # Especifica o caminho para o .env se ele não estiver na raiz do projeto
        dotenv_path = os.path.join(BaseDir, '.env') # Ajuste se necessário
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            API_KEY = os.environ.get("GEMINI_API_KEY")
            logger.info(f"Chave API carregada de: {dotenv_path}")
        else:
             logger.info(f"Arquivo .env não encontrado em: {dotenv_path}")
    except ImportError:
        logger.info("Biblioteca python-dotenv não instalada. Não é possível carregar .env.")
        pass
    except Exception as e_env:
        logger.error(f"Erro ao carregar .env: {e_env}")


if not API_KEY:
    logger.info("AVISO: Chave da API Gemini não encontrada nas variáveis de ambiente ou .env.")
    # Removido o fallback para chave placeholder para maior segurança.
    # O código abaixo com a chave hardcoded será usado, mas NÃO É RECOMENDADO.
    # API_KEY = "SUA_API_KEY_AQUI" # Substitua se necessário, mas prefira variáveis de ambiente

# ATENÇÃO: A chave hardcoded abaixo ainda está presente. Remova-a ou substitua por API_KEY.
# É ALTAMENTE RECOMENDADO usar a variável API_KEY carregada acima.
try:
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"), # <-- SUBSTITUA por API_KEY ou remova se API_KEY for None
        # api_key=API_KEY, # <-- Use esta linha em vez da acima se API_KEY for carregada
        http_options=types.HttpOptions(api_version='v1alpha')
    )
    logger.info("Cliente Gemini inicializado.")
except Exception as e_client:
    logger.error(f"ERRO CRÍTICO ao inicializar cliente Gemini: {e_client}")
    logger.info("Verifique a API Key e a conexão.")
    client = None # Define como None para indicar falha
    # exit(1) # Descomente para sair se o cliente for essencial

# --- Ferramentas Gemini (Function Calling) ---
tools = [
    types.Tool(code_execution=types.ToolCodeExecution),
    types.Tool(google_search=types.GoogleSearch()),
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="save_known_face",
                description="Salva o rosto da pessoa atualmente em foco pela câmera. Se 'person_name' não for fornecido, a IA deve solicitar o nome ao usuário com uma mensagem clara, como 'Por favor, informe o nome da pessoa para salvar o rosto.' Após receber o nome, a função salva o rosto e confirma o salvamento com 'Rosto salvo com sucesso para [nome].' Se a captura falhar, retorna 'Erro: Não foi possível capturar o rosto. Tente novamente.'",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "person_name": types.Schema(
                            type=types.Type.STRING,
                            description="Nome da pessoa a ser salvo. Se omitido, a IA deve solicitar ao usuário."
                        )
                    }
                )
            ),
            types.FunctionDeclaration(
                name="identify_person_in_front",
                description="Identifica a pessoa atualmente em foco pela câmera usando o banco de dados de rostos conhecidos. Deve ser chamado apenas quando o usuário expressa explicitamente a intenção de identificar uma pessoa. Se múltiplos rostos forem detectados, retorna o mais próximo. Inclui a confiança da identificação (ex: 'Identificado como [nome] com 95% de confiança.'). Se não houver correspondência, retorna 'Pessoa não reconhecida.'",
                parameters=types.Schema(type=types.Type.OBJECT, properties={})
            ),
            types.FunctionDeclaration(
                name="find_object_and_estimate_distance",
                description="Localiza um objeto específico na visão da câmera usando detecção de objetos (YOLO) e estima sua distância em passos com MiDaS. O 'object_type' deve ser uma das categorias do modelo YOLO (ex: 'person', 'car', 'bottle'). Retorna a direção (frente, esquerda, direita), se está sobre uma superfície (ex: mesa), e a distância estimada. Se o objeto não for encontrado, retorna 'Objeto não encontrado na cena.'",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "object_description": types.Schema(
                            type=types.Type.STRING,
                            description="Descrição completa do objeto (ex: 'garrafa azul')."
                        ),
                        "object_type": types.Schema(
                            type=types.Type.STRING,
                            description="Tipo principal do objeto (ex: 'bottle'). Deve ser uma categoria válida do modelo YOLO."
                        )
                    },
                    required=["object_description", "object_type"]
                )
            )
        ]
    ),
]

# --- Carregar Instrução do Sistema do Arquivo ---
system_instruction_text = "Você é um assistente prestativo." # Prompt padrão mínimo
try:
    if not os.path.exists(SYSTEM_INSTRUCTION_PATH):
         logger.warning(f"AVISO: Arquivo de instrução do sistema não encontrado em '{SYSTEM_INSTRUCTION_PATH}'. Usando prompt padrão.")
         exit()
    else:
        with open(SYSTEM_INSTRUCTION_PATH, 'r', encoding='utf-8') as f:
            system_instruction_text = f.read()
        logger.info(f"Instrução do sistema carregada de: {SYSTEM_INSTRUCTION_PATH}")
except Exception as e:
    logger.error(f"Erro ao ler o arquivo de instrução do sistema: {e}")
    logger.info("Usando um prompt padrão mínimo.")
    traceback.print_exc()


# --- Configuração da Sessão LiveConnect Gemini ---
CONFIG = types.LiveConnectConfig(
    temperature=0.2,
    response_modalities=["audio"], # AJUSTADO: Corrigido para ser uma lista de strings
    speech_config=types.SpeechConfig(
        language_code="pt-BR",
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Orus")
        )
    ),
    tools=tools,
    system_instruction=types.Content(
        parts=[
            types.Part.from_text(text=f"O nome do seu usuário é {trckuser}, "), # Usa f-string para formatar
            types.Part.from_text(text=system_instruction_text)
        ],
        role="system"
    )
)



# --- Classe Principal do Assistente ---
class AudioLoop:
    """
    Gerencia o loop principal do assistente multimodal.
    """
    def __init__(self, video_mode: str = DEFAULT_MODE, show_preview: bool = False):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self.trckuser = cfg.get("trckuser", "Usuário")
        except Exception as e:
            logger.warning(f"Não foi possível ler {CONFIG_PATH}: {e}")
            self.trckuser = "Usuário"        
        self.video_mode = video_mode
        self.show_preview = show_preview if video_mode == "camera" else False
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None # Será recriado na conexão
        self.cmd_queue: Optional[asyncio.Queue] = asyncio.Queue(maxsize=50) # Mantido, mas não usado explicitamente no fluxo principal

        self.thinking_event = asyncio.Event()
        self.session: Optional[genai.live.AsyncLiveSession] = None
        self.yolo_model: Optional[YOLO] = None
        self.preview_window_active: bool = False
        self.stop_event = asyncio.Event()
        self.frame_lock = threading.Lock()
        self.latest_bgr_frame: Optional[np.ndarray] = None
        self.latest_yolo_results: Optional[List[Any]] = None

        # --- Novo estado para o fluxo de salvar rosto ---
        self.awaiting_name_for_save_face: bool = False

        # --- Carregamento de Modelos ---
        if self.video_mode == "camera":
            try:
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                logger.info(f"Modelo YOLO '{YOLO_MODEL_PATH}' carregado.")
            except FileNotFoundError:
                 logger.error(f"ERRO: Modelo YOLO não encontrado em '{YOLO_MODEL_PATH}'. YOLO desabilitado.")
                 self.yolo_model = None
            except Exception as e:
                logger.error(f"Erro ao carregar o modelo YOLO: {e}. YOLO desabilitado.")
                traceback.print_exc()
                self.yolo_model = None

        if not os.path.exists(DB_PATH):
            try:
                os.makedirs(DB_PATH)
                logger.info(f"Diretório DeepFace DB criado em: {DB_PATH}")
            except Exception as e:
                logger.error(f"Erro ao criar diretório {DB_PATH}: {e}")

        try:
            logger.info("Pré-carregando modelos DeepFace...")
            # Cria um frame dummy para forçar o carregamento
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            # Executa uma ação leve (emoção) sem exigir detecção
            DeepFace.analyze(img_path=dummy_frame, actions=['emotion'], enforce_detection=False)
            logger.info("Modelos DeepFace pré-carregados.")
        except Exception as e:
            logger.warning(f"AVISO: Erro ao pré-carregar modelos DeepFace: {e}.")
            # traceback.print_exc() # Descomente se precisar depurar o pré-carregamento

        self.midas_model = None
        self.midas_transform = None
        self.midas_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            logger.info(f"Carregando modelo MiDaS ({MIDAS_MODEL_TYPE}) para {self.midas_device}...")
            # Carrega o modelo MiDaS do Torch Hub
            self.midas_model = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)
            # Carrega as transformações correspondentes
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if MIDAS_MODEL_TYPE == "MiDaS_small":
                 self.midas_transform = midas_transforms.small_transform
            else:
                 # Assume DPT transform para outros modelos (pode precisar ajustar)
                 self.midas_transform = midas_transforms.dpt_transform
            # Move o modelo para o dispositivo (GPU ou CPU)
            self.midas_model.to(self.midas_device)
            # Define o modelo para modo de avaliação (desativa dropout, etc.)
            self.midas_model.eval()
            logger.info("Modelo MiDaS carregado.")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo MiDaS: {e}. Estimativa de profundidade desabilitada.")
            # traceback.print_exc() # Descomente para depurar o carregamento do MiDaS
            self.midas_model = None
            self.midas_transform = None
          
 


    async def send_text(self):
        """Lê input de texto do usuário, trata comandos de debug e envia ao Gemini."""
        logger.info("Pronto para receber comandos de texto. Digite 'q' para sair, 'p' para salvar rosto (debug).")
        while not self.stop_event.is_set():
            try:
                # Lê input do usuário em uma thread separada para não bloquear asyncio
                text = await asyncio.to_thread(input, "message > ")
                if self.out_queue:
                    while not self.out_queue.empty():
                        self.out_queue.get_nowait()
                        self.out_queue.task_done()
                        

                # --- Tratamento de Comandos Locais/Debug ---   
                if text.lower() == "q":
                    self.stop_event.set()
                    logger.info("Sinal de parada ('q') recebido. Encerrando...")
                    break # Sai do loop while

                # --- ADIÇÃO: Comando de Debug 'p' ---
                elif text.lower() == "p":
                    logger.info("[DEBUG] Comando 'p' recebido. Tentando salvar rosto como 'pedro'...")
                    if self.video_mode == "camera":
                        # Chama a função de salvar rosto diretamente em outra thread
                        try:
                            # Bloqueia o 'pensamento' para evitar conflitos? Opcional.
                            # self.thinking_event.set()
                            logger.info("  [DEBUG] Chamando _handle_save_known_face('pedro')...")
                            result = await asyncio.to_thread(self._handle_save_known_face, "pedro")
                            logger.info(f"  [DEBUG] Resultado do salvamento direto: {result}")
                        except Exception as e_debug_save:
                            logger.info(f"  [DEBUG] Erro ao tentar salvar rosto diretamente: {e_debug_save}")
                            traceback.print_exc()
                        # finally:
                            # self.thinking_event.clear()
                    else:
                        logger.info("  [DEBUG] Salvar rosto só funciona no modo câmera.")
                    continue # Pula o envio para Gemini e espera o próximo input

                # --- Envio Normal para Gemini ---
                # Verifica se a sessão existe e está ativa antes de enviar
                if self.session:
                    logger.info(f"Enviando texto para Gemini: '{text}'")
                    # Envia o texto para o Gemini, marcando o fim do turno do usuário
                    # Envia "." se o texto estiver vazio para manter a sessão ativa
                    await self.session.send(input=text or ".", end_of_turn=True)
                else:
                    # Avisa se a sessão não estiver ativa
                    if not self.stop_event.is_set():
                        logger.info("Sessão Gemini não está ativa. Não é possível enviar mensagem.")
                        await asyncio.sleep(0.5) # Evita spamming da mensagem
                        

            except asyncio.CancelledError:
                logger.info("send_text cancelado.")
                break
            except Exception as e:
                logger.error(f"Erro em send_text: {e}")
                # Adiciona traceback para depuração em caso de erro inesperado
                # traceback.print_exc()
                # Verifica se o erro indica sessão fechada para parar
                error_str_upper = str(e).upper()
                if "LIVESESSION CLOSED" in error_str_upper or "LIVESESSION NOT CONNECTED" in error_str_upper:
                    logger.info("Erro em send_text indica sessão fechada. Sinalizando parada.")
                    self.stop_event.set()
                # Considerar parar em outros erros graves também?
                # else: self.stop_event.set()
                break # Sai do loop em caso de erro
        logger.info("send_text finalizado.")


    def _get_frame(self, cap: cv2.VideoCapture) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        # (Função _get_frame inalterada - omitida para brevidade)
        ret, frame = cap.read()
        latest_frame_copy = None
        current_yolo_results = None

        if ret:
            latest_frame_copy = frame.copy()

        yolo_alerts = []
        display_frame = None
        if ret and self.yolo_model:
            frame_rgb = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
            try:
                results = self.yolo_model.predict(frame_rgb, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
                current_yolo_results = results

                if self.show_preview:
                    display_frame = latest_frame_copy.copy()

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        class_name_yolo = self.yolo_model.names[cls_id] # Renomeado para evitar conflito
                        conf = float(box.conf[0])

                        if display_frame is not None:
                            label = f"{class_name_yolo} {conf:.2f}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Verifica se class_name_yolo está em alguma das listas dentro de DANGER_CLASSES.values()
                        is_dangerous = any(class_name_yolo in danger_list for danger_list in DANGER_CLASSES.values())
                        if is_dangerous and conf >= YOLO_CONFIDENCE_THRESHOLD:
                            yolo_alerts.append(class_name_yolo)
            except Exception as e:
                logger.error(f"Erro na inferência YOLO: {e}")
                # traceback.print_exc() # Descomente para depuração detalhada do YOLO
                current_yolo_results = None
        elif self.show_preview and ret:
            display_frame = latest_frame_copy.copy()

        with self.frame_lock:
            if ret:
                self.latest_bgr_frame = latest_frame_copy
                self.latest_yolo_results = current_yolo_results
            else:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
                return None, []

        if self.show_preview and display_frame is not None:
            try:
                cv2.imshow("Trackie YOLO Preview", display_frame)
                cv2.waitKey(1)
                self.preview_window_active = True
            except cv2.error as e:
                if "DISPLAY" in str(e).upper() or "GTK" in str(e).upper() or "QT" in str(e).upper() or "COULD NOT CONNECT TO DISPLAY" in str(e).upper() or "plugin \"xcb\"" in str(e):
                    logger.info("--------------------------------------------------------------------")
                    logger.info("AVISO: Não foi possível mostrar a janela de preview da câmera.")
                    logger.info("Desabilitando feedback visual para esta sessão.")
                    logger.info("--------------------------------------------------------------------")
                    self.show_preview = False
                    self.preview_window_active = False
                else:
                    logger.error(f"Erro inesperado no OpenCV ao tentar mostrar preview: {e}")
                    # traceback.print_exc() # Descomente para depuração detalhada do OpenCV
            except Exception as e_gen:
                logger.error(f"Erro geral ao tentar mostrar preview: {e_gen}")
                # traceback.print_exc() # Descomente para depuração detalhada
                self.show_preview = False
                self.preview_window_active = False

        image_part = None
        if ret:
            try:
                if 'frame_rgb' not in locals() or frame_rgb is None: # Adicionado 'or frame_rgb is None'
                     frame_rgb = cv2.cvtColor(latest_frame_copy, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.thumbnail([1024, 1024]) # Redimensiona mantendo proporção
                image_io = io.BytesIO()
                img.save(image_io, format="jpeg", quality=85) # Ajusta qualidade JPEG
                image_io.seek(0)
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_io.read()).decode('utf-8')
                }
            except Exception as e:
                logger.error(f"Erro na conversão do frame para JPEG: {e}")
                # traceback.print_exc() # Descomente para depuração detalhada

        return image_part, list(set(yolo_alerts))


    async def get_frames(self):
        # (Função get_frames inalterada, exceto por traceback em erro crítico - omitida para brevidade)
        cap = None
        try:
            logger.info("Iniciando captura da câmera...")
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            target_fps = 1
            # Tenta definir FPS, mas não é garantido que funcione em todas as câmeras/drivers
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"FPS solicitado: {target_fps}, FPS real da câmera: {actual_fps if actual_fps > 0 else 'Não disponível'}")

            if actual_fps > 0 and actual_fps < target_fps * 5: # Usa FPS real se razoável
                sleep_interval = 1 / actual_fps
            else:
                sleep_interval = 1 / target_fps # Usa FPS alvo como fallback
            sleep_interval = max(0.1, min(sleep_interval, 1.0)) # Limita entre 0.1s e 1.0s
            logger.info(f"Intervalo de captura de frame: {sleep_interval:.2f}s")


            if not cap.isOpened():
                logger.info("Erro: Não foi possível abrir a câmera.")
                with self.frame_lock:
                    self.latest_bgr_frame = None
                    self.latest_yolo_results = None
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                if not cap.isOpened():
                    logger.info("Câmera desconectada ou fechada inesperadamente.")
                    self.stop_event.set()
                    break

                # Executa a captura e processamento síncrono em outra thread
                image_part, yolo_alerts = await asyncio.to_thread(self._get_frame, cap)

                with self.frame_lock:
                    frame_was_read = self.latest_bgr_frame is not None

                if not frame_was_read:
                     if not cap.isOpened():
                         logger.info("Leitura do frame falhou e câmera fechada. Encerrando get_frames.")
                         self.stop_event.set()
                         break
                     else:
                         logger.info("Aviso: Falha temporária na leitura do frame.")
                         await asyncio.sleep(0.5) # Espera um pouco antes de tentar de novo
                         continue

                # Envia frame para a fila de saída
                if image_part is not None and self.out_queue:
                    try:
                        if self.out_queue.full():
                            # Descarta o mais antigo para dar espaço ao novo
                            discarded = await self.out_queue.get()
                            self.out_queue.task_done() # Marca o descartado como concluído
                            # logger.info("Aviso: Fila de saída cheia, descartando frame antigo.") # Log opcional
                        self.out_queue.put_nowait(image_part)
                    except asyncio.QueueFull:
                         # Isso não deveria acontecer se a lógica acima estiver correta, mas por segurança
                         pass # logger.info("Aviso: Fila de saída ainda cheia ao tentar enfileirar frame.") # Log opcional
                    except Exception as q_e:
                         logger.error(f"Erro inesperado ao manipular out_queue em get_frames: {q_e}")


                # Envia alertas YOLO urgentes diretamente
                if yolo_alerts and self.session:
                    for alert_class_name in yolo_alerts:
                        try:
                            alert_msg = f"YOLO DIZENDO (DETECÇÃO DE PERIGOS):TRACKIE CUIDADO! AVISE AO {self.trckuser} QUE {alert_class_name.upper()} FOI DETECTADO!"
                           # play_wav_file_sync(DANGER_SOUND_PATH)
                            # Não precisa verificar out_queue aqui, pois envia direto
                            await self.session.send(input=alert_msg, end_of_turn=True)
                            logger.info(f"ALERTA URGENTE ENVIADO: {alert_msg}")
                        except Exception as e:
                            logger.error(f"Erro ao enviar alerta urgente: {e}")
                            if "LiveSession closed" in str(e) or "LiveSession not connected" in str(e):
                                logger.info("Erro ao enviar alerta indica sessão fechada. Sinalizando parada.")
                                self.stop_event.set()
                                break # Sai do loop de alertas se a sessão fechar

                # Aguarda antes do próximo ciclo
                await asyncio.sleep(sleep_interval)

        except asyncio.CancelledError:
            logger.info("get_frames cancelado.")
        except Exception as e:
            logger.error(f"Erro crítico em get_frames: {e}")
            traceback.print_exc() # Imprime traceback para erros críticos
            self.stop_event.set()
        finally:
            logger.info("Finalizando get_frames...")
            if cap and cap.isOpened():
                cap.release()
                logger.info("Câmera liberada.")
            # Garante que o estado do frame seja limpo
            with self.frame_lock:
                self.latest_bgr_frame = None
                self.latest_yolo_results = None
            # Garante que a janela de preview seja fechada
            if self.preview_window_active:
                try:
                    # Tenta fechar a janela específica primeiro
                    cv2.destroyWindow("Trackie YOLO Preview")
                    logger.info("Janela de preview 'Trackie YOLO Preview' fechada.")
                except Exception:
                    try:
                        # Se falhar, tenta fechar todas as janelas OpenCV
                        cv2.destroyAllWindows()
                        logger.info("Todas as janelas OpenCV fechadas.")
                    except Exception as e_cv_destroy_all:
                        logger.warning(f"AVISO: erro ao tentar fechar janelas de preview no finally: {e_cv_destroy_all}")
            self.preview_window_active = False # Garante que o estado está correto
            logger.info("get_frames concluído.")


    def _get_screen(self) -> Optional[Dict[str, Any]]:
        # (Função _get_screen inalterada - omitida para brevidade)
        sct = mss.mss()
        monitor_number = 1 # Tenta usar o monitor 1 (geralmente o principal)
        try:
            monitors = sct.monitors
            if len(monitors) > monitor_number:
                 monitor = monitors[monitor_number]
            elif monitors: # Se não houver monitor 1, mas houver algum monitor (geralmente o monitor 0 é 'todos')
                 monitor = monitors[0] # Usa o primeiro monitor disponível (pode ser 'todos')
                 if len(monitors) > 1: # Se houver mais de um, pega o segundo (índice 1), que geralmente é o primário real
                     monitor = monitors[1]
            else: # Nenhum monitor encontrado
                logger.info("Erro: Nenhum monitor detectado por mss.")
                return None

            # Captura a imagem do monitor selecionado
            sct_img = sct.grab(monitor)

            # Cria a imagem PIL a partir dos dados brutos BGR
            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb, 'raw', 'BGR')
            # Não precisa converter para RGB se salvar como PNG, mas não faz mal
            # img = img.convert('RGB')

            # Salva em memória como PNG (melhor para capturas de tela que JPEG)
            image_io = io.BytesIO()
            img.save(image_io, format="PNG")
            image_io.seek(0)

            # Codifica em Base64 e retorna no formato esperado
            return {
                "mime_type": "image/png",
                "data": base64.b64encode(image_io.read()).decode('utf-8')
            }
        except Exception as e:
            logger.error(f"Erro ao capturar tela: {e}")
            # traceback.print_exc() # Descomente para depuração detalhada do mss
            return None
   
   
    async def get_screen(self):
        # (Função get_screen inalterada, exceto por traceback em erro crítico - omitida para brevidade)
        logger.info("Iniciando captura de tela...")
        try:
            while not self.stop_event.is_set():
                # Executa a captura síncrona em outra thread
                frame_data = await asyncio.to_thread(self._get_screen)

                if frame_data is None:
                    logger.info("Falha ao capturar frame da tela.")
                    await asyncio.sleep(1.0) # Espera antes de tentar novamente
                    continue

                # Envia frame para a fila de saída
                if self.out_queue:
                    try:
                         if self.out_queue.full():
                             # Descarta o mais antigo
                             discarded = await self.out_queue.get()
                             self.out_queue.task_done()
                             # logger.info("Aviso: Fila de saída cheia, descartando frame de tela antigo.") # Log opcional
                         self.out_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                         pass # logger.info("Aviso: Fila de saída ainda cheia ao tentar enfileirar frame de tela.") # Log opcional
                    except Exception as q_e:
                         logger.error(f"Erro inesperado ao manipular out_queue em get_screen: {q_e}")

                # Espera 1 segundo entre capturas de tela
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("get_screen cancelado.")
        except Exception as e:
            logger.error(f"Erro crítico em get_screen: {e}")
            traceback.print_exc() # Imprime traceback para erros críticos
            self.stop_event.set()
        finally:
            logger.info("get_screen finalizado.")


    async def send_realtime(self):
        # (Função send_realtime inalterada, exceto por traceback em erro crítico - omitida para brevidade)
        logger.info("Send_realtime pronto para enviar dados...")
        try:
            while not self.stop_event.is_set():
                # Pausa se o Gemini estiver processando uma função (thinking_event)
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

                # Verifica se a fila de saída existe (pode ser None durante reconexão)
                if not self.out_queue:
                    await asyncio.sleep(0.1)
                    continue

                # Tenta obter uma mensagem da fila com timeout
                try:
                    msg = await asyncio.wait_for(self.out_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Timeout é normal se não houver dados por 1s
                    continue
                except asyncio.QueueEmpty:
                    # Fila vazia também é normal
                    continue
                except Exception as q_get_e:
                    # Erro inesperado ao obter da fila
                    logger.error(f"Erro ao obter da out_queue em send_realtime: {q_get_e}")
                    await asyncio.sleep(0.1)
                    continue


                # Verifica se a sessão Gemini está ativa
                if not self.session:
                    # logger.info("Sessão Gemini não está ativa (send_realtime). Descartando mensagem.") # Log opcional
                    if self.out_queue: self.out_queue.task_done() # Marca a tarefa como concluída mesmo descartando
                    if not self.stop_event.is_set():
                        await asyncio.sleep(0.5)
                    continue

                # Tenta enviar a mensagem para o Gemini
                try:
                    if isinstance(msg, dict) and "data" in msg and "mime_type" in msg : # Mensagem multimodal (imagem/áudio)
                        await self.session.send(input=msg, end_of_turn=True)
                    elif isinstance(msg, str): # Mensagem de texto (raro neste fluxo)
                        # AJUSTADO: end_of_turn=True para garantir que o modelo processe como um turno completo
                        # se uma string chegar aqui (ex: alerta sistêmico ou, hipoteticamente, msg de usuário perdida).
                        # Inputs de usuário normais são via send_text (com end_of_turn=True).
                        # Alertas YOLO são enviados de get_frames (com end_of_turn=True).
                        logger.info(f"Enviando texto via send_realtime (tratando como turno completo): {msg}")
                        await self.session.send(input=msg, end_of_turn=True)
                    else:
                        logger.info(f"Mensagem desconhecida em send_realtime: {type(msg)}")

                    # Marca a tarefa como concluída na fila após envio bem-sucedido
                    if self.out_queue: self.out_queue.task_done()

                except Exception as e_send:
                    logger.error(f"Erro ao enviar para Gemini em send_realtime: {e_send}")
                    # Garante task_done mesmo em erro para não bloquear a fila
                    if self.out_queue: self.out_queue.task_done()
                    # Verifica se o erro indica sessão fechada/perdida
                    error_str_upper = str(e_send).upper()
                    if "LIVESESSION CLOSED" in error_str_upper or \
                       "LIVESESSION NOT CONNECTED" in error_str_upper or \
                       "DEADLINE EXCEEDED" in error_str_upper or \
                       "RST_STREAM" in error_str_upper or \
                       "UNAVAILABLE" in error_str_upper:
                        logger.info("Erro de envio indica sessão Gemini fechada/perdida. Sinalizando parada.")
                        self.stop_event.set()
                        break # Sai do loop while
                    else:
                        # Outros erros podem ser temporários, imprime traceback para análise
                        traceback.print_exc()
                        await asyncio.sleep(0.5) # Pausa antes de tentar processar próximo item

        except asyncio.CancelledError:
            logger.info("send_realtime cancelado.")
        except Exception as e:
            logger.error(f"Erro fatal em send_realtime: {e}")
            traceback.print_exc() # Imprime traceback para erros fatais
            self.stop_event.set()
        finally:
            logger.info("send_realtime finalizado.")


    async def listen_audio(self):
        # (Função listen_audio inalterada, exceto por traceback em erro crítico - omitida para brevidade)
        if not pya:
            logger.info("PyAudio não inicializado. Tarefa listen_audio não pode iniciar.")
            return

        audio_stream = None
        try:
            logger.info("Configurando stream de áudio de entrada...")
            mic_info = pya.get_default_input_device_info()
            logger.info(f"Usando microfone: {mic_info['name']}")
            audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
                input=True, input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE
            )
            logger.info("Escutando áudio do microfone...")

            while not self.stop_event.is_set():
                # Pausa se o Gemini estiver processando
                if self.thinking_event.is_set():
                    await asyncio.sleep(0.05)
                    continue

                # Verifica se o stream ainda está ativo
                if not audio_stream or not audio_stream.is_active():
                     logger.info("Stream de áudio de entrada não está ativo. Encerrando listen_audio.")
                     self.stop_event.set()
                     break

                # Lê dados do microfone em outra thread
                try:
                    data = await asyncio.to_thread(
                        audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    # Envia dados para a fila de saída
                    if self.out_queue:
                         try:
                             if self.out_queue.full():
                                 # Descarta o mais antigo se a fila estiver cheia (menos provável para áudio)
                                 discarded = await self.out_queue.get()
                                 self.out_queue.task_done()
                                 # pass # Ou simplesmente não envia se estiver cheio
                                 # logger.info("Aviso: Fila de saída cheia, áudio pode ser atrasado/descartado.")
                             self.out_queue.put_nowait({"data": data, "mime_type": "audio/pcm"})
                         except asyncio.QueueFull:
                             pass # logger.info("Aviso: Fila de saída cheia ao tentar enfileirar áudio.")
                         except Exception as q_e:
                              logger.error(f"Erro inesperado ao manipular out_queue em listen_audio: {q_e}")

                except OSError as e_os:
                    # Erros comuns de stream fechado ou overflow
                    if e_os.errno == -9988 or "Stream closed" in str(e_os) or "Input overflowed" in str(e_os):
                        logger.info(f"Stream de áudio fechado ou com overflow (OSError: {e_os}). Encerrando listen_audio.")
                        self.stop_event.set()
                        break
                    else:
                        # Outros erros de OS podem ser mais sérios
                        logger.error(f"Erro de OS ao ler do stream de áudio: {e_os}")
                        traceback.print_exc()
                        self.stop_event.set()
                        break
                except Exception as e_read:
                    # Erro genérico durante a leitura
                    logger.error(f"Erro durante a leitura do áudio em listen_audio: {e_read}")
                    traceback.print_exc()
                    self.stop_event.set() # Para a tarefa em caso de erro de leitura
                    break
        except asyncio.CancelledError:
            logger.info("listen_audio cancelado.")
        except Exception as e:
            # Erro crítico na configuração ou loop principal
            logger.error(f"Erro crítico em listen_audio: {e}")
            traceback.print_exc()
            self.stop_event.set()
        finally:
            logger.info("Finalizando listen_audio...")
            # Garante que o stream seja fechado
            if audio_stream:
                try:
                    if audio_stream.is_active():
                        audio_stream.stop_stream()
                    audio_stream.close()
                    logger.info("Stream de áudio de entrada fechado.")
                except Exception as e_close_stream:
                    logger.error(f"Erro ao fechar stream de áudio de entrada: {e_close_stream}")
            logger.info("listen_audio concluído.")


    def _handle_save_known_face(self, person_name: str) -> str:
        """Processa a chamada de função para salvar um rosto."""
        # --- LOGGING INÍCIO ---
        logger.info("[LOG] Executando: _handle_save_known_face")
        logger.info(f"[LOG]   - Argumentos: person_name='{person_name}'")
        # --- FIM LOGGING ---

        start_time = time.time()
        result_message = "" # Inicializa a mensagem de resultado

        logger.info(f"[DeepFace] Iniciando salvamento para: {person_name}")
        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            logger.info("[DeepFace] Erro: Nenhum frame disponível para salvar.")
            result_message = "Não foi possível capturar a imagem para salvar o rosto."
            # --- LOGGING FIM ---
            logger.info("[LOG] Finalizado: _handle_save_known_face (Erro: Sem frame)")
            logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
            logger.info(f"[LOG]   - Resultado: '{result_message}'")
            # --- FIM LOGGING ---
            return result_message

        # Sanitiza nome para diretório e arquivo
        safe_person_name_dir = "".join(c if c.isalnum() or c in [' '] else '_' for c in person_name).strip().replace(" ", "_")
        if not safe_person_name_dir: safe_person_name_dir = "desconhecido"
        person_dir = os.path.join(DB_PATH, safe_person_name_dir)

        try:
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
                logger.info(f"[DeepFace] Diretório criado: {person_dir}")

            # Extrai rosto(s) do frame
            detected_faces = DeepFace.extract_faces(
                img_path=frame_to_process,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                enforce_detection=True, # Garante que um rosto foi detectado
                align=True,
               # silent=True
            )

            # Verifica se algum rosto foi detectado
            if not detected_faces or not isinstance(detected_faces, list) or 'facial_area' not in detected_faces[0]:
                logger.info(f"[DeepFace] Nenhum rosto detectado para {person_name}.")
                result_message = f"Não consegui detectar um rosto claro para {person_name}."
                # --- LOGGING FIM ---
                logger.info("[LOG] Finalizado: _handle_save_known_face (Erro: Rosto não detectado)")
                logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
                logger.info(f"[LOG]   - Resultado: '{result_message}'")
                # --- FIM LOGGING ---
                return result_message

            # Pega o primeiro rosto detectado (geralmente o maior/mais central)
            face_data = detected_faces[0]['facial_area']
            x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']

            # Recorta a imagem do rosto com uma margem
            margin = 10
            y1, y2 = max(0, y - margin), min(frame_to_process.shape[0], y + h + margin)
            x1, x2 = max(0, x - margin), min(frame_to_process.shape[1], x + w + margin)
            face_image = frame_to_process[y1:y2, x1:x2]

            if face_image.size == 0:
                 logger.info(f"[DeepFace] Erro ao recortar rosto para {person_name} (imagem vazia).")
                 result_message = f"Erro ao processar o rosto de {person_name}."
                 # --- LOGGING FIM ---
                 logger.info("[LOG] Finalizado: _handle_save_known_face (Erro: Recorte vazio)")
                 logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
                 logger.info(f"[LOG]   - Resultado: '{result_message}'")
                 # --- FIM LOGGING ---
                 return result_message

            # Cria nome de arquivo único
            timestamp = int(time.time())
            safe_file_name_base = "".join(c if c.isalnum() else '_' for c in person_name).strip()
            if not safe_file_name_base: safe_file_name_base = "rosto"
            file_name = f"{safe_file_name_base.lower()}_{timestamp}.jpg"
            file_path = os.path.join(person_dir, file_name)

            # Salva a imagem do rosto
            save_success = cv2.imwrite(file_path, face_image)
            if not save_success:
                logger.info(f"[DeepFace] Erro ao salvar imagem em {file_path}")
                result_message = f"Erro ao salvar a imagem do rosto de {person_name}."
                # --- LOGGING FIM ---
                logger.info("[LOG] Finalizado: _handle_save_known_face (Erro: Falha no imwrite)")
                logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
                logger.info(f"[LOG]   - Resultado: '{result_message}'")
                # --- FIM LOGGING ---
                return result_message


            # Remove cache de representações para forçar recálculo
            model_name_safe = DEEPFACE_MODEL_NAME.lower().replace('-', '_')
            representations_pkl_path = os.path.join(DB_PATH, f"representations_{model_name_safe}.pkl")
            if os.path.exists(representations_pkl_path):
                try:
                    os.remove(representations_pkl_path)
                    logger.info(f"[DeepFace] Cache de representações '{representations_pkl_path}' removido para atualização.")
                except Exception as e_pkl:
                    logger.info(f"[DeepFace] Aviso: Falha ao remover cache de representações: {e_pkl}")

            logger.info(f"[DeepFace] Rosto de {person_name} salvo em {file_path}")
            result_message = f"Rosto de {person_name} salvo com sucesso."

        except ValueError as ve: # Captura erro específico do DeepFace se enforce_detection=True falhar
             logger.info(f"[DeepFace] Nenhum rosto detectado (ValueError) para {person_name}: {ve}")
             result_message = f"Não consegui detectar um rosto claro para salvar para {person_name}."
        except Exception as e:
            logger.info(f"[DeepFace] Erro inesperado ao salvar rosto para {person_name}: {e}")
            traceback.print_exc()
            result_message = f"Ocorreu um erro ao tentar salvar o rosto de {person_name}."

        # --- LOGGING FIM ---
        logger.info("[LOG] Finalizado: _handle_save_known_face")
        logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
        logger.info(f"[LOG]   - Resultado: '{result_message}'")
        # --- FIM LOGGING ---
        return result_message


    def _handle_identify_person_in_front(self) -> str:
        """Processa a chamada de função para identificar uma pessoa."""
        # --- LOGGING INÍCIO ---
        logger.info("[LOG] Executando: _handle_identify_person_in_front")
        # --- FIM LOGGING ---

        start_time = time.time()
        result_message = ""

        # Verifica se pandas foi importado com sucesso
        if pd is None:
            logger.info("[DeepFace] Erro: Biblioteca 'pandas' não está disponível. Identificação desabilitada.")
            result_message = "Erro interno: dependência 'pandas' faltando para identificação."
            # --- LOGGING FIM ---
            logger.info("[LOG] Finalizado: _handle_identify_person_in_front (Erro: Sem pandas)")
            logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
            logger.info(f"[LOG]   - Resultado: '{result_message}'")
            # --- FIM LOGGING ---
            return result_message


        logger.info("[DeepFace] Iniciando identificação de pessoa...")
        frame_to_process = None
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()

        if frame_to_process is None:
            logger.info("[DeepFace] Erro: Nenhum frame disponível para identificar.")
            result_message = "Não foi possível capturar a imagem para identificar."
            # --- LOGGING FIM ---
            logger.info("[LOG] Finalizado: _handle_identify_person_in_front (Erro: Sem frame)")
            logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
            logger.info(f"[LOG]   - Resultado: '{result_message}'")
            # --- FIM LOGGING ---
            return result_message

        try:
            # Usa DeepFace.find para buscar no banco de dados
            # enforce_detection=True: Exige detecção clara de rosto na imagem de entrada
            # silent=True: Reduz o output do DeepFace
            dfs = DeepFace.find(
                img_path=frame_to_process,
                db_path=DB_PATH,
                model_name=DEEPFACE_MODEL_NAME,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                distance_metric=DEEPFACE_DISTANCE_METRIC,
                enforce_detection=True, # Alterado para True para buscar rostos mais claros
                #silent=True,
                align=True
            )

            # DeepFace.find retorna uma lista de DataFrames. Pegamos o primeiro.
            # Verifica se a lista ou o DataFrame estão vazios
            if not dfs or not isinstance(dfs, list) or not isinstance(dfs[0], pd.DataFrame) or dfs[0].empty:
                logger.info("[DeepFace] Nenhuma correspondência encontrada ou rosto não detectado claramente.")
                result_message = "Não consegui reconhecer ninguém ou não detectei um rosto claro."
                # --- LOGGING FIM ---
                logger.info("[LOG] Finalizado: _handle_identify_person_in_front (Nenhuma correspondência)")
                logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
                logger.info(f"[LOG]   - Resultado: '{result_message}'")
                # --- FIM LOGGING ---
                return result_message

            df = dfs[0]

            # Encontra a coluna de distância correta (pode variar com modelo/métrica)
            distance_col_name = f"{DEEPFACE_MODEL_NAME}_{DEEPFACE_DISTANCE_METRIC}"
            if distance_col_name not in df.columns:
                # Fallback para nomes comuns ou que contenham a métrica
                if 'distance' in df.columns:
                    distance_col_name = 'distance'
                else:
                    found_col = None
                    for col in df.columns:
                        if DEEPFACE_DISTANCE_METRIC in col.lower():
                            found_col = col
                            break
                    if found_col:
                        distance_col_name = found_col
                    else:
                        logger.info(f"[DeepFace] Erro: Coluna de distância '{distance_col_name}' ou similar não encontrada. Colunas: {df.columns.tolist()}")
                        result_message = "Erro ao processar resultado da identificação (coluna de distância)."
                        # --- LOGGING FIM ---
                        logger.info("[LOG] Finalizado: _handle_identify_person_in_front (Erro: Coluna distância)")
                        logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
                        logger.info(f"[LOG]   - Resultado: '{result_message}'")
                        # --- FIM LOGGING ---
                        return result_message

            # Ordena por distância (menor é melhor) e pega o melhor resultado
            df = df.sort_values(by=distance_col_name, ascending=True)
            best_match = df.iloc[0]

            # Extrai informações do melhor match
            best_match_identity_path = best_match['identity']
            # O nome da pessoa é o nome do diretório pai do arquivo de imagem
            person_name = os.path.basename(os.path.dirname(best_match_identity_path))
            distance = best_match[distance_col_name]

            logger.info(f"[DeepFace] Pessoa potencialmente identificada: {person_name} (Distância: {distance:.4f})")

            # Define limiares de distância (ajustar experimentalmente!)
            thresholds = {
                'VGG-Face': {'cosine': 0.40, 'euclidean': 0.60, 'euclidean_l2': 0.86},
                'Facenet': {'cosine': 0.40, 'euclidean': 0.90, 'euclidean_l2': 1.10},
                'Facenet512': {'cosine': 0.30, 'euclidean': 0.70, 'euclidean_l2': 0.95},
                'ArcFace': {'cosine': 0.68, 'euclidean': 1.13, 'euclidean_l2': 1.13},
                'Dlib': {'cosine': 0.07, 'euclidean': 0.6, 'euclidean_l2': 0.6},
            }
            # Usa um threshold padrão se o modelo/métrica não estiver mapeado
            threshold = thresholds.get(DEEPFACE_MODEL_NAME, {}).get(DEEPFACE_DISTANCE_METRIC, 0.5)

            # Compara a distância com o limiar
            if distance <= threshold:
                result_message = f"A pessoa na sua frente parece ser {person_name}."
            else:
                logger.info(f"[DeepFace] Distância {distance:.4f} > limiar ({threshold}). Não reconhecido com confiança.")
                # Poderia retornar o nome com baixa confiança, ou uma mensagem genérica
                result_message = "Não tenho certeza de quem é, mas detectei um rosto."
                # result_message = f"Detectei um rosto, mas não tenho certeza. Pode ser {person_name}?"

        except ValueError as ve: # Captura erro se enforce_detection=True e nenhum rosto for encontrado
            logger.info(f"[DeepFace] Erro (ValueError) ao identificar: {ve}")
            result_message = "Não detectei um rosto claro para identificar."
        # except ImportError: # Já tratado no início da função
        #      pass
        except Exception as e:
            logger.info(f"[DeepFace] Erro inesperado ao identificar: {e}")
            traceback.print_exc()
            result_message = "Ocorreu um erro ao tentar identificar a pessoa."

        # --- LOGGING FIM ---
        logger.info("[LOG] Finalizado: _handle_identify_person_in_front")
        logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
        logger.info(f"[LOG]   - Resultado: '{result_message}'")
        # --- FIM LOGGING ---
        return result_message


    def _run_midas_inference(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        # (Função _run_midas_inference inalterada - omitida para brevidade)
        if not self.midas_model or not self.midas_transform:
            logger.info("[MiDaS] Modelo ou transformador não carregado.")
            return None
        try:
            # Converte BGR para RGB
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Aplica transformações específicas do MiDaS e envia para o dispositivo (CPU/GPU)
            input_batch = self.midas_transform(img_rgb).to(self.midas_device)

            with torch.no_grad(): # Desabilita cálculo de gradientes para inferência
                prediction = self.midas_model(input_batch)
                # Redimensiona a predição para o tamanho original da imagem
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic", # ou "bilinear"
                    align_corners=False,
                ).squeeze()

            # Move o resultado de volta para a CPU e converte para NumPy array
            depth_map = prediction.cpu().numpy()
            return depth_map
        except Exception as e:
            logger.info(f"[MiDaS] Erro durante a inferência: {e}")
            # traceback.print_exc() # Descomente para depuração detalhada do MiDaS
            return None


    def _find_best_yolo_match(self, object_type: str, yolo_results: List[Any]) -> Optional[Tuple[Dict[str, int], float, str]]:
        # (Função _find_best_yolo_match inalterada - omitida para brevidade)
        best_match = None
        highest_conf = -1.0
        # Obtém a lista de nomes de classe YOLO correspondentes ao tipo de objeto pedido
        target_yolo_classes = YOLO_CLASS_MAP.get(object_type.lower(), [object_type.lower()])
        # logger.info(f"[YOLO Match] Procurando por classes: {target_yolo_classes}") # Log opcional

        # Verifica se há resultados YOLO e se o modelo está carregado
        if not yolo_results or not self.yolo_model:
             # logger.info("[YOLO Match] Sem resultados YOLO ou modelo não carregado.") # Log opcional
             return None

        # Itera sobre os resultados (pode haver mais de um se o modelo processar em batch, embora aqui seja 1)
        for result in yolo_results:
            # Verifica se o objeto de resultado tem o atributo 'boxes' e se não está vazio
            if hasattr(result, 'boxes') and result.boxes:
                # Itera sobre cada caixa delimitadora detectada
                for box in result.boxes:
                    # Verifica se a caixa tem os atributos necessários
                    if not (hasattr(box, 'cls') and hasattr(box, 'conf') and hasattr(box, 'xyxy')):
                        # logger.info("[YOLO Match] Caixa malformada encontrada.") # Log opcional
                        continue # Pula caixas malformadas

                    # Obtém ID da classe, confiança e coordenadas
                    cls_id_tensor = box.cls
                    if cls_id_tensor.nelement() == 0: continue # Tensor vazio
                    cls_id = int(cls_id_tensor[0])

                    conf_tensor = box.conf
                    if conf_tensor.nelement() == 0: continue
                    conf = float(conf_tensor[0])

                    # Obtém o nome da classe a partir do ID
                    if cls_id < len(self.yolo_model.names):
                        class_name = self.yolo_model.names[cls_id]
                    else:
                        # logger.info(f"[YOLO Match] ID de classe inválido: {cls_id}") # Log opcional
                        continue # ID inválido

                    # Verifica se a classe detectada é uma das classes alvo
                    if class_name in target_yolo_classes:
                        # Se a confiança for maior que a melhor encontrada até agora
                        if conf > highest_conf:
                            highest_conf = conf
                            coords_tensor = box.xyxy[0]
                            if coords_tensor.nelement() < 4: continue # Coordenadas inválidas
                            coords = list(map(int, coords_tensor))
                            bbox_dict = {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3]}
                            best_match = (bbox_dict, conf, class_name)
                            logger.info(f"[YOLO Match] Novo melhor match: {class_name} ({conf:.2f})") # Log opcional

        # Retorna a melhor correspondência encontrada (ou None)
        return best_match


    def _estimate_direction(self, bbox: Dict[str, int], frame_width: int) -> str:
        # (Função _estimate_direction inalterada - omitida para brevidade)
        # Calcula o centro horizontal da caixa
        box_center_x = (bbox['x1'] + bbox['x2']) / 2
        # Define a largura da zona central (um terço da largura do frame)
        center_zone_width = frame_width / 3

        # Verifica em qual terço (esquerda, centro, direita) o centro da caixa está
        if box_center_x < center_zone_width:
            return "à sua esquerda"
        elif box_center_x > (frame_width - center_zone_width):
            return "à sua direita"
        else:
            return "à sua frente"


    def _check_if_on_surface(self, target_bbox: Dict[str, int], yolo_results: List[Any]) -> bool:
        # (Função _check_if_on_surface inalterada - omitida para brevidade)
        # Define as classes que representam superfícies de apoio
        surface_classes_keys = ["mesa", "mesa de jantar", "bancada", "prateleira"] # Expandido
        surface_yolo_names = []
        for key in surface_classes_keys:
            surface_yolo_names.extend(YOLO_CLASS_MAP.get(key, [])) # Pega nomes YOLO do mapa
        surface_yolo_names = list(set(surface_yolo_names)) # Remove duplicatas

        if not surface_yolo_names: return False # Se não há classes de superfície mapeadas

        # Coordenadas do objeto alvo
        target_bottom_y = target_bbox['y2']
        target_center_x = (target_bbox['x1'] + target_bbox['x2']) / 2

        if not yolo_results or not self.yolo_model:
            return False

        # Itera sobre as detecções YOLO procurando por superfícies
        for result in yolo_results:
             if hasattr(result, 'boxes') and result.boxes:
                for box in result.boxes:
                    if not (hasattr(box, 'cls') and hasattr(box, 'xyxy')):
                        continue

                    cls_id_tensor = box.cls
                    if cls_id_tensor.nelement() == 0: continue
                    cls_id = int(cls_id_tensor[0])

                    if cls_id < len(self.yolo_model.names):
                        class_name = self.yolo_model.names[cls_id]
                    else:
                        continue

                    # Se a classe detectada for uma superfície
                    if class_name in surface_yolo_names:
                        coords_tensor = box.xyxy[0]
                        if coords_tensor.nelement() < 4: continue
                        s_x1, s_y1, s_x2, s_y2 = map(int, coords_tensor)

                        # --- Heurística para verificar se o objeto está SOBRE a superfície ---
                        # 1. Alinhamento Horizontal: O centro X do objeto está dentro da largura da superfície.
                        horizontally_aligned = s_x1 < target_center_x < s_x2

                        # 2. Alinhamento Vertical: A base do objeto (target_bottom_y) está
                        #    próxima ou ligeiramente acima do topo da superfície (s_y1).
                        #    Permite uma pequena sobreposição ou espaço.
                        y_tolerance_pixels = 30 # Tolerância em pixels (ajustar)
                        # Objeto está "descansando" perto do topo da superfície
                        vertically_aligned = (s_y1 - y_tolerance_pixels) < target_bottom_y < (s_y1 + y_tolerance_pixels * 1.5)

                        # 3. (Opcional) Tamanho Relativo: Evitar que um objeto muito grande "sobre" um pequeno.
                        # target_height = target_bbox['y2'] - target_bbox['y1']
                        # surface_height = s_y2 - s_y1
                        # reasonable_size = target_height < surface_height * 2 # Exemplo

                        # 4. (Opcional) Proximidade: A superfície deve estar relativamente próxima do objeto.
                        # (Poderia usar MiDaS aqui, mas complica)

                        if horizontally_aligned and vertically_aligned: # and reasonable_size:
                            # logger.info(f"[Surface Check] Objeto em ({target_center_x},{target_bottom_y}) considerado sobre '{class_name}' em ({s_x1}-{s_x2}, {s_y1}-{s_y2})") # Log opcional
                            return True # Encontrou uma superfície sob o objeto
        return False # Nenhuma superfície encontrada sob o objeto


    def _handle_find_object_and_estimate_distance(self, object_description: str, object_type: str) -> str:
        """Processa a chamada de função para localizar um objeto e estimar distância."""
        # --- LOGGING INÍCIO ---
        logger.info("[LOG] Executando: _handle_find_object_and_estimate_distance")
        logger.info(f"[LOG]   - Argumentos: object_description='{object_description}', object_type='{object_type}'")
        # --- FIM LOGGING ---

        start_time = time.time()
        result_message = ""

        logger.info(f"[Localizar Objeto] Buscando por '{object_description}' (tipo: '{object_type}')...")
        frame_to_process = None
        yolo_results_for_frame = None
        frame_height, frame_width = 0, 0

        # Obtém o último frame e resultados YOLO de forma segura
        with self.frame_lock:
            if self.latest_bgr_frame is not None:
                frame_to_process = self.latest_bgr_frame.copy()
                yolo_results_for_frame = self.latest_yolo_results # Pega os resultados correspondentes
                if frame_to_process is not None:
                    frame_height, frame_width, _ = frame_to_process.shape
            # else: # Frame é None, tratado abaixo

        # Verifica se temos um frame válido
        if frame_to_process is None or frame_width == 0 or frame_height == 0:
             logger.info("[Localizar Objeto] Erro: Nenhum frame válido disponível.")
             result_message = f"{self.trckuser}, não estou enxergando nada no momento para localizar o {object_type}."
             # --- LOGGING FIM ---
             logger.info("[LOG] Finalizado: _handle_find_object_and_estimate_distance (Erro: Sem frame)")
             logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
             logger.info(f"[LOG]   - Resultado: '{result_message}'")
             # --- FIM LOGGING ---
             return result_message

        # Verifica se temos resultados YOLO para este frame
        if not yolo_results_for_frame:
            logger.info("[Localizar Objeto] Erro: Nenhum resultado YOLO disponível para o frame atual.")
            # Isso pode acontecer se o YOLO falhou ou ainda não processou o frame
            result_message = f"{self.trckuser}, não consegui processar a imagem a tempo para encontrar o {object_type}."
            # --- LOGGING FIM ---
            logger.info("[LOG] Finalizado: _handle_find_object_and_estimate_distance (Erro: Sem resultados YOLO)")
            logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
            logger.info(f"[LOG]   - Resultado: '{result_message}'")
            # --- FIM LOGGING ---
            return result_message

        # Encontra a melhor correspondência YOLO para o tipo de objeto
        best_yolo_match = self._find_best_yolo_match(object_type, yolo_results_for_frame)

        # Fallback: Se não encontrou pelo tipo, tenta pela última palavra da descrição
        if not best_yolo_match:
            logger.info(f"[Localizar Objeto] Nenhum objeto do tipo '{object_type}' encontrado. Tentando fallback com descrição...")
            last_word = object_description.split(" ")[-1].lower()
            # Evita tentar o mesmo tipo duas vezes se type já era a última palavra
            if last_word != object_type.lower():
                target_yolo_classes_fallback = YOLO_CLASS_MAP.get(last_word, [last_word])
                logger.info(f"[Localizar Objeto] Fallback: Buscando por classes: {target_yolo_classes_fallback}")
                best_yolo_match = self._find_best_yolo_match(last_word, yolo_results_for_frame)

            # Se ainda não encontrou após o fallback
            if not best_yolo_match:
                 logger.info(f"[Localizar Objeto] Objeto '{object_description}' não encontrado mesmo com fallback.")
                 result_message = f"{self.trckuser}, não consegui encontrar um(a) {object_description} na imagem."
                 # --- LOGGING FIM ---
                 logger.info("[LOG] Finalizado: _handle_find_object_and_estimate_distance (Não encontrado)")
                 logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
                 logger.info(f"[LOG]   - Resultado: '{result_message}'")
                 # --- FIM LOGGING ---
                 return result_message

        # Se encontrou um objeto
        target_bbox, confidence, detected_class = best_yolo_match
        logger.info(f"[Localizar Objeto] Melhor correspondência YOLO: Classe '{detected_class}', Conf: {confidence:.2f}, BBox: {target_bbox}")

        # Verifica se está sobre uma superfície
        is_on_surface = self._check_if_on_surface(target_bbox, yolo_results_for_frame)
        surface_msg = "sobre uma superfície (como uma mesa)" if is_on_surface else ""

        # Estima a direção
        direction = self._estimate_direction(target_bbox, frame_width)

        # Estima a distância usando MiDaS
        distance_steps = -1
        depth_map = None
        if self.midas_model:
            logger.info("[Localizar Objeto] Executando MiDaS...")
            depth_map = self._run_midas_inference(frame_to_process) # Executa inferência MiDaS
        else:
            logger.info("[Localizar Objeto] MiDaS não disponível. Não é possível estimar distância.")

        if depth_map is not None:
            try:
                # Pega o valor de profundidade no centro da caixa do objeto
                center_x = int((target_bbox['x1'] + target_bbox['x2']) / 2)
                center_y = int((target_bbox['y1'] + target_bbox['y2']) / 2)
                # Garante que as coordenadas estão dentro dos limites do mapa de profundidade
                center_y = max(0, min(center_y, depth_map.shape[0] - 1))
                center_x = max(0, min(center_x, depth_map.shape[1] - 1))
                depth_value = depth_map[center_y, center_x]

                # --- Heurística de Conversão MiDaS (MUITO BRUTA - PRECISA CALIBRAR) ---
                # MiDaS_small retorna profundidade inversa (maior valor = mais perto)
                # Valores dependem muito da escala da cena e do modelo.
                # Esta é uma tentativa de mapeamento muito simplificada.
                if depth_value > 1e-6: # Evita divisão por zero ou valores inválidos
                    # Mapeamento heurístico (ajustar com base em testes reais)
                    # Exemplo: depth > 250 -> muito perto; depth < 30 -> longe
                    if depth_value > 300:  # Ajuste estes valores!
                        estimated_meters = np.random.uniform(0.5, 1.5) # Ex: 0.5-1.5m
                    elif depth_value > 150: # Ajuste!
                        estimated_meters = np.random.uniform(1.5, 3.5) # Ex: 1.5-3.5m
                    elif depth_value > 50:  # Ajuste!
                        estimated_meters = np.random.uniform(3.5, 7.0) # Ex: 3.5-7m
                    else: # Longe
                        estimated_meters = np.random.uniform(7.0, 15.0) # Ex: 7-15m

                    # Limita a estimativa a um alcance razoável
                    estimated_meters = max(0.5, min(estimated_meters, 20))
                    # Converte metros para passos
                    distance_steps = max(1, round(estimated_meters / METERS_PER_STEP)) # Garante pelo menos 1 passo
                    logger.info(f"[Localizar Objeto] Profundidade MiDaS no centro ({center_y},{center_x}): {depth_value:.4f}, Metros Estimados (heurístico): {estimated_meters:.2f}, Passos: {distance_steps}")
                else:
                     logger.info("[Localizar Objeto] Valor de profundidade MiDaS inválido ou muito baixo no centro do objeto.")
            except IndexError:
                logger.info(f"[Localizar Objeto] Erro: Coordenadas ({center_y},{center_x}) fora dos limites do mapa de profundidade ({depth_map.shape}).")
            except Exception as e_depth:
                logger.info(f"[Localizar Objeto] Erro ao extrair/processar profundidade do MiDaS: {e_depth}")
                # traceback.print_exc() # Descomente para depuração
                distance_steps = -1 # Reseta se houve erro
        # else: MiDaS não disponível ou falhou, distance_steps continua -1

        # --- Constrói a Mensagem de Resposta ---
        # Usa a descrição original do usuário para a resposta
        object_name_for_response = object_description

        response_parts = [f"{self.trckuser}, o {object_name_for_response} está"]
        if surface_msg:
            response_parts.append(surface_msg) # Adiciona "sobre uma superfície..."

        if distance_steps > 0:
            # Adiciona a distância em passos
            response_parts.append(f"a aproximadamente {distance_steps} passo{'s' if distance_steps > 1 else ''}")

        # Adiciona a direção (sempre)
        response_parts.append(direction)

        # Junta as partes da resposta
        if len(response_parts) > 1: # Se adicionou algo além de "Usuário, o obj está"
            # Junta com vírgulas, exceto antes da direção final
            if len(response_parts) > 2:
                # Ex: "..., sobre a superfície, a X passos, à sua frente."
                result_message = ", ".join(response_parts[:-1]) + " " + response_parts[-1] + "."
            else:
                # Ex: "..., a X passos à sua frente." ou "..., sobre a superfície à sua frente."
                result_message = " ".join(response_parts) + "."
        else:
            # Caso muito raro onde só temos a direção (sem distância e sem superfície)
            result_message = f"{self.trckuser}, o {object_name_for_response} está {direction}."


        # --- LOGGING FIM ---
        logger.info("[LOG] Finalizado: _handle_find_object_and_estimate_distance")
        logger.info(f"[LOG]   - Duração: {time.time() - start_time:.2f}s")
        logger.info(f"[LOG]   - Resultado: '{result_message}'")
        # --- FIM LOGGING ---
        return result_message


    async def receive_audio(self):
        # (Função receive_audio inalterada, exceto por traceback e comentário - omitida para brevidade)
        logger.info("Receive_audio pronto para receber respostas do Gemini...")
        try:
            if not self.session:
                logger.info("Sessão Gemini não estabelecida em receive_audio. Encerrando.")
                self.stop_event.set()
                return

            while not self.stop_event.is_set():
                # Verifica se a sessão ainda existe (pode ser fechada por outra tarefa)
                if not self.session:
                    logger.info("Sessão Gemini desconectada em receive_audio. Aguardando reconexão ou parada.")
                    await asyncio.sleep(1)
                    if not self.session and not self.stop_event.is_set(): # Verifica de novo
                        logger.info("Sessão ainda não disponível. Sinalizando parada para reconexão.")
                        self.stop_event.set() # Sinaliza para o loop run tentar reconectar
                    elif self.session:
                        logger.info("Sessão Gemini reconectada.")
                    break # Sai do loop interno para o run tentar reconectar ou parar


                try:
                    has_received_data_in_turn = False
                    # logger.info("Aguardando próximo turno de resposta do Gemini...") # Log opcional

                    # NOTA: Este loop `async for` não tem um timeout explícito.
                    # Se a API travar ou parar de enviar dados sem fechar o stream,
                    # esta tarefa pode bloquear indefinidamente. A detecção de erros
                    # abaixo (LiveSession closed, etc.) mitiga isso parcialmente.
                    async for response_part in self.session.receive():
                   #     logger.info(f"--- RAW GEMINI RESPONSE PART ---")
                    #    logger.info(f"Response_part.text: {response_part.text if hasattr(response_part, 'text') else 'N/A'}")
                     #   logger.info(f"Response_part.function_call: {response_part.function_call if hasattr(response_part, 'function_call') else 'N/A'}")
                      #  logger.info(f"Response_part.data (presente?): {hasattr(response_part, 'data') and bool(response_part.data)}")
                       # logger.info(f"Awaiting name flag: {self.awaiting_name_for_save_face}")
                        #logger.info(f"Thinking event set: {self.thinking_event.is_set()}")
                        if self.stop_event.is_set():
                            break
                        if has_received_data_in_turn and response_part.text:
                            logger.info("Descartando texto antigo para nova entrada.")
                            continue
                        has_received_data_in_turn = True

                        if self.stop_event.is_set():
                            logger.info("Sinal de parada recebido durante processamento de resposta.")
                            break

                        # --- Processa Áudio ---
                        if response_part.data:
                            if self.audio_in_queue:
                                try:
                                    self.audio_in_queue.put_nowait(response_part.data)
                                except asyncio.QueueFull:
                                    logger.info("Aviso: Fila de áudio de entrada cheia. Áudio descartado.")
                            continue # Processou áudio, vai para próxima parte

                        # --- Processa Nome Pendente (Fluxo save_known_face) ---
                        if self.awaiting_name_for_save_face:
                            user_provided_name = None
                            if response_part.text: # Gemini transcreveu fala ou usuário digitou
                                user_provided_name = response_part.text.strip()
                                logger.info(f"[Trackie] Recebido texto enquanto aguardava nome: '{user_provided_name}'")

                            if user_provided_name:
                                logger.info(f"[Trackie] Processando nome '{user_provided_name}' para salvar rosto...")
                                self.awaiting_name_for_save_face = False # Reseta a flag

                                original_function_name_pending = "save_known_face" # Nome da função original

                                logger.info("Pensando...") # Feedback visual
                                self.thinking_event.set() # Pausa envio de dados

                                # Feedback de voz ANTES de executar a função
                                voice_feedback_msg = f"{self.trckuser}, salvando rosto de {user_provided_name}, um momento..."
                                if self.session:
                                    try:
                                        # Envia feedback e termina o turno da IA para que ela fale
                                        await self.session.send(input=voice_feedback_msg, end_of_turn=True)
                                        logger.info(f"  [Feedback Enviado]: {voice_feedback_msg}")
                                    except Exception as e_feedback:
                                        logger.error(f"Erro ao enviar feedback de voz (awaiting name): {e_feedback}")

                                # Executa a função síncrona em outra thread
                                result_message = await asyncio.to_thread(self._handle_save_known_face, user_provided_name)

                                # Envia o resultado da função de volta para o Gemini
                                logger.info(f"  [Trackie] Resultado da Função '{original_function_name_pending}': {result_message}")
                                if self.session:
                                    try:
                                        await self.session.send(
                                            input=types.Content(
                                                role="tool", # Importante: role="tool" para respostas de função
                                                parts=[types.Part.from_function_response(
                                                    name=original_function_name_pending,
                                                    response={"result": Value(string_value=result_message)}
                                                )]
                                            )
                                            # Não usar end_of_turn=True aqui, deixa Gemini decidir quando responder
                                        )
                                        logger.info("  [Trackie] Resultado da função (awaiting name) enviado.")
                                    except Exception as e_send_fc_resp:
                                        logger.error(f"Erro ao enviar FunctionResponse (awaiting name): {e_send_fc_resp}")
                                else:
                                    logger.info("  [Trackie] Sessão inativa. Não foi possível enviar resultado da função (awaiting name).")

                                # Libera o envio de dados
                                if self.thinking_event.is_set():
                                    self.thinking_event.clear()
                                logger.info("Pensamento concluído (awaiting name).")
                                continue # Processamos este input, vamos para o próximo response_part

                        # --- Processa Texto da IA ---
                        if response_part.text:
                            # Imprime texto recebido do Gemini (pode ser parcial)
                            # Usando print() diretamente para garantir que o 'end=""' funcione como esperado no console.
                            print(f"\n[Gemini Texto]: {response_part.text}", end="")
                            # logger.info(f"\n[Gemini Texto]: {response_part.text}", extra={'end': ''}) # Alternativa se o logger for configurado para lidar com 'end'


                        # --- Processa Chamada de Função ---
                        if getattr(response_part, "function_call", None):
                            fc = response_part.function_call
                            function_name = fc.name
                            args = {key: val for key, val in fc.args.items()}
                            logger.info(f"\n[Gemini Function Call] Recebido: {function_name}, Args: {args}")

                            result_message = None # Inicializa resultado

                            # --- Caso Especial: save_known_face sem nome ---
                            if function_name == "save_known_face" and not args.get("person_name"):
                                self.awaiting_name_for_save_face = True # Ativa a flag
                                if self.thinking_event.is_set(): # Garante que não está pensando enquanto pergunta
                                    self.thinking_event.clear()
                                logger.info("[Trackie] Nome não fornecido para save_known_face. Solicitando ao usuário.")
                                # Pede o nome ao usuário via Gemini (voz)
                                if self.session:
                                    try:
                                        # Envia a pergunta e termina o turno da IA
                                        await self.session.send(input=f"{self.trckuser}, por favor forneça o nome da pessoa para salvar o rosto.", end_of_turn=True)
                                    except Exception as e_ask_name:
                                        logger.error(f"Erro ao pedir nome para save_face: {e_ask_name}")
                                # Não executa a função local nem envia FC response agora

                            # --- Caso Normal: Outras funções ou save_known_face com nome ---
                            else:
                                logger.info("Pensando...") # Feedback visual
                                self.thinking_event.set() # Pausa envio

                                # Monta mensagem de feedback de voz
                                voice_feedback_msg = f"{self.trckuser}, processando {function_name}, um momento..." # Padrão
                                if function_name == "save_known_face":
                                    person_name_fb = args.get('person_name', 'pessoa')
                                    voice_feedback_msg = f"{self.trckuser}, salvando rosto de {person_name_fb}, um momento..."
                                    logger.info(f"Salvando rosto de {person_name_fb} ")
                                elif function_name == "identify_person_in_front":
                                    voice_feedback_msg = "Usuário, identificando pessoa, um momento..."
                                    logger.info(f"Identificando rosto... {person_name_fb} ")
                                elif function_name == "find_object_and_estimate_distance":
                                    obj_desc_fb = args.get('object_description', 'objeto')
                                    voice_feedback_msg = f"{self.trckuser}, localizando {obj_desc_fb}, um momento..."
                                    logger.info(f"Salvando rosto de {person_name_fb} ")

                                # Envia feedback de voz ANTES de executar a função
                                if self.session:
                                    try:
                                        # Envia feedback e termina o turno da IA
                                        await self.session.send(input=voice_feedback_msg, end_of_turn=True)
                                        logger.info(f"  [Feedback Enviado]: {voice_feedback_msg}")
                                    except Exception as e_feedback:
                                        logger.error(f"Erro ao enviar feedback pré-função: {e_feedback}")

                                # --- Executa a Função Local Correspondente ---
                                # Verifica se a função requer modo câmera
                                vision_functions = ["save_known_face", "identify_person_in_front", "find_object_and_estimate_distance"]
                                if self.video_mode != "camera" and function_name in vision_functions:
                                    logger.info(f"[Function Call] '{function_name}' requer modo câmera, mas modo atual é '{self.video_mode}'.")
                                    result_message = "Desculpe, esta função só está disponível quando a câmera está ativa."
                                else:
                                    logger.info(f"  [Trackie] Processando Função '{function_name}' em background...")
                                    # Chama o handler apropriado em outra thread
                                    try:
                                        if function_name == "save_known_face":
                                            person_name_arg = args.get("person_name")
                                            if person_name_arg:
                                                result_message = await asyncio.to_thread(self._handle_save_known_face, person_name_arg)
                                            else:
                                                result_message = "Erro interno: nome não disponível para salvar rosto neste ponto."
                                                logger.error("ERRO LÓGICO: Tentativa de chamar _handle_save_known_face sem nome.")
                                        elif function_name == "identify_person_in_front":
                                            # Verifica se pandas está disponível
                                            if pd is None:
                                                 result_message = "Erro interno: dependência 'pandas' faltando para identificação."
                                            else:
                                                result_message = await asyncio.to_thread(self._handle_identify_person_in_front)

                                        elif function_name == "find_object_and_estimate_distance":
                                            desc = args.get("object_description")
                                            obj_type = args.get("object_type")
                                            if desc and obj_type:
                                                # Verifica se MiDaS está funcional antes de chamar
                                                if not self.midas_model:
                                                    result_message = "Usuário, desculpe, o módulo de estimativa de distância não está funcionando no momento."
                                                else:
                                                    result_message = await asyncio.to_thread(
                                                        self._handle_find_object_and_estimate_distance, desc, obj_type
                                                    )
                                            else:
                                                result_message = "Descrição ou tipo do objeto não fornecido corretamente para localização."
                                                logger.error(f"ERRO: Argumentos faltando para find_object_and_estimate_distance: desc='{desc}', type='{obj_type}'")
                                        else:
                                            result_message = f"Função '{function_name}' desconhecida ou não implementada."
                                            logger.warning(f"AVISO: Recebida chamada para função não mapeada: {function_name}")
                                    except Exception as e_handler:
                                         logger.error(f"Erro ao executar handler para '{function_name}': {e_handler}")
                                         traceback.print_exc()
                                         result_message = f"Ocorreu um erro interno ao processar a função {function_name}."


                            # --- Envia Resultado da Função de Volta (se houver) ---
                            if result_message is not None: # Só envia se um resultado foi gerado
                                logger.info(f"  [Trackie] Resultado da Função '{function_name}': {result_message}")
                                if self.session:
                                    try:
                                        await self.session.send(
                                            input=types.Content(
                                                role="tool", # Importante
                                                parts=[types.Part.from_function_response(
                                                    name=function_name,
                                                    response={"result": Value(string_value=result_message)}
                                                )]
                                            )
                                        )
                                        logger.info("  [Trackie] Resultado da função enviado.")
                                    except Exception as e_send_fc_resp_main:
                                        logger.error(f"Erro ao enviar FunctionResponse (main): {e_send_fc_resp_main}")
                                else:
                                    logger.info("  [Trackie] Sessão inativa. Não foi possível enviar resultado da função.")

                                # Libera o envio de dados após processar a função
                                if self.thinking_event.is_set():
                                     self.thinking_event.clear()
                                logger.info("Pensamento concluído (function call).")
                            # Se result_message é None (caso de pedir nome), thinking_event já foi limpo antes

                    # --- Fim do processamento de um turno da IA ---
                    if not self.stop_event.is_set():
                        if has_received_data_in_turn:
                            # logger.info("\nFim do turno de resposta do Gemini.") # Log opcional
                            # Adiciona uma nova linha se o último log foi um texto parcial do Gemini
                            if response_part and response_part.text and not response_part.text.endswith('\n'):
                                print() # Garante que o próximo log comece em uma nova linha
                            pass # Continua esperando o próximo turno
                        else:
                            # Se o loop `async for` terminar sem dados, pode indicar fim normal ou problema
                            # logger.info("Stream do turno atual terminou sem dados.") # Log opcional
                            await asyncio.sleep(0.05) # Pequena pausa
                    if self.stop_event.is_set():
                        break # Sai do loop `async for` se stop foi chamado

                # --- Tratamento de Erros no Loop de Recebimento ---
                except Exception as e_inner_loop:
                    logger.error(f"Erro durante o recebimento/processamento de resposta: {e_inner_loop}")
                    error_str_upper = str(e_inner_loop).upper()
                    # Verifica erros que indicam sessão fechada/perdida
                    if "LIVESESSION CLOSED" in error_str_upper or \
                       "LIVESESSION NOT CONNECTED" in error_str_upper or \
                       "DEADLINE EXCEEDED" in error_str_upper or \
                       "RST_STREAM" in error_str_upper or \
                       "UNAVAILABLE" in error_str_upper:
                        logger.info("Erro indica que a sessão Gemini foi fechada/perdida. Sinalizando parada.")
                        self.stop_event.set()
                        break # Sai do loop while principal
                    else:
                        # Outros erros podem ser temporários ou bugs
                        traceback.print_exc()
                        await asyncio.sleep(0.5) # Pausa antes de tentar continuar

            # Se o loop while terminar por causa do stop_event
            if self.stop_event.is_set():
                logger.info("Loop de recebimento de áudio interrompido pelo stop_event.")

        except asyncio.CancelledError:
            logger.info("receive_audio foi cancelado.")
        except Exception as e:
            # Erro crítico fora do loop principal (ex: na configuração inicial)
            logger.error(f"Erro crítico em receive_audio: {e}")
            traceback.print_exc()
            self.stop_event.set() # Garante que tudo pare
        finally:
            logger.info("receive_audio finalizado.")
            # Limpa a flag caso a tarefa seja cancelada enquanto esperava nome
            self.awaiting_name_for_save_face = False
            # Garante que thinking_event seja limpo na saída
            if self.thinking_event.is_set():
                self.thinking_event.clear()


    async def play_audio(self):
        # (Função play_audio inalterada, exceto por traceback em erro crítico - omitida para brevidade)
        if not pya:
            logger.info("PyAudio não inicializado. Tarefa play_audio não pode iniciar.")
            return

        stream = None
        output_rate = RECEIVE_SAMPLE_RATE # Taxa padrão
        try:
            logger.info("Configurando stream de áudio de saída...")
            try:
                # Tenta usar a taxa de amostragem do dispositivo padrão
                out_device_info = pya.get_default_output_device_info()
                logger.info(f"Usando dispositivo de saída: {out_device_info['name']} @ {output_rate} Hz")
            except Exception as e_dev_info:
                logger.info(f"Não foi possível obter info do dispositivo de saída padrão ({e_dev_info}). Usando taxa padrão: {output_rate} Hz")

            # Abre o stream de saída em outra thread
            stream = await asyncio.to_thread(
                pya.open, format=FORMAT, channels=CHANNELS, rate=output_rate, output=True
            )
            logger.info("Player de áudio pronto.")

            while not self.stop_event.is_set():
                # Verifica se a fila de entrada existe
                if not self.audio_in_queue:
                    await asyncio.sleep(0.1)
                    continue

                # Tenta obter áudio da fila com timeout
                try:
                    if self.out_queue and not self.out_queue.empty():
                        logger.info("Nova entrada detectada, interrompendo áudio atual.")
                        stream.stop_stream()
                        await asyncio.sleep(0.1)
                        stream.start_stream()
                    bytestream = await asyncio.wait_for(self.audio_in_queue.get(), timeout=0.5)

                    if bytestream is None: # Sinal de parada da fila
                        logger.info("Recebido sinal de encerramento (None) para play_audio.")
                        break # Sai do loop while

                    # Verifica se o stream está ativo e escreve os dados
                    if stream and stream.is_active():
                        # Escreve no stream em outra thread
                        await asyncio.to_thread(stream.write, bytestream)
                    else:
                        logger.info("Stream de áudio para playback não está ativo. Descartando áudio.")
                        # Não marca task_done se descartou? Ou marca?
                        # Melhor marcar para não bloquear a fila caso o stream morra.
                        if self.audio_in_queue: self.audio_in_queue.task_done()


                    # Marca a tarefa como concluída na fila
                    if self.audio_in_queue: self.audio_in_queue.task_done()

                except asyncio.TimeoutError:
                    # Timeout é normal se não houver áudio por 0.5s
                    continue
                except asyncio.QueueEmpty:
                    # Fila vazia também é normal
                    continue
                except OSError as e_os_play:
                    # Erro comum se o stream for fechado inesperadamente
                    if "Stream closed" in str(e_os_play):
                        logger.info("Stream de playback fechado (OSError). Encerrando play_audio.")
                        break # Sai do loop while
                    else:
                        logger.error(f"Erro de OS ao reproduzir áudio: {e_os_play}")
                        traceback.print_exc()
                        break # Sai em outros erros de OS também
                except Exception as e_inner:
                    # Erro genérico durante a reprodução
                    logger.error(f"Erro ao reproduzir áudio (interno): {e_inner}")
                    if "Stream closed" in str(e_inner): # Verifica se o erro indica stream fechado
                        logger.info("Stream de playback fechado (Exception). Encerrando play_audio.")
                        break # Sai do loop while
                    traceback.print_exc()
                    # Decide se continua ou para em outros erros
                    # break

        except asyncio.CancelledError:
            logger.info("play_audio foi cancelado.")
        except Exception as e:
            # Erro crítico na configuração ou loop principal
            logger.error(f"Erro crítico em play_audio: {e}")
            traceback.print_exc()
            # Não seta stop_event aqui, deixa o run gerenciar
        finally:
            logger.info("Finalizando play_audio...")
            # Garante que o stream seja fechado
            if stream:
                try:
                    # Espera o buffer esvaziar antes de fechar (opcional)
                    # await asyncio.to_thread(stream.stop_stream)
                    if stream.is_active():
                         stream.stop_stream()
                    stream.close()
                    logger.info("Stream de áudio de saída fechado.")
                except Exception as e_close:
                    logger.error(f"Erro ao fechar stream de áudio de saída: {e_close}")
            logger.info("play_audio concluído.")


    async def run(self):
        # (Função run inalterada, exceto por traceback em erro crítico - omitida para brevidade)
        logger.info("Iniciando AudioLoop...")
        max_retries = 3
        retry_delay_base = 2.0 # Aumentado ligeiramente o delay base

        attempt = 0
        while attempt <= max_retries and not self.stop_event.is_set():
            retry_delay = retry_delay_base * (2 ** attempt) # Backoff exponencial
            try:
                # Se for uma tentativa de reconexão, espera
                if attempt > 0:
                     logger.info(f"Tentativa de reconexão {attempt}/{max_retries} após {retry_delay:.1f}s...")
                     await asyncio.sleep(retry_delay)

                # --- Limpa estado da sessão anterior ---
                # Garante que a sessão antiga seja fechada se existir
                if self.session:
                    try:
                        await self.session.close()
                    except Exception: pass # Ignora erros ao fechar sessão antiga
                self.session = None
                self.audio_in_queue = None # Será recriado
                self.out_queue = None      # Será recriado
                self.awaiting_name_for_save_face = False # Reseta estado da função
                if self.thinking_event.is_set(): # Garante que não comece pensando
                    self.thinking_event.clear()
                # --- Fim da Limpeza ---


                # --- Tenta conectar ---
                # Verifica se o cliente Gemini foi inicializado com sucesso
                if client is None:
                    logger.info("ERRO: Cliente Gemini não inicializado. Não é possível conectar.")
                    self.stop_event.set()
                    break

                logger.info("Tentando conectar ao Gemini (Tentativa {attempt+1})...")
                async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                    self.session = session
                    session_id_str = 'N/A'
                    # Tenta obter o ID da sessão se o atributo existir
                    if hasattr(session, 'session_id'):
                         session_id_str = session.session_id
                    elif hasattr(session, '_session_id'): # Tenta atributo privado como fallback
                         session_id_str = session._session_id

                    logger.info(f"Sessão Gemini LiveConnect estabelecida (Tentativa {attempt+1}). ID: {session_id_str}")
                    attempt = 0 # Reseta tentativas em caso de sucesso

                    # Recria filas para a nova sessão
                    self.audio_in_queue = asyncio.Queue()
                    # Aumenta tamanho da fila de saída para acomodar bursts
                    self.out_queue = asyncio.Queue(maxsize=150)

                    # --- Inicia todas as tarefas da sessão ---
                    async with asyncio.TaskGroup() as tg:
                        logger.info("Iniciando tarefas da sessão...")
                        # Tarefa para ler input de texto do console
                        tg.create_task(self.send_text(), name="send_text_task")
                        # Tarefa para enviar dados (áudio/vídeo) da out_queue para Gemini
                        tg.create_task(self.send_realtime(), name="send_realtime_task")
                        # Tarefa para capturar áudio do microfone (se PyAudio estiver ok)
                        if pya: tg.create_task(self.listen_audio(), name="listen_audio_task")

                        # Tarefa para capturar vídeo (câmera ou tela)
                        if self.video_mode == "camera":
                            tg.create_task(self.get_frames(), name="get_frames_task")
                        elif self.video_mode == "screen":
                            tg.create_task(self.get_screen(), name="get_screen_task")
                        # Se mode for "none", nenhuma tarefa de vídeo é iniciada

                        # Tarefa para receber e processar respostas (áudio/texto/FC) do Gemini
                        tg.create_task(self.receive_audio(), name="receive_audio_task")
                        # Tarefa para tocar áudio recebido do Gemini (se PyAudio estiver ok)
                        if pya: tg.create_task(self.play_audio(), name="play_audio_task")

                        logger.info("Todas as tarefas da sessão iniciadas. Aguardando conclusão ou parada...")
                    # O bloco `async with tg:` espera todas as tarefas terminarem

                    logger.info("TaskGroup da sessão finalizado.")
                    # Se o TaskGroup terminou sem o stop_event ser setado,
                    # significa que a sessão Gemini provavelmente fechou ou uma tarefa crítica falhou.
                    if not self.stop_event.is_set():
                         logger.info("Sessão Gemini terminou inesperadamente ou TaskGroup concluído. Tentando reconectar...")
                         attempt += 1 # Incrementa para tentar reconectar
                    else:
                        # Se stop_event foi setado, saímos do loop principal
                        logger.info("Stop event detectado após TaskGroup. Encerrando loop de conexão.")
                        break


            except asyncio.CancelledError:
                logger.info("Loop principal (run) cancelado.")
                self.stop_event.set() # Garante que o evento de parada seja definido
                break
            except ExceptionGroup as eg:
                # Erro vindo do TaskGroup (uma ou mais tarefas falharam)
                logger.error(f"Erro(s) no TaskGroup (Tentativa {attempt+1}):")
                self.stop_event.set() # Para tudo se uma tarefa falhar criticamente
                for i, exc in enumerate(eg.exceptions):
                    logger.info(f"  Erro {i+1}: {type(exc).__name__} - {exc}")
                    # Imprime traceback para cada exceção no grupo
                    # traceback.print_exception(type(exc), exc, exc.__traceback__)
                attempt += 1 # Tenta reconectar após falha no TaskGroup
                self.session = None # Garante que a sessão seja considerada inválida
            except Exception as e:
                # Erro durante a conexão inicial ou outro erro inesperado no loop run
                logger.error(f"Erro ao conectar ou erro inesperado no método run (Tentativa {attempt+1}): {type(e).__name__} - {e}")
                traceback.print_exc() # Imprime traceback completo

                # Verifica se o erro é relacionado à conexão/sessão para decidir se retenta
                error_str_upper = str(e).upper()
                # Adiciona mais verificações de strings comuns de erro de conexão/gRPC
                is_connection_error = any(err_str in error_str_upper for err_str in [
                    "RST_STREAM", "UNAVAILABLE", "DEADLINE_EXCEEDED",
                    "LIVESESSION CLOSED", "LIVESESSION NOT CONNECTED",
                    "CONNECTIONCLOSEDERROR", "GOAWAY", "INTERNALERROR",
                    "FAILED TO ESTABLISH CONNECTION", "AUTHENTICATION" # Adiciona erro de autenticação
                ])

                if is_connection_error:
                    logger.info(f"Detectado erro relacionado à sessão ou conexão Gemini: {e}")
                else:
                    logger.info("Erro não parece ser diretamente de conexão. Verifique o traceback.")

                attempt += 1 # Incrementa tentativa
                self.session = None # Garante que a sessão seja considerada inválida
                if attempt > max_retries:
                     logger.info("Máximo de tentativas de reconexão atingido após erro. Encerrando.")
                     self.stop_event.set() # Define parada após exceder retries
                     break # Sai do loop while

        # --- Fim do Loop de Conexão ---
        if not self.stop_event.is_set() and attempt > max_retries:
             logger.info("Não foi possível restabelecer a conexão com Gemini após múltiplas tentativas.")
             self.stop_event.set() # Garante que o evento de parada esteja definido

        # --- Limpeza Final ---
        logger.info("Iniciando limpeza final em AudioLoop.run()...")
        self.stop_event.set() # Garante que todas as tarefas saibam que devem parar

        # Fecha a sessão Gemini se ainda estiver ativa
        if self.session:
            try:
                logger.info("Fechando sessão LiveConnect ativa...")
                await self.session.close()
                logger.info("Sessão LiveConnect fechada.")
            except Exception as e_close_session:
                logger.error(f"Erro ao fechar sessão LiveConnect na limpeza final: {e_close_session}")
        self.session = None

        # Sinaliza para a tarefa play_audio parar (colocando None na fila)
        if self.audio_in_queue:
            try:
                # Não espera se a fila estiver cheia, apenas tenta colocar
                self.audio_in_queue.put_nowait(None)
            except asyncio.QueueFull:
                logger.info("Aviso: Não foi possível colocar None na audio_in_queue (cheia) durante a limpeza.")
            except Exception as e_q_put:
                logger.error(f"Erro ao colocar None na audio_in_queue durante a limpeza: {e_q_put}")

        # Fecha janelas OpenCV se estiverem ativas
        if self.preview_window_active:
            logger.info("Fechando janelas OpenCV...")
            try:
                cv2.destroyAllWindows()
                logger.info("Janelas OpenCV destruídas no finally de run.")
            except Exception as e_cv_destroy_all:
                 logger.warning(f"AVISO: erro ao tentar fechar janelas de preview na limpeza final: {e_cv_destroy_all}")
            self.preview_window_active = False

        # Termina PyAudio
        if pya:
            try:
                logger.info("Terminando PyAudio...")
                # Não precisa chamar stop_stream/close aqui, pois play_audio/listen_audio já o fazem
                pya.terminate()
                logger.info("Recursos de PyAudio liberados.")
            except Exception as e_pya:
                logger.error(f"Erro ao terminar PyAudio: {e_pya}")
        logger.info("Limpeza de AudioLoop.run() concluída.")


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

    # Valida o argumento show_preview
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
            logger.error(f"ERRO CRÍTICO: Modelo YOLO '{YOLO_MODEL_PATH}' Não encontrado.")
            logger.info("Verifique o caminho em BaseDir e YOLO_MODEL_PATH ou baixe o modelo.")
            exit(1) # Sai se o modelo não for encontrado

    # Verifica se PyAudio foi inicializado
    if not pya:
         logger.info("ERRO CRÍTICO: PyAudio não pôde ser inicializado.")
         logger.info("Verifique a instalação do PyAudio e suas dependências (como PortAudio).")
         logger.info("O programa não pode funcionar sem áudio. Encerrando.")
         exit(1) # Sai se PyAudio falhou

    # Verifica se o cliente Gemini foi inicializado
    if client is None:
        logger.info("ERRO CRÍTICO: Cliente Gemini não pôde ser inicializado (verifique API Key/conexão). Encerrando.")
        exit(1)

    # Verifica se o arquivo de prompt foi carregado (system_instruction_text deve existir)
    if 'system_instruction_text' not in globals() or not system_instruction_text or system_instruction_text == "Você é um assistente prestativo.":
         logger.info("AVISO: Falha ao carregar a instrução do sistema do arquivo ou arquivo não encontrado. Usando prompt padrão.")
         # Decide se quer continuar com o prompt padrão ou sair
         # exit(1) # Descomente para sair se o prompt for essencial

    main_loop = None
    try:
        logger.info(f"Iniciando Trackie no modo: {args.mode}")
        # Cria a instância principal
        main_loop = AudioLoop(video_mode=args.mode, show_preview=show_actual_preview)
        # Executa o loop principal assíncrono
        asyncio.run(main_loop.run())

    except KeyboardInterrupt:
        logger.info("\nInterrupção pelo teclado recebida (Ctrl+C). Encerrando...")
        if main_loop:
            logger.info("Sinalizando parada para as tarefas...")
            main_loop.stop_event.set()
            # Dê um pequeno tempo para as tarefas tentarem limpar antes de sair
            # time.sleep(1) # Opcional
    except Exception as e:
        # Captura qualquer outra exceção não tratada no nível superior
        logger.error(f"Erro inesperado e não tratado no bloco __main__: {type(e).__name__}: {e}")
        traceback.print_exc()
        if main_loop:
            logger.info("Sinalizando parada devido a erro inesperado...")
            main_loop.stop_event.set()
    finally:
        # Este bloco sempre será executado, mesmo após interrupção ou erro
        logger.info("Bloco __main__ finalizado.")
        # Verifica se PyAudio ainda precisa ser terminado (caso run() não tenha chegado ao fim)
        # if pya and main_loop and not main_loop.stop_event.is_set(): # Se run não terminou normalmente
        #      try:
        #          logger.info("Tentando terminar PyAudio no finally do main...")
        #          pya.terminate()
        #      except Exception as e_pya_final:
        #          logger.error(f"Erro ao terminar PyAudio no finally do main: {e_pya_final}")

        logger.info("Programa completamente finalizado.")