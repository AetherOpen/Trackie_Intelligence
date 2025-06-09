# trackie_app/app_config.py
import os
import json
from platform import architecture
from .logger_config import get_logger # Import relativo

logger = get_logger(__name__)

# --- Constantes ---
DEFAULT_MODE = "camera"

from pathlib import Path
script_path = Path(__file__).resolve()
script_dir = script_path.parent
BASE_DIR = script_dir.parent
ARCHITECTURE = BASE_DIR / 'architecture'


print(f"Caminho do script: {script_path}")
print(f"Diretório do script: {script_dir}")
print(f"BASE_DIR definido como: {BASE_DIR}")


DANGER_SOUND_PATH = os.path.join(BASE_DIR, "WorkTools", "SoundBibTrackie", "Trackiedanger.wav")

# --- Caminho para o arquivo de prompt e config ---
SYSTEM_INSTRUCTION_PATH = os.path.join(BASE_DIR, "WorkTools", "ForSystemInstructions", "TrckItcs.txt")
CONFIG_PATH = os.path.join(BASE_DIR, "UserSettings", "trckconfig.json")

# Carregar o JSON e extrair CFG'S
TRCKUSER = 'Usuário Padrão' # Valor padrão
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    TRCKUSER = config_data.get('trckuser', 'Usuário Padrão')
except FileNotFoundError:
    logger.error(f"Arquivo de configuração não encontrado: {CONFIG_PATH}")
except json.JSONDecodeError as e:
    logger.error(f"Erro ao decodificar JSON em {CONFIG_PATH}: {e}")
except Exception as e:
    logger.error(f"Erro ao carregar configuração de {CONFIG_PATH}: {e}")

# YOLO
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "WorkTools", "yolov8n.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.40
DANGER_CLASSES={
	"faca": [ "knife" ],
	"tesoura": [ "scissors" ],
	"barbeador": [ "razor" ],
	"serra": [ "saw" ],
	"machado": [ "axe" ],
	"machadinha": [ "hatchet" ],
	"arma_de_fogo": [ "gun" ],
	"pistola": [ "pistol" ],
	"rifle": [ "rifle" ],
	"espingarda": [ "shotgun" ],
	"rev�lver": [ "revolver" ],
	"bomba": [ "bomb" ],
	"granada": [ "grenade" ],
	"fogo": [ "fire" ],
	"chama": [ "flame" ],
	"fuma�a": [ "smoke" ],
	"isqueiro": [ "lighter" ],
	"f�sforos": [ "matches" ],
	"fog�o": [ "stove" ],
	"superf�cie_quente": [ "hot surface" ],
	"vela": [ "candle" ],
	"queimador": [ "burner" ],
	"fio_energizado": [ "live_wire" ],
	"tomada_el�trica": [ "electric_outlet" ],
	"bateria": [ "battery" ],
	"vidro_quebrado": [ "broken_glass" ],
	"estilha�o": [ "shard" ],
	"agulha": [ "needle" ],
	"seringa": [ "syringe" ],
	"martelo": [ "hammer" ],
	"chave_de_fenda": [ "wrench" ],
	"furadeira": [ "drill" ],
	"motosserra": [ "chainsaw" ],
	"carro": [ "car" ],
	"motocicleta": [ "motorcycle" ],
	"bicicleta": [ "bicycle" ],
	"caminh�o": [ "truck" ],
	"�nibus": [ "bus" ],
	"urso": [ "bear" ],
	"cobra": [ "snake" ],
	"aranha": [ "spider" ],
	"jacaré": [ "alligator" ],
	"penhasco": [ "cliff" ],
	"buraco": [ "hole" ],
	"escada": [ "stairs" ]
}
YOLO_CLASS_MAP={
  "pessoa": [ "person" ],
  "gato": [ "cat" ],
  "cachorro": [ "dog" ],
  "coelho": [ "rabbit" ],
  "urso": [ "bear" ],
  "elefante": [ "elephant" ],
  "zebra": [ "zebra" ],
  "girafa": [ "giraffe" ],
  "vaca": [ "cow" ],
  "cavalo": [ "horse" ],
  "ovelha": [ "sheep" ],
  "macaco": [ "monkey" ],
  "bicicleta": [ "bicycle" ],
  "moto": [ "motorcycle" ],
  "carro": [ "car", "automobile" ],
  "ônibus": [ "bus" ],
  "trem": [ "train" ],
  "caminhão": [ "truck" ],
  "avião": [ "airplane" ],
  "barco": [ "boat" ],
  "skate": [ "skateboard" ],
  "prancha de surf": [ "surfboard" ],
  "tênis": [ "sneakers", "tennis shoes" ],
  "mesa de jantar": [ "dining table" ],
  "mesa": [ "table", "desk" ],
  "cadeira": [ "chair" ],
  "sofá": [ "couch", "sofa" ],
  "cama": [ "bed" ],
  "vaso de planta": [ "potted plant", "flower pot" ],
  "banheiro": [ "bathroom", "restroom", "toilet" ],
  "televisão": [ "tv", "television" ],
  "abajur": [ "lamp", "table lamp" ],
  "espelho": [ "mirror" ],
  "laptop": [ "laptop" ],
  "computador": [ "computer", "desktop computer" ],
  "teclado": [ "keyboard" ],
  "mouse": [ "mouse" ],
  "controle remoto": [ "remote", "remote control" ],
  "celular": [ "cell phone", "mobile phone" ],
  "micro-ondas": [ "microwave", "microwave oven" ],
  "forno": [ "oven" ],
  "torradeira": [ "toaster" ],
  "geladeira": [ "refrigerator", "fridge" ],
  "caixa de som": [ "speaker" ],
  "câmera": [ "camera" ],
  "garrafa": [ "bottle" ],
  "copo": [ "cup", "glass" ],
  "taça de vinho": [ "wine glass" ],
  "taça": [ "glass", "goblet" ],
  "prato": [ "plate", "dish" ],
  "tigela": [ "bowl" ],
  "garfo": [ "fork" ],
  "faca": [ "knife" ],
  "colher": [ "spoon" ],
  "panela": [ "pot", "pan" ],
  "frigideira": [ "frying pan", "skillet" ],
  "martelo": [ "hammer" ],
  "chave inglesa": [ "wrench", "spanner" ],
  "furadeira": [ "drill" ],
  "parafusadeira": [ "screwdriver", "power screwdriver", "electric screwdriver" ],
  "serra": [ "saw" ],
  "roçadeira": [ "brush cutter", "strimmer", "weed whacker" ],
  "alicate": [ "pliers" ],
  "chave de fenda": [ "screwdriver" ],
  "lanterna": [ "flashlight", "torch" ],
  "fita métrica": [ "tape measure" ],
  "mochila": [ "backpack", "rucksack" ],
  "bolsa": [ "bag", "handbag", "purse" ],
  "carteira": [ "wallet" ],
  "óculos": [ "glasses", "eyeglasses", "spectacles" ],
  "relógio": [ "watch", "clock" ],
  "chinelo": [ "flip-flops", "sandals", "slipper" ],
  "sapato": [ "shoe" ],
  "sanduíche": [ "sandwich" ],
  "hambúrguer": [ "hamburger", "burger" ],
  "banana": [ "banana" ],
  "maçã": [ "apple" ],
  "laranja": [ "orange" ],
  "bolo": [ "cake" ],
  "rosquinha": [ "donut", "doughnut" ],
  "pizza": [ "pizza" ],
  "cachorro-quente": [ "hot dog" ],
  "escova de dentes": [ "toothbrush" ],
  "secador de cabelo": [ "hair dryer" ],
  "cotonete": [ "cotton swab", "Q-tip" ],
  "sacola plástica": [ "plastic bag" ],
  "livro": [ "book" ],
  "vaso": [ "vase" ],
  "bola": [ "ball", "sports ball" ],
  "bexiga": [ "balloon" ],
  "pipa": [ "kite" ],
  "luva": [ "glove" ],
  "skis": [ "skis" ],
  "snowboard": [ "snowboard" ],
  "tesoura": [ "scissors" ]
}

# DeepFace
DB_PATH = os.path.join(BASE_DIR, "UserSettings", "known_faces")
DEEPFACE_MODEL_NAME = 'VGG-Face'
DEEPFACE_DETECTOR_BACKEND = 'opencv'
DEEPFACE_DISTANCE_METRIC = 'cosine'

# MiDaS
MIDAS_MODEL_TYPE = "MiDaS_large"
METERS_PER_STEP = 0.7

# Áudio
AUDIO_FORMAT = None # Será definido em external_apis.py
AUDIO_CHANNELS = 1
AUDIO_SEND_SAMPLE_RATE = 16000
AUDIO_RECEIVE_SAMPLE_RATE = 24000
AUDIO_CHUNK_SIZE = 1024

# Gemini Model
#GEMINI_MODEL_NAME = "models/gemini-2.5-flash-preview-native-audio-dialog"
GEMINI_MODEL_NAME =  "models/gemini-2.0-flash-live-001"
# --- Carregar Instrução do Sistema do Arquivo ---
SYSTEM_INSTRUCTION_TEXT = "Você é um assistente prestativo." # Prompt padrão mínimo
try:
    if not os.path.exists(SYSTEM_INSTRUCTION_PATH):
        logger.warning(f"AVISO: Arquivo de instrução do sistema não encontrado em '{SYSTEM_INSTRUCTION_PATH}'. Usando prompt padrão.")
        # exit() # Removido para permitir execução mesmo sem o arquivo, usando o padrão.
    else:
        with open(SYSTEM_INSTRUCTION_PATH, 'r', encoding='utf-8') as f:
            SYSTEM_INSTRUCTION_TEXT = f.read()
        logger.info(f"Instrução do sistema carregada de: {SYSTEM_INSTRUCTION_PATH}")
except Exception as e:
    logger.error(f"Erro ao ler o arquivo de instrução do sistema {SYSTEM_INSTRUCTION_PATH}: {e}")
    logger.info("Usando um prompt padrão mínimo.")
    # traceback.print_exc() # Já logado pelo logger.error

# API Key - Carregada em external_apis.py para manter este arquivo focado em config.
API_KEY = None