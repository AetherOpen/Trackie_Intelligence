# trackie_app/gemini_settings.py
from google import genai
from google.genai import types as genai_types
from google.genai.types import Content, Part, GenerateContentConfig, LiveConnectConfig, Modality # Explicitamente importado
from google.genai import errors as genai_errors
from google.protobuf.struct_pb2 import Value

from .app_config import TRCKUSER, SYSTEM_INSTRUCTION_TEXT # Importa configurações necessárias
from .logger_config import get_logger

logger = get_logger(__name__)

# --- Ferramentas Gemini (Function Calling) ---
GEMINI_TOOLS = [
    genai_types.Tool(code_execution=genai_types.ToolCodeExecution()), #padrão recomendado pela google
    genai_types.Tool(google_search=genai_types.GoogleSearch()), #padrão recomendado pela google
    genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name="save_known_face",
                description=(
                    "Salva o rosto da pessoa atualmente em foco pela câmera. "
                    "Esta função requer o nome da pessoa. Se 'person_name' não for fornecido na chamada inicial, "
                    "a IA deve solicitar explicitamente ao usuário: 'Por favor, informe o nome da pessoa para salvar o rosto.' "
                    "Após receber o nome, a função tenta salvar o rosto. "
                    "Confirma o sucesso com: 'Rosto salvo com sucesso para [nome].' "
                    "Em caso de falha na captura, retorna: 'Erro: Não foi possível capturar o rosto. Tente novamente.'"
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "person_name": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description="O nome da pessoa cujo rosto será salvo. Este nome é necessário para o salvamento."
                        )
                    },
                    required=["person_name"]
                )
            ),
            genai_types.FunctionDeclaration(
                name="identify_person_in_front",
                description=(
                    "Identifica a pessoa atualmente em foco pela câmera usando o banco de dados de rostos conhecidos. "
                    "Deve ser chamada apenas quando o usuário expressa explicitamente a intenção de identificar alguém. "
                    "Se múltiplos rostos forem detectados, a função prioriza o rosto mais proeminente ou central na imagem. "
                    "Retorna a identificação com um grau de confiança (ex: 'Identificado como [nome] com 95% de confiança.'). "
                    "Se nenhum rosto conhecido corresponder, retorna: 'Pessoa não reconhecida.' "
                    "Se nenhum rosto for detectado pela câmera, retorna: 'Nenhum rosto detectado pela câmera.'"
                ),
                parameters=genai_types.Schema(type=genai_types.Type.OBJECT, properties={})
            ),
            genai_types.FunctionDeclaration(
                name="locate_object_and_estimate_distance",
                description=(
                    "Localiza um objeto especificado pelo usuário no campo de visão da câmera em tempo real, "
                    "estima a distância até ele (usando MiDaS internamente) e informa essa distância em passos, "
                    "juntamente com uma direção relativa (ex: 'à sua esquerda', 'em frente', 'à sua direita', 'ligeiramente à esquerda/direita'). "
                    "Esta função é projetada para auxiliar usuários com deficiência visual ou baixa visão. "
                    "Se o nome do objeto ('object_name') não for fornecido, a IA deve perguntar: 'Qual objeto você gostaria de localizar?' "
                    "A resposta deve ser clara, por exemplo: 'Cadeira localizada a aproximadamente 5 passos à sua frente.' "
                    "Se o objeto não for encontrado no campo de visão atual, retorna: 'Não foi possível localizar o [nome_do_objeto] no momento.' "
                    "Se o objeto for encontrado mas a distância não puder ser estimada confiavelmente, retorna: "
                    "'Objeto [nome_do_objeto] localizado, mas não foi possível estimar a distância com precisão.'"
                ),
                parameters=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "object_name": genai_types.Schema(
                            type=genai_types.Type.STRING,
                            description="O nome do objeto que o usuário deseja localizar e cuja distância deve ser estimada (ex: 'cadeira', 'mesa', 'porta')."
                        )
                    },
                    required=["object_name"]
                )
            )
        ]
    )
]
# --- Configuração da Sessão LiveConnect Gemini ---
GEMINI_LIVE_CONNECT_CONFIG = None
try:
    GEMINI_LIVE_CONNECT_CONFIG = genai_types.LiveConnectConfig(
        temperature=0.2,
        response_modalities=["audio"],
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
        speech_config=genai_types.SpeechConfig(
            language_code="pt-BR",
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name="Zephyr")
            )
        ),
        tools=GEMINI_TOOLS,
        system_instruction=genai_types.Content(
            parts=[
                genai_types.Part.from_text(text=f"the name of your user is:  {TRCKUSER}, "),
                genai_types.Part.from_text(text=SYSTEM_INSTRUCTION_TEXT)
            ],
            role="system"
        )
    )
    logger.info("Configuração LiveConnect do Gemini criada.")
except Exception as e:
    logger.error(f"Erro ao criar LiveConnectConfig do Gemini: {e}")
    GEMINI_LIVE_CONNECT_CONFIG = None