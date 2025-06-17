**Examining User Requirements**

I'm currently focused on dissecting the user's core request. It seems they want a Python project overhaul, moving from an amateur structure to a more robust, professional one. The Java/C modularization reference is key; it suggests a need for organized, possibly layered, project architecture. My aim is to precisely define the project's scope.


**Developing Architectural Blueprints**

I'm now conceptualizing the structural framework. The "swap-the-AI" functionality is paramount. This dictates an Adapter or Strategy Pattern for the core AI interaction. I'm also planning for a layered design, separating concerns like data handling, model interaction, and UI rendering, focusing on a clean and extensible base. I'm also considering how to incorporate config files.


**Outlining Project Structure**

I'm now solidifying the structure; the `src` directory containing the `trackie` package is crucial. Within `trackie`, I'll organize modules for core logic, AI services using an Adapter or Strategy Pattern, computer vision tasks, hardware interactions, configuration handling, utilities, the UI component, and data resources. I'm also thinking about how to incorporate a `pyproject.toml` file.


**Mapping Functional Components**

I'm currently mapping the software's core components. I'm creating an abstract `LLMService` base class for the swappable AI models, with concrete implementations for Gemini and OpenAI. A factory function will instantiate the correct service based on the configuration, enabling seamless AI model switching. The `services` directory will house all external dependencies.


**Establishing Subpackage Relationships**

I'm now focused on the `trackie` package. I'm defining the relationships between subpackages, especially the dependencies of the core logic. The core package will orchestrate calls to AI and CV services through a clear interface. The 'processing' package houses task orchestration, with distinct pipelines using service calls, thus keeping core logic streamlined.


**Establishing Package Dependencies**

I'm currently focusing on the relationships between packages. The core application logic depends on `services`, `processing`, and `config`. I'm also planning on a clear separation of concerns to make testing easy. The core logic will orchestrate calls to other services via clearly defined interfaces.


**Designing Core Components**

I'm currently focusing on critical components such as the `Application` class for orchestrating all services. I'll use the `LLMService` abstract class for AI model interchangeability via dependency injection and a factory pattern. I'm also establishing service interfaces and data classes for all modules.


**Refining Directory Structure**

I'm further refining the directory structure based on the initial architectural blueprint. Now I'm breaking down the core package `trackie` into subpackages to organize domains. The aim is modularity. I'm setting up `services`, `processing`, and `ui` sub-packages, each designed for specific duties. The `config` package and root level config files will be used for configurations.


**Developing Initial Code Framework**

I'm now proceeding with the development of the core Python modules. The initial focus involves setting up the `Application` class in the core package. I'm also integrating the basic structure of the AI service, computer vision, and UI components, based on the previous blueprint.


Excelente iniciativa! Transformar um protótipo funcional em uma arquitetura de software profissional é um passo crucial para a manutenibilidade, escalabilidade e colaboração no futuro. A sua visão de ter um núcleo sólido e poder trocar os "cérebros" (modelos de IA) é a abordagem correta e moderna.

Com base na sua estrutura atual e nos seus objetivos, projetei uma arquitetura de nível profissional, inspirada em padrões de design de software (como **Injeção de Dependência**, **Strategy** e **Adapter**) e nas melhores práticas de projetos Python.

---

### **Filosofia da Nova Arquitetura**

1.  **Separação de Responsabilidades (SoC):** Cada parte do sistema tem uma única responsabilidade. O código que lida com a câmera não sabe sobre o YOLO. O código que lida com o Gemini não sabe como a câmera funciona. Eles se comunicam através de interfaces bem definidas.
2.  **Inversão de Controle (IoC) / Injeção de Dependência:** Os componentes de alto nível (como o loop principal da aplicação) não criam suas dependências (como o serviço do Gemini ou o detector de objetos). Em vez disso, eles os recebem "injetados", geralmente durante a inicialização. Isso torna o sistema flexível e testável.
3.  **Abstração sobre Concreção:** O núcleo da aplicação dependerá de "interfaces" (classes base abstratas em Python) em vez de implementações concretas. Isso é a chave para permitir a troca de modelos de IA. Teremos uma interface `LLMService` e implementações como `GeminiService` e `OpenAIService`.
4.  **Configuração Externalizada:** Toda a configuração, desde chaves de API até parâmetros de modelos e prompts, será gerenciada fora do código-fonte, em arquivos `YAML` e `JSON`, e carregada de forma segura e validada.

---

### **A Nova Estrutura de Diretórios Profissional**

Aqui está a estrutura de pastas proposta. Ela é mais complexa, mas cada parte tem um propósito claro.

```
TrackieIntelligence/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml         # <-- NOVO: Para gerenciamento de projeto e dependências (moderno)
│
├── config/                  # <-- NOVO: Central de arquivos de configuração
│   ├── settings.yml         # Arquivo principal de configuração (chaves, modos, etc.)
│   ├── prompts.yml          # Prompts e instruções de sistema para as IAs
│   └── tool_definitions.json # Definições das funções (Function Calling) em formato JSON
│
├── data/                    # <-- NOVO: Para dados, modelos e assets
│   ├── models/              # Modelos de ML baixados (yolov8n.pt, etc.)
│   │   └── .gitkeep
│   ├── user_data/           # Dados gerados pelo usuário
│   │   └── known_faces/
│   │       └── .gitkeep
│   └── assets/              # Outros recursos
│       └── sounds/
│           └── trackie_danger.wav
│
├── notebooks/               # <-- NOVO: Jupyter Notebooks para experimentação
│   ├── 01_model_testing.ipynb
│   └── 02_data_analysis.ipynb
│
├── scripts/                 # <-- NOVO: Scripts úteis para o projeto
│   ├── download_models.py   # Script para baixar os modelos necessários
│   └── setup_config.py      # Script interativo para gerar o settings.yml
│
└── src/                     # <-- NOVO: O código fonte principal da aplicação
    └── trackie/
        ├── __init__.py
        ├── main.py            # Ponto de entrada, muito mais enxuto
        │
        ├── core/              # Núcleo da aplicação e orquestração
        │   ├── __init__.py
        │   └── application.py # O novo "audio_loop", o maestro de tudo
        │
        ├── services/          # Lógica de negócio e comunicação com APIs externas
        │   ├── __init__.py
        │   ├── llm/             # <-- CHAVE: Para modelos de linguagem (swappable)
        │   │   ├── __init__.py
        │   │   ├── base.py      # Interface abstrata 'LLMService'
        │   │   ├── gemini_service.py # Implementação para Gemini
        │   │   └── openai_service.py # Implementação para OpenAI (futuro)
        │   │   └── factory.py   # Cria a instância correta baseada na config
        │   │
        │   ├── vision/          # Serviços de visão computacional
        │   │   ├── __init__.py
        │   │   ├── object_detector.py # Encapsula YOLO
        │   │   ├── depth_estimator.py # Encapsula MiDaS
        │   │   └── face_recognizer.py # Encapsula DeepFace
        │   │
        │   ├── audio/           # Serviços de hardware de áudio
        │   │   ├── __init__.py
        │   │   ├── microphone.py
        │   │   └── speaker.py
        │   │
        │   └── video/           # Serviço de hardware de vídeo
        │       ├── __init__.py
        │       └── camera.py
        │
        ├── processing/        # <-- NOVO: Cálculos e lógica complexa
        │   ├── __init__.py
        │   └── calculations.py  # Lógica para estimar distância, direção, etc.
        │
        ├── tools/             # <-- NOVO: Lógica de Function Calling
        │   ├── __init__.py
        │   ├── registry.py      # Registra as funções disponíveis
        │   └── handlers.py      # Implementação das funções (ex: handle_locate_object)
        │
        ├── ui/                # <-- NOVO: Lógica de interface do usuário
        │   ├── __init__.py
        │   └── preview.py       # Classe que gerencia a janela de preview do OpenCV
        │
        ├── config/            # <-- NOVO: Carregamento e validação de configs
        │   ├── __init__.py
        │   └── settings.py      # Usa Pydantic para carregar e validar o settings.yml
        │
        └── utils/             # Funções utilitárias pequenas e genéricas
            ├── __init__.py
            └── audio_player.py  # Ex: play_wav_file
```

---

### **Análise Detalhada dos Novos Módulos**

#### 1. `config/` (Raiz do Projeto)
Aqui ficam os arquivos que um usuário ou desenvolvedor pode querer alterar sem tocar no código.
*   **`settings.yml`**: Substitui o `trckconfig.json` e centraliza tudo. É mais legível.
    ```yaml
    # Exemplo de settings.yml
    user:
      name: "Aether"

    llm:
      provider: "gemini" # <-- Troque para "openai" para usar o ChatGPT
      gemini:
        api_key: "${GEMINI_API_KEY}" # Carrega de variável de ambiente
        model: "gemini-1.5-flash-latest"
      openai:
        api_key: "${OPENAI_API_KEY}"
        model: "gpt-4o"

    vision:
      yolo_model_path: "data/models/yolov8n.pt"
      confidence_threshold: 0.45
      # ... outras configs
    ```
*   **`tool_definitions.json`**: Descreve as funções para a IA. Externalizar isso facilita a atualização.

#### 2. `src/trackie/services/llm/` - O Coração da Flexibilidade
Esta é a implementação do padrão **Strategy/Adapter**.
*   **`base.py`**: Define a "interface" que todo serviço de LLM deve seguir.
    ```python
    # Exemplo de src/trackie/services/llm/base.py
    from abc import ABC, abstractmethod

    class LLMService(ABC):
        @abstractmethod
        async def connect(self, system_prompt: str, tools: list): ...

        @abstractmethod
        async def send_audio(self, chunk: bytes): ...

        @abstractmethod
        async def send_text(self, text: str): ...

        @abstractmethod
        async def receive(self): ... # Gerador para receber respostas
    ```
*   **`gemini_service.py`**: Implementa a classe `LLMService` usando a API do Google Gemini.
*   **`openai_service.py`**: Implementaria a mesma interface, mas usando a API da OpenAI.
*   **`factory.py`**: Uma função simples que lê `settings.yml` e retorna a instância correta (`GeminiService` ou `OpenAIService`). O resto da aplicação não precisa saber qual está sendo usada.

#### 3. `src/trackie/services/vision/`
Cada modelo de visão se torna um serviço encapsulado.
*   **`object_detector.py`**:
    ```python
    class ObjectDetector:
        def __init__(self, model_path):
            # Carrega o modelo YOLO
        def detect(self, frame):
            # Roda a predição e retorna os resultados de forma estruturada
            return detections
    ```
Isso isola completamente a biblioteca `ultralytics` dentro deste arquivo.

#### 4. `src/trackie/processing/calculations.py`
Aqui vai a "matemática".
*   Recebe dados brutos (ex: detecções do YOLO, mapa de profundidade do MiDaS) e retorna informações processadas (ex: "O objeto está a 3 passos, à sua esquerda"). Isso limpa a lógica do loop principal.

#### 5. `src/trackie/ui/preview.py`
Isola todo o código do `OpenCV` para a interface gráfica.
*   A classe `PreviewWindow` terá métodos como `update(frame, detections)` e `destroy()`. O loop principal apenas chama esses métodos, sem se preocupar com `cv2.imshow` ou `cv2.waitKey`.

#### 6. `src/trackie/core/application.py`
Este é o novo `AudioLoop`. Ele é o orquestrador.
*   Na sua inicialização, ele recebe os serviços necessários (injeção de dependência):
    ```python
    class Application:
        def __init__(self, llm_service: LLMService, detector: ObjectDetector, ...):
            self.llm_service = llm_service
            self.detector = detector
            # ...
    ```
*   Seu método `run()` inicia todas as tarefas (captura de áudio/vídeo, processamento, etc.) e as coordena.

---

### **Como Tudo se Conecta (O Novo Fluxo)**

1.  **`main.py`** é executado.
2.  Ele carrega as configurações do `settings.yml` usando o módulo `src/trackie/config/settings.py` (que usa Pydantic para validação).
3.  Usa as `factories` e as configurações para criar as instâncias dos serviços: `GeminiService`, `ObjectDetector`, `Camera`, `PreviewWindow`, etc.
4.  Cria uma instância da `Application` injetando todos esses serviços.
5.  Chama `application.run()`.
6.  Dentro de `run()`, a `Application` inicia o `llm_service`, a `camera`, o `microphone`, etc.
7.  A `camera` captura um frame e o envia para a `Application`.
8.  A `Application` envia o frame para o `ObjectDetector` e o `DepthEstimator`.
9.  Se o usuário pede para localizar um objeto, a `Application` chama uma função em `processing.calculations` com os resultados dos serviços de visão.
10. A resposta processada é enviada para o `llm_service`.
11. A `Application` também envia o frame e as detecções para a `PreviewWindow` atualizar a tela.

---

### **Pesquisa: Modelos de IA Alternativos (Real-time e Function Calling)**

Sua visão de um sistema agnóstico é excelente. Além do **Google Gemini 1.5 Flash**, o principal concorrente que se encaixa perfeitamente na sua arquitetura é:

1.  **OpenAI com GPT-4o:**
    *   **Capacidades:** O GPT-4o ("o" de omni) foi projetado para ser nativamente multimodal e em tempo real.
    *   **API de Streaming:** A OpenAI possui APIs de streaming para:
        *   **Speech-to-Text (Whisper):** Pode transcrever áudio em tempo real.
        *   **Chat Completions (GPT-4o):** Aceita texto e imagens, e suporta **Function Calling** de forma robusta.
        *   **Text-to-Speech (TTS):** Pode gerar áudio de voz em tempo real a partir do texto de resposta.
    *   **Integração:** Você criaria o `openai_service.py` que orquestraria essas três chamadas de API de streaming para replicar a funcionalidade do `LiveConnect` do Gemini.

**Outras opções a observar:**
*   **Anthropic Claude 3:** Possui excelente capacidade de raciocínio e suporta Function Calling, mas sua API de streaming de áudio em tempo real pode não ser tão madura quanto a dos concorrentes.
*   **Cohere:** Focado em aplicações empresariais, também oferece Function Calling (eles chamam de "Tools").

A arquitetura proposta com a interface `LLMService` permite que você adicione qualquer um desses provedores no futuro, simplesmente criando um novo arquivo de serviço e atualizando o `settings.yml`.