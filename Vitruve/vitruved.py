import sys
import subprocess
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QFrame, QGraphicsOpacityEffect
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize, QTimer
from PySide6.QtGui import QFont, QIcon

# Tente encontrar um ícone (opcional, coloque um 'icon.png' na pasta Vitruve se quiser)
# Se não encontrar, não tem problema.
ICON_PATH = Path(__file__).resolve().parent / "trackie_icon.png"

class TrackieApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_process = None
        self.base_dir = self._get_base_dir()

        self.setWindowTitle("Trackie Controller")
        self.setMinimumSize(QSize(500, 350)) # Tamanho mínimo da janela

        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        # Efeito de opacidade para fade-in
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(800) # 0.8 segundos para fade-in
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Widget Central e Layout Principal
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setContentsMargins(30, 30, 30, 30) # Margens internas
        main_layout.setSpacing(25) # Espaçamento entre widgets

        # Título
        title_label = QLabel("Trackie Intelligence", self)
        title_label.setObjectName("titleLabel") # Para QSS
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Linha Separadora (decorativa)
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setObjectName("separator")
        main_layout.addWidget(separator)

        # Botões
        button_layout = QVBoxLayout() # Layout vertical para os botões
        button_layout.setSpacing(15)

        self.btn_start_trackie = QPushButton("🚀 Iniciar Trackie", self)
        self.btn_start_trackie.setObjectName("actionButton")
        self.btn_start_trackie.clicked.connect(self.start_trackie_normal)
        self.btn_start_trackie.setToolTip("Executa o Trackie em modo normal (sem preview).")
        button_layout.addWidget(self.btn_start_trackie)

        self.btn_start_preview = QPushButton("👁️ Ver a Mente do Trackie", self)
        self.btn_start_preview.setObjectName("actionButton")
        self.btn_start_preview.clicked.connect(self.start_trackie_preview)
        self.btn_start_preview.setToolTip("Executa o Trackie com visualização de depuração.")
        button_layout.addWidget(self.btn_start_preview)
        
        self.btn_stop_trackie = QPushButton("🛑 Parar Trackie", self)
        self.btn_stop_trackie.setObjectName("stopButton")
        self.btn_stop_trackie.clicked.connect(self.stop_current_trackie)
        self.btn_stop_trackie.setToolTip("Para qualquer instância do Trackie em execução.")
        self.btn_stop_trackie.setEnabled(False) # Começa desabilitado
        button_layout.addWidget(self.btn_stop_trackie)


        main_layout.addLayout(button_layout)
        main_layout.addStretch() # Adiciona espaço flexível na parte inferior

        # Aplicar Estilos (QSS)
        self._apply_styles()

        # Iniciar animação de fade-in
        self.animation.start()

    def _get_base_dir(self):
        # Assume que este script está em Vitruve/nome_do_arquivo.py
        # BASE_DIR é o pai da pasta Vitruve
        script_path = Path(__file__).resolve()
        vitruve_dir = script_path.parent
        base_dir = vitruve_dir.parent
        print(f"BASE_DIR detectado: {base_dir}") # Para depuração
        return base_dir

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2B2B2B; /* Fundo escuro (Material Darker) */
            }
            QWidget { /* Estilo padrão para texto em outros widgets se necessário */
                color: #EEFFFF;
            }
            QLabel#titleLabel {
                color: #82AAFF; /* Azul claro para o título */
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 32px;
                font-weight: 600; /* Semi-bold */
                padding-bottom: 10px;
                qproperty-alignment: 'AlignCenter';
            }
            QFrame#separator {
                background-color: #4E5254;
                height: 1px;
                margin-bottom: 15px;
            }
            QPushButton#actionButton, QPushButton#stopButton {
                color: #EEFFFF; /* Texto claro */
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 17px;
                font-weight: 500;
                padding: 18px;
                border-radius: 8px; /* Cantos arredondados */
                min-height: 50px; /* Altura mínima */
                border: 1px solid #555555; /* Borda sutil */
            }
            QPushButton#actionButton {
                 background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #007ACC, stop:1 #005C99); /* Gradiente azul */
            }
            QPushButton#actionButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #008AE6, stop:1 #006BB3); /* Azul mais claro no hover */
                border: 1px solid #008AE6;
            }
            QPushButton#actionButton:pressed {
                background-color: #004C80; /* Azul mais escuro quando pressionado */
            }
            QPushButton#stopButton {
                 background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #C62828, stop:1 #A31515); /* Gradiente vermelho */
            }
            QPushButton#stopButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #D32F2F, stop:1 #B71C1C);
                border: 1px solid #D32F2F;
            }
            QPushButton#stopButton:pressed {
                background-color: #930E0E;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888888;
                border: 1px solid #5A5A5A;
            }
            QToolTip {
                background-color: #3E3E3E;
                color: #EEFFFF;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 4px;
            }
        """)

    def _run_command(self, arguments):
        if self.current_process and self.current_process.poll() is None:
            print("Trackie já está em execução. Pare-o primeiro.")
            # Poderia mostrar uma mensagem na GUI aqui
            # self.statusBar().showMessage("Trackie já está em execução. Pare-o primeiro.", 3000)
            return

        command = [sys.executable, "-m", "Architecture.main"] + arguments
        print(f"Executando comando: {' '.join(command)} em {self.base_dir}")

        # Configurações para ocultar o console do subprocesso
        startupinfo = None
        kwargs_subprocess = {}
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            # startupinfo.wShowWindow = subprocess.SW_HIDE # Oculta a janela do console
            # Usar CREATE_NO_WINDOW é mais eficaz para evitar qualquer flash de console
            kwargs_subprocess['startupinfo'] = startupinfo
            kwargs_subprocess['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:
            # Para Unix-like, desanexar da sessão de controle pode ser útil
            # e redirecionar saídas se necessário, embora -m geralmente não crie um novo terminal.
            # kwargs_subprocess['start_new_session'] = True
            # Se ainda aparecer console, pode ser necessário:
            # kwargs_subprocess['stdout'] = subprocess.DEVNULL
            # kwargs_subprocess['stderr'] = subprocess.DEVNULL
            pass

        try:
            self.current_process = subprocess.Popen(command, cwd=self.base_dir, **kwargs_subprocess)
            print(f"Trackie iniciado com PID: {self.current_process.pid}")
            self.btn_start_trackie.setEnabled(False)
            self.btn_start_preview.setEnabled(False)
            self.btn_stop_trackie.setEnabled(True)
            # Opcional: Monitorar o processo para reabilitar botões quando terminar
            # self.check_process_timer = QTimer(self)
            # self.check_process_timer.timeout.connect(self.check_if_process_finished)
            # self.check_process_timer.start(1000) # Verificar a cada segundo
        except Exception as e:
            print(f"Erro ao iniciar Trackie: {e}")
            # Poderia mostrar um erro na GUI
            # self.statusBar().showMessage(f"Erro ao iniciar: {e}", 5000)
            self.reset_buttons()

    def check_if_process_finished(self):
        if self.current_process and self.current_process.poll() is not None: # Processo terminou
            print(f"Trackie (PID: {self.current_process.pid}) terminou.")
            self.current_process = None
            self.reset_buttons()
            if hasattr(self, 'check_process_timer'):
                self.check_process_timer.stop()
    
    def reset_buttons(self):
        self.btn_start_trackie.setEnabled(True)
        self.btn_start_preview.setEnabled(True)
        self.btn_stop_trackie.setEnabled(False)

    def start_trackie_normal(self):
        print("Botão 'Iniciar Trackie' clicado.")
        self._run_command(["--mode", "camera"])

    def start_trackie_preview(self):
        print("Botão 'Ver a Mente do Trackie' clicado.")
        self._run_command(["--mode", "camera", "--show_preview"])

    def stop_current_trackie(self):
        if self.current_process and self.current_process.poll() is None: # Se o processo existe e está rodando
            print(f"Tentando parar Trackie (PID: {self.current_process.pid})...")
            try:
                # Tenta terminar graciosamente primeiro
                self.current_process.terminate()
                try:
                    # Espera um pouco para o processo terminar
                    self.current_process.wait(timeout=5) # 5 segundos de timeout
                    print(f"Trackie (PID: {self.current_process.pid}) terminado.")
                except subprocess.TimeoutExpired:
                    print(f"Trackie (PID: {self.current_process.pid}) não terminou graciosamente, forçando...")
                    self.current_process.kill() # Força o término
                    self.current_process.wait() # Espera a finalização forçada
                    print(f"Trackie (PID: {self.current_process.pid}) forçadamente terminado.")
            except Exception as e:
                print(f"Erro ao tentar parar o processo {self.current_process.pid}: {e}")
            finally:
                self.current_process = None
                self.reset_buttons()
                if hasattr(self, 'check_process_timer'):
                    self.check_process_timer.stop()
        else:
            print("Nenhum processo Trackie ativo para parar.")
            self.reset_buttons() # Garante que os botões estejam no estado correto

    def closeEvent(self, event):
        # Garante que o processo filho seja terminado quando a GUI for fechada
        self.stop_current_trackie()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrackieApp()
    window.show()
    # Iniciar um timer para verificar o processo em segundo plano
    # Isso é útil se o processo backend puder terminar por conta própria
    # e queremos reabilitar os botões.
    timer = window.check_process_timer = QTimer() # Anexa o timer à janela para que não seja coletado pelo GC
    timer.timeout.connect(window.check_if_process_finished)
    timer.start(1000) # Verifica a cada 1000 ms (1 segundo)

    sys.exit(app.exec())
