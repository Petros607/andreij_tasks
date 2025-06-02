import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QComboBox, QLabel,
    QMenuBar, QTextEdit, QStatusBar, QSplitter, QSizePolicy,
    QTextBrowser, QStackedWidget, QHBoxLayout, QVBoxLayout, QLineEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from pyollama import run_model, convert_nanosec_to_sec, list_models_names


class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama Chat")
        self.setMinimumSize(500, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.model_selector = QComboBox()
        self.model_selector.addItems(list_models_names())
        layout.addWidget(QLabel("Выберите модель:"))
        layout.addWidget(self.model_selector)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Arial", 12))
        layout.addWidget(self.chat_history)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите сообщение...")
        self.input_field.setFont(QFont("Arial", 12))
        self.send_button = QPushButton("Отправить")
        self.input_field.setFont(QFont("Arial", 12))
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)

        layout.addLayout(input_layout)
        self.setLayout(layout)

    def send_message(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return

        model_name = self.model_selector.currentText()

        self.chat_history.append(f"[Пользователь]: {user_input}")
        self.input_field.clear()

        try:
            result = run_model(model_name, user_input)
            response = result.get('response', '[Нет ответа от модели]')
        except Exception as e:
            response = f"[Ошибка]: Не удалось получить ответ от модели. Проверьте, запущен ли Ollama.\n{e}"

        self.chat_history.append(f"[Модель]: {response}\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec())
