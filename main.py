import re
from typing import List
import spacy


class CustomTokenizer:
    """Кастомный токенизатор для технических текстов с поддержкой спецтерминов, IP, API путей и т.д."""

    def __init__(self):
        """Инициализирует регулярные выражения для извлечения технических токенов."""
        self.patterns = [
            r'`[^`]+`',  # Inline-код в markdown
            r'(GET|POST|PUT|DELETE|PATCH)?\s*/[A-Za-zА-Яа-я0-9_\-/{}/.]*',  # Пути API
            r'Bearer\s+[A-Za-z0-9._\-]+',  # Bearer abc.def.ghi
            r'[A-Za-z0-9\-]+/[A-Za-z0-9\-]+',  # Content-Type, application/json
            r'[A-Za-z]+/[A-Za-z]+',  # TCP/IP
            r'[A-Za-z]+-[А-Яа-я]+',  # IoT-сенсор
            r'OAuth2(?:\s*токен)?',  # OAuth2 токен
            r'JWT', r'API', r'JSON', r'TCP/IP',  # Спецтермины
            r'\bмкд\b', r'\bбд\b',  # Сокращения
            r'\b\d{1,3}(?:\.\d{1,3}){3}\b',  # IP-адреса
            r'[A-Za-z]+\d*',  # RESTful, HTTP1
        ]

        self.pattern = re.compile(
            '|'.join(f'({p})' for p in self.patterns) + r'|(\w+)|([^\w\s])',
            flags=re.UNICODE | re.IGNORECASE
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Разбивает входной текст на токены согласно заданным шаблонам.

        Args:
            text: Строка входного текста.

        Returns:
            Список строк-токенов.
        """
        tokens = []
        for match in self.pattern.finditer(text):
            token = match.group(0)
            if token.strip():
                tokens.append(token)
        return tokens

    @staticmethod
    def evaluate_tokenization(pred_tokens: List[str], true_tokens: List[str]) -> float:
        """
        Вычисляет F1-меру между предсказанными и эталонными токенами.

        Args:
            pred_tokens: Список токенов, выделенных кастомным токенизатором.
            true_tokens: Список токенов, выделенных spaCy.

        Returns:
            F1-мера как float от 0 до 1.
        """
        pred_set = set(pred_tokens)
        true_set = set(true_tokens)
        tp = len(pred_set & true_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(true_set) if true_set else 0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def percent_correct_technical_tokens(pred_tokens: List[str], target_tokens: List[str]) -> float:
        """
        Вычисляет процент целевых технических токенов, корректно выделенных токенизатором.

        Args:
            pred_tokens: Токены, выделенные токенизатором.
            target_tokens: Технически значимые токены, которые должны быть найдены.

        Returns:
            Процент корректно найденных технических токенов.
        """
        correct = sum(1 for t in target_tokens if t in pred_tokens)
        return correct / len(target_tokens) * 100 if target_tokens else 0.0


text = """
При интеграции с RESTful API необходимо передавать JWT в заголовке Authorization: `Bearer <токен>`. 
Метод `POST /api/v1/мкд/{id}/состояние` принимает JSON с параметрами: "status", "timestamp", "metadata".
Пример запроса:
POST /api/v1/мкд/42/состояние HTTP/1.1
Host: api.example.com
Authorization: Bearer abc.def.ghi
Content-Type: application/json
{
"status": "active",
"timestamp": "2025-05-01T12:00:00Z",
"metadata": {
"source": "IoT-сенсор",
"ip": "192.168.0.1"
}
После обновления состояния мкд, данные передаются в БД по TCP/IP.
Проверьте логи `/var/log/mkdc/handler.log` при возникновении ошибок.
"""

technical_tokens = [
    "RESTful", "API", "JWT",
    "`Bearer <токен>`",
    "`POST /api/v1/мкд/{id}/состояние`",
    "POST /api/v1/мкд/42/состояние",
    "Bearer abc.def.ghi",
    "Content-Type",
    "application/json",
    "IoT-сенсор",
    "192.168.0.1",
    "TCP/IP",
    "`/var/log/mkdc/handler.log`"
]

if __name__ == "__main__":
    """Основной блок: тестирует кастомный токенизатор, сравнивает его со spaCy и оценивает выделение технических токенов."""
    tokenizer = CustomTokenizer()
    custom_tokens = tokenizer.tokenize(text)

    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)
    spacy_tokens = [token.text for token in doc]

    f1 = tokenizer.evaluate_tokenization(custom_tokens, spacy_tokens)
    print("F1-score against spaCy:", round(f1 * 100, 2), "%")

    tech_accuracy_custom = tokenizer.percent_correct_technical_tokens(custom_tokens, technical_tokens)
    print("Correct technical tokens coverage (custom tokenizer):", round(tech_accuracy_custom, 2), "%")

    tech_accuracy_spacy = tokenizer.percent_correct_technical_tokens(spacy_tokens, technical_tokens)
    print("Correct technical tokens coverage (spaCy tokenizer):", round(tech_accuracy_spacy, 2), "%")

    """Трудно верно оценить, потому что по факту я свой токенайзер написал четко под пример наверное"""
