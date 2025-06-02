import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import string
import random


def download_nltk_resources():
    """Загружает необходимые ресурсы NLTK."""
    resources = [
        'movie_reviews',
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    for resource in resources:
        nltk.download(resource)


def preprocess_text(text, lemmatizer, stop_words, punctuation):
    """
    Предварительная обработка текста: токенизация, удаление стоп-слов,
    пунктуации и лемматизация.
    """
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        token for token in tokens
        if token not in stop_words and token not in punctuation
    ]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)


def load_and_preprocess_data():
    """
    Загружает и предварительно обрабатывает данные.
    """
    documents = [
        (movie_reviews.raw(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    random.shuffle(documents)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    texts = [
        preprocess_text(text, lemmatizer, stop_words, punctuation)
        for text, _ in documents
    ]
    labels = [1 if label == 'pos' else 0 for _, label in documents]
    
    return documents, texts, labels


def train_and_evaluate_model(texts, labels):
    """
    Обучает модель и оценивает её точность.
    """
    vectorizer = CountVectorizer(max_features=10000)
    X = vectorizer.fit_transform(texts)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=4000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Точность модели на тестовой выборке: {accuracy:.4f}")
    
    return model, X_test, y_test


def show_prediction_examples(model, X_test, y_test, documents, num_examples=5):
    """
    Выводит примеры предсказаний модели.
    """
    print("\nПримеры прогнозов:")
    test_indices = train_test_split(range(len(documents)), test_size=0.2, random_state=42)[1]
    y_pred = model.predict(X_test)
    
    for i in range(num_examples):
        idx = random.randint(0, X_test.shape[0] - 1)
        original_idx = test_indices[idx]
        original_text, original_label = documents[original_idx]
        snippet = original_text[:200] + "..." if len(original_text) > 200 else original_text
        
        prediction = "благоприятная" if y_pred[idx] == 1 else "негативная"
        actual = "благоприятная" if y_test[idx] == 1 else "негативная"
        
        print(f"\nПример {i + 1}:")
        print(f"Текст: {snippet}")
        print(f"Прогноз: {prediction}")
        print(f"Фактическая оценка: {actual}")
        print(f"Совпадение: {'✓' if y_pred[idx] == y_test[idx] else '✗'}")


def main():
    # download_nltk_resources()
    
    documents, texts, labels = load_and_preprocess_data()
    model, X_test, y_test = train_and_evaluate_model(texts, labels)
    
    show_prediction_examples(model, X_test, y_test, documents)


if __name__ == "__main__":
    main()
