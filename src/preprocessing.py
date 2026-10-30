import re
from nltk.corpus import stopwords


FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were",
    "be", "been", "being", "and", "or", "but", "if",
    "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "to", "from", "in", "on", "out", "over", "under",
    "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "them",
    "my", "your", "our", "their", "me", "him", "her",
    "as", "do", "does", "did", "doing", "have", "has", "had",
    "can", "could", "will", "would", "should", "may", "might",
    "not", "no", "so", "than", "too", "very"
}


def clean_text(text):
    """
    Convert text to lowercase and remove punctuation,
    numbers, and unnecessary whitespace.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.lower()

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = text.replace("_", " ")

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    """
    Split cleaned text into individual words.
    """
    cleaned_text = clean_text(text)

    tokens = re.findall(r"[^\W\d_]+", cleaned_text, flags=re.UNICODE)

    return tokens


def remove_stopwords(tokens):
    """
    Remove common English words that usually carry
    little keyword-related information.
    """
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = FALLBACK_STOPWORDS

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return filtered_tokens


def preprocess_text(text, remove_common_words=True):
    """
    Complete preprocessing pipeline:
    clean text -> tokenize -> optionally remove stopwords.
    """
    tokens = tokenize_text(text)

    if remove_common_words:
        tokens = remove_stopwords(tokens)

    return tokens