from src.keyword_extraction import extract_frequency_keywords
from src.tfidf_extraction import extract_tfidf_keywords
from src.textrank_extraction import extract_textrank_keywords


def compare_keyword_methods(text, documents=None, top_n=10):
    """
    Compare Frequency, TF-IDF, and TextRank keyword extraction.

    Parameters:
        text (str): Document to analyze.
        documents (list[str] | None): Collection used by TF-IDF.
        top_n (int): Number of keywords to return per method.

    Returns:
        dict: Keywords produced by each method.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    if documents is None:
        documents = [text]
    elif not isinstance(documents, (list, tuple)):
        raise TypeError("documents must be a list or tuple of strings.")

    if not all(isinstance(doc, str) for doc in documents):
        raise TypeError("Every document must be a string.")

    if not documents:
        raise ValueError("documents must not be empty.")

    frequency_keywords = extract_frequency_keywords(text, top_n)

    if text not in documents:
        tfidf_documents = [text, *documents]
        target_index = 0
    else:
        tfidf_documents = list(documents)
        target_index = tfidf_documents.index(text)

    tfidf_keywords = extract_tfidf_keywords(
        tfidf_documents,
        document_index=target_index,
        top_n=top_n
    )

    textrank_keywords = extract_textrank_keywords(text, top_n)

    return {
        "Frequency": frequency_keywords,
        "TF-IDF": tfidf_keywords,
        "TextRank": textrank_keywords
    }