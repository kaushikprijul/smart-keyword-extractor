from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf_keywords(documents, document_index=0, top_n=10):
    """
    Extract keywords from one document using TF-IDF.

    Parameters:
        documents (list[str]): Collection of documents.
        document_index (int): Index of the document to analyze.
        top_n (int): Maximum number of keywords to return.

    Returns:
        list: A list of (keyword, score) tuples.
    """
    if isinstance(documents, str):
        raise TypeError("documents must be a list of strings.")

    if not isinstance(documents, (list, tuple)) or not documents:
        raise ValueError("Provide a non-empty list of documents.")

    if not all(isinstance(doc, str) for doc in documents):
        raise TypeError("Every document must be a string.")

    if (
        not isinstance(document_index, int)
        or isinstance(document_index, bool)
        or not 0 <= document_index < len(documents)
    ):
        raise ValueError("document_index is outside the document collection.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf_matrix[document_index].toarray().flatten()

    ranked_indices = scores.argsort()[::-1]

    keywords = [
        (feature_names[index], round(float(scores[index]), 4))
        for index in ranked_indices
        if scores[index] > 0
    ][:top_n]

    return keywords