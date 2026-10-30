import re
from collections import Counter
from src.preprocessing import preprocess_text


def rank_sentences(text, top_n=5):
    """
    Rank sentences according to the frequency of their
    meaningful words.

    Parameters:
        text (str): Input document.
        top_n (int): Maximum number of sentences to return.

    Returns:
        list: A list of (sentence, score) tuples.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:
        return []

    words = preprocess_text(text)

    word_frequencies = Counter(words)

    scored_sentences = []

    for index, sentence in enumerate(sentences):
        sentence_words = preprocess_text(sentence)

        if not sentence_words:
            score = 0.0
        else:

            score = sum(
                word_frequencies[word] for word in sentence_words
            ) / len(sentence_words)

        scored_sentences.append((index, sentence, score))

    scored_sentences.sort(key=lambda item: (-item[2], item[0]))

    return [
        (sentence, round(score, 3))
        for _, sentence, score in scored_sentences[:top_n]
    ]