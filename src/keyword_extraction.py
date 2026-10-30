import re
from collections import Counter
from nltk.corpus import stopwords

from src.preprocessing import preprocess_text


def extract_frequency_keywords(text, top_n=10):
    """
    Extract keywords based on word frequency.

    Parameters:
        text (str): Input text.
        top_n (int): Maximum number of keywords to return.

    Returns:
        list: A list of (keyword, frequency) tuples.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    words = preprocess_text(text)

    word_counts = Counter(words)

    return word_counts.most_common(top_n)


def extract_keyphrases(text, top_n=10, ngram_range=(2, 3)):
    """
    Extract repeated two-word and three-word phrases without
    crossing sentence boundaries or including stopwords.
    """
    import re
    from collections import Counter
    from nltk.corpus import stopwords

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    if (
        not isinstance(ngram_range, tuple)
        or len(ngram_range) != 2
        or any(
            not isinstance(n, int) or isinstance(n, bool)
            for n in ngram_range
        )
        or ngram_range[0] < 2
        or ngram_range[1] < ngram_range[0]
    ):
        raise ValueError(
            "ngram_range must be a tuple such as (2, 3), "
            "with the minimum length at least 2."
        )

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    sentences = re.split(r"[.!?]+", text.lower())

    phrase_counts = Counter()

    for sentence in sentences:
        tokens = re.findall(r"[a-z]+", sentence)

        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                phrase_tokens = tokens[i:i + n]

                if any(word in stop_words for word in phrase_tokens):
                    continue

                if len(set(phrase_tokens)) < 2:
                    continue

                phrase = " ".join(phrase_tokens)
                phrase_counts[phrase] += 1

    return phrase_counts.most_common(top_n)