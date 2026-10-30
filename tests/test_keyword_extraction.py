import pytest

from src.keyword_extraction import (
    extract_frequency_keywords,
    extract_keyphrases,
)


def test_frequency_keywords():
    text = "Machine learning is useful. Machine learning helps computers learn."

    result = extract_frequency_keywords(text, top_n=5)

    assert ("machine", 2) in result
    assert ("learning", 2) in result


def test_frequency_keywords_top_n():
    text = "Python Java Python C++ Java Python"

    result = extract_frequency_keywords(text, top_n=2)

    assert len(result) <= 2


def test_keyphrases():
    text = (
        "Machine learning is useful. "
        "Machine learning helps computers learn. "
        "Machine learning is widely used."
    )

    result = extract_keyphrases(text, top_n=10)

    assert ("machine learning", 3) in result


def test_keyphrases_do_not_cross_sentences():
    text = "Useful results. Machine learning is important."

    result = extract_keyphrases(text, top_n=10)

    phrases = [phrase for phrase, count in result]

    assert "results machine" not in phrases


def test_three_word_keyphrases():
    text = (
        "Natural language processing is useful. "
        "Natural language processing helps computers. "
        "Natural language processing is important."
    )

    result = extract_keyphrases(
        text,
        top_n=10,
        ngram_range=(3, 3),
    )

    assert ("natural language processing", 3) in result


def test_frequency_keywords_invalid_text():
    with pytest.raises(TypeError):
        extract_frequency_keywords(123)


def test_frequency_keywords_invalid_top_n():
    with pytest.raises(ValueError):
        extract_frequency_keywords("Python is useful", top_n=0)


def test_keyphrases_invalid_text():
    with pytest.raises(TypeError):
        extract_keyphrases(123)


def test_keyphrases_invalid_top_n():
    with pytest.raises(ValueError):
        extract_keyphrases("Machine learning is useful", top_n=0)