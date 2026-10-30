import pytest

from src.sentence_ranking import rank_sentences
from src.tfidf_extraction import extract_tfidf_keywords
from src.textrank_extraction import extract_textrank_keywords
from src.keyword_comparison import compare_keyword_methods


def test_sentence_ranking():
    text = (
        "Machine learning identifies patterns in data. "
        "The weather is sunny today."
    )

    result = rank_sentences(text, top_n=2)

    assert len(result) > 0
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], str)
    assert isinstance(result[0][1], (int, float))


def test_tfidf_extraction():
    documents = [
        "Machine learning identifies patterns in data.",
        "Cooking recipes use ingredients and cooking methods.",
        "Machine learning algorithms analyze data."
    ]

    result = extract_tfidf_keywords(
        documents,
        document_index=0,
        top_n=5
    )

    assert len(result) > 0
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], str)
    assert isinstance(result[0][1], (int, float))


def test_textrank_extraction():
    text = (
        "Machine learning algorithms identify patterns. "
        "Machine learning analyzes data."
    )

    result = extract_textrank_keywords(text, top_n=5)

    assert len(result) > 0
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], str)
    assert isinstance(result[0][1], (int, float))


def test_keyword_comparison():
    text = (
        "Machine learning algorithms identify patterns in data. "
        "Machine learning helps analyze data."
    )

    result = compare_keyword_methods(text, top_n=5)

    assert isinstance(result, dict)
    assert len(result) > 0


def test_sentence_ranking_invalid_text():
    with pytest.raises(TypeError):
        rank_sentences(123)


def test_textrank_invalid_text():
    with pytest.raises(TypeError):
        extract_textrank_keywords(123)