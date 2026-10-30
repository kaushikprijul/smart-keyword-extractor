from collections import Counter
import re
import networkx as nx
from nltk.corpus import stopwords


def extract_textrank_keywords(text, top_n=10, window_size=4):
    """
    Extract keywords using a TextRank-style word co-occurrence graph.

    Words that frequently occur near other important words
    contribute to the graph-based ranking.

    Parameters:
        text (str): Input document.
        top_n (int): Maximum number of keywords to return.
        window_size (int): Number of nearby tokens used to connect words.

    Returns:
        list: A list of (keyword, score) tuples.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size < 2
    ):
        raise ValueError("window_size must be an integer of at least 2.")

    tokens = re.findall(r"[a-z]+", text.lower())

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    tokens = [word for word in tokens if word not in stop_words]

    if not tokens:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(set(tokens))

    for i, word in enumerate(tokens):
        for j in range(i + 1, min(i + window_size, len(tokens))):
            neighbour = tokens[j]

            if word != neighbour:
                if graph.has_edge(word, neighbour):
                    graph[word][neighbour]["weight"] += 1
                else:
                    graph.add_edge(word, neighbour, weight=1)

    if graph.number_of_edges() == 0:
        counts = Counter(tokens)
        return [
            (word, float(count))
            for word, count in counts.most_common(top_n)
        ]

    scores = nx.pagerank(graph, weight="weight")

    ranked_words = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    return [
        (word, round(float(score), 4))
        for word, score in ranked_words[:top_n]
    ]