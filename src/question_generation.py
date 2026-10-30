import re


def split_into_sentences(text):

    if not text or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip().split()) >= 5
    ]


def generate_questions(text, max_questions=10):

    if not isinstance(text, str):
        raise TypeError("Document text must be a string.")

    if not isinstance(max_questions, int) or isinstance(max_questions, bool):
        raise TypeError("max_questions must be an integer.")

    if max_questions < 1:
        raise ValueError("max_questions must be greater than zero.")

    sentences = split_into_sentences(text)

    results = []
    seen_questions = set()

    patterns = [
        (
            r"^(.{1,100}?)\s+is\s+(?:a|an|the)\s+(.+)$",
            lambda m: f"What is {m.group(1).strip()}?"
        ),
        (
            r"^(.{1,100}?)\s+refers to\s+(.+)$",
            lambda m: f"What does {m.group(1).strip()} refer to?"
        ),
        (
            r"^(.{1,100}?)\s+enables\s+(.+)$",
            lambda m: f"What does {m.group(1).strip()} enable?"
        ),
        (
            r"^(.{1,100}?)\s+measures\s+(.+)$",
            lambda m: f"What does {m.group(1).strip()} measure?"
        ),
        (
            r"^(.{1,100}?)\s+identifies\s+(.+)$",
            lambda m: f"What does {m.group(1).strip()} identify?"
        ),
        (
            r"^(.{1,100}?)\s+uses\s+(.+)$",
            lambda m: f"What does {m.group(1).strip()} use?"
        ),
        (
            r"^(.{1,100}?)\s+helps\s+(.+)$",
            lambda m: f"How does {m.group(1).strip()} help?"
        ),
        (
            r"^(.{1,100}?)\s+consists of\s+(.+)$",
            lambda m: f"What does {m.group(1).strip()} consist of?"
        ),
    ]

    for sentence in sentences:
        cleaned_sentence = re.sub(
            r"\s+",
            " ",
            sentence
        ).strip()

        for pattern, question_builder in patterns:
            match = re.match(
                pattern,
                cleaned_sentence,
                flags=re.IGNORECASE
            )

            if not match:
                continue

            question = question_builder(match)

            if question.lower() in seen_questions:
                break

            seen_questions.add(question.lower())

            results.append({
                "question": question,
                "answer_source": cleaned_sentence
            })

            break

        if len(results) >= max_questions:
            break

    return results