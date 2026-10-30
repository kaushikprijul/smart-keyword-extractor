import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
import re
import html
from pathlib import Path

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from src.keyword_comparison import compare_keyword_methods
from src.keyword_extraction import extract_keyphrases
from src.sentence_ranking import rank_sentences
from src.question_generation import generate_questions

st.set_page_config(
    page_title="SKE | Smart Keyword Extractor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = Path(__file__).parent / "style.css"
with css_path.open(encoding="utf-8") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def download_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

download_nltk_data()

stop_words = set(stopwords.words("english"))

with st.sidebar:

    st.html("""
    <div style="padding: 20px 8px 22px;">
        <div style="
            font-family: Georgia, serif;
            font-size: 2.35rem;
            letter-spacing: 9px;
            line-height: 1.2;
            color: #F6F1E9;">
            SKE
        </div>
        <div style="
            width: 42px;
            height: 1px;
            background: #B99B85;
            margin: 18px 0 14px;">
        </div>
        <div style="
            font-family: Arial, sans-serif;
            font-size: 0.66rem;
            letter-spacing: 2.2px;
            line-height: 2.05;
            color: #D7C4B7;">
            SMART<br>KEYWORD<br>EXTRACTOR
        </div>
    </div>
    """)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Keywords",
            "Word Frequency",
            "Topics",
            "Key Sentences",
            "Revision Notes",
            "Download",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.html(
        """
        <div style="
            margin-top: 34px;
            padding: 12px 5px;
            color: #D7C4B7;
            font-family: Georgia, serif;
            font-style: italic;
            font-size: 0.88rem;
            line-height: 1.8;">
            “Information is a competitive advantage.”
            <br><br>
            <span style="
                font-family: Arial, sans-serif;
                font-style: normal;
                font-size: 0.62rem;
                letter-spacing: 1.8px;">
                SKE · EST. 2026
            </span>
        </div>
        """
    )

st.markdown(
    '<div class="ske-eyebrow">READ · ANALYZE · UNDERSTAND · STUDY</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<h1>Smart Keyword <span style="color:#76252A;">Extractor</span></h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ske-subtitle">'
    'Turn academic text into clear keywords, potential topics, '
    'and focused revision notes.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

input_col, settings_col = st.columns([1.55, 1], gap="large", vertical_alignment="top")

with input_col:

    st.markdown("### 01. Your Document")

    st.caption(
        "Paste your text or upload a .txt file to get started."
    )

    input_method = st.radio(
        "Input method",
        ["Paste Text", "Upload .txt File"],
        horizontal=True,
        label_visibility="collapsed"
    )

    text = ""

    if input_method == "Paste Text":

        text = st.text_area(
            "Academic document",
            height=180,
            placeholder=(
                "Paste your lecture notes, research abstract, "
                "or academic document here..."
            ),
            label_visibility="collapsed"
        )

    else:

        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=["txt"]
        )

        if uploaded_file is not None:

            try:
                text = uploaded_file.getvalue().decode("utf-8")
                st.success("Document uploaded successfully.")

            except UnicodeDecodeError:
                st.error(
                    "Unable to read this file. "
                    "Please save it using UTF-8 encoding."
                )

    if text:
        word_count = len(text.split())

        st.caption(
            f"{word_count:,} words · "
            f"{len(text):,} characters"
        )

with settings_col:

    st.markdown("### 02. Analysis Settings")

    st.caption("Choose how many keywords to display.")

    num_keywords = st.slider(
        "Number of keywords",
        min_value=5,
        max_value=20,
        value=10
    )

    st.markdown("<br>", unsafe_allow_html=True)

    analyze_button = st.button(
        "✦  Analyze Document  →",
        use_container_width=True,
        type="primary"
    )

    st.markdown("""
    <div style="
        font-family: Arial, sans-serif;
        font-size: 0.78rem;
        color: #8A7A72;
        margin-top: 12px;
        line-height: 1.6;">
        Your text is processed locally by this Python application.
        This version does not send the document to an AI service.
    </div>
    """, unsafe_allow_html=True)

def analyze_document(document, limit):

    cleaned_text = document.lower()

    cleaned_text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        cleaned_text
    )

    words = word_tokenize(cleaned_text)

    filtered_words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    if not filtered_words:
        return None

    word_counts = Counter(filtered_words)

    phrases = []

    for i in range(len(words) - 1):

        first = words[i]
        second = words[i + 1]

        if (
            first not in stop_words
            and second not in stop_words
            and len(first) > 2
            and len(second) > 2
        ):
            phrases.append(first + " " + second)

    phrase_counts = Counter(phrases)

    keyword_scores = {}

    for word, count in word_counts.items():
        keyword_scores[word] = float(count)

    for phrase, count in phrase_counts.items():

        if count >= 2:
            keyword_scores[phrase] = count * 1.5

    ranked_keywords = sorted(
        keyword_scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    sentences = sent_tokenize(document)

    top_words = {
        word
        for word, count in word_counts.most_common(limit)
    }

    scored_sentences = []

    for sentence in sentences:

        sentence_words = re.findall(
            r"[a-zA-Z]+",
            sentence.lower()
        )

        score = sum(
            1 for word in sentence_words
            if word in top_words
        )

        if score > 0:
            scored_sentences.append((sentence.strip(), score))

    scored_sentences.sort(
        key=lambda item: (-item[1], item[0])
    )

    return {
        "word_counts": word_counts,
        "phrase_counts": phrase_counts,
        "ranked_keywords": ranked_keywords,
        "scored_sentences": scored_sentences,
        "word_total": len(document.split()),
        "unique_terms": len(set(filtered_words)),
        "sentences": sentences
    }

if analyze_button:

    if not text.strip():
        st.warning("Please enter text or upload a document.")

    else:

        with st.spinner("Analyzing your document..."):

            result = analyze_document(text, num_keywords)

        if result is None:
            st.warning(
                "No suitable words were found. "
                "Please try a longer academic document."
            )

        else:
            st.session_state["ske_result"] = result
            st.session_state["ske_document"] = text
            st.session_state["ske_limit"] = num_keywords

            st.success("Document analysis completed.")

result = st.session_state.get("ske_result")
limit = st.session_state.get("ske_limit", num_keywords)

if result is None:

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    st.html(
            """
            <div class="ske-card" style="
            text-align: center;
            padding: 30px 24px 24px;
            margin-top: 4px;">
            <div style="
                font-family: Georgia, 'Times New Roman', serif;
                font-size: clamp(1.35rem, 2.2vw, 1.7rem);
                line-height: 1.35;
                color: #35191A;
                margin-bottom: 12px;">
                Clarity begins with understanding.
            </div>
            <div style="
                max-width: 680px;
                margin: 0 auto;
                font-family: Arial, sans-serif;
                font-size: 0.9rem;
                line-height: 1.8;
                color: #766965;">
                Enter your academic text to discover important terms,
                potential topics, key sentences, and revision prompts.
            </div>
            <div style="
                margin-top: 20px;
                font-family: Arial, sans-serif;
                font-size: 0.66rem;
                letter-spacing: 1.8px;
                color: #9B8679;">
                READ &nbsp; · &nbsp; ANALYZE &nbsp; · &nbsp; REVISE
            </div>
        </div>
        """
    )

else:

    word_counts = result["word_counts"]
    phrase_counts = result["phrase_counts"]
    ranked_keywords = result["ranked_keywords"]
    scored_sentences = result["scored_sentences"]

    top_topics = [
        (phrase, count)
        for phrase, count in phrase_counts.most_common()
        if count >= 2
    ][:10]

    top_keywords = ranked_keywords[:limit]

    if page == "Overview":

        st.markdown("---")
        st.markdown("### Analysis Overview")

        c1, c2, c3, c4 = st.columns(4)

        metrics = [
            (
                c1,
                len(top_keywords),
                "KEYWORDS DISPLAYED"
            ),
            (
                c2,
                result["unique_terms"],
                "UNIQUE TERMS"
            ),
            (
                c3,
                len(top_topics),
                "POTENTIAL TOPICS"
            ),
            (
                c4,
                len(scored_sentences),
                "SENTENCES FOUND"
            )
        ]

        for col, value, label in metrics:

            with col:

                st.html(
                        f"""
                        <div class="ske-card">
                        <div class="ske-label">{label}</div>
                        <div class="ske-value">{value}</div>
                    </div>
                    """
                )

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### Top Keywords")

            st.caption("Ranked by the current frequency-based score.")

            for index, (keyword, score) in enumerate(
                top_keywords[:7],
                start=1
            ):

                safe_keyword = html.escape(keyword)

                st.html(
                    f"""
                    <div style="
                        display:flex;
                        align-items:center;
                        gap:10px;
                        padding:8px 0;
                        border-bottom:1px solid #E6DDD2;
                        font-family:Arial,sans-serif;">

                        <span style="
                            color:#9C8679;
                            font-size:0.75rem;">
                            {index:02}
                        </span>

                        <span style="
                            color:#342624;
                            flex:1;">
                            {safe_keyword}
                        </span>

                        <span style="
                            color:#76252A;
                            font-size:0.8rem;">
                            {score:.1f}
                        </span>
                    </div>
                    """
                )

        with right:
            st.markdown("### Potential Topics")

            st.caption("Repeated two-word phrases.")

            if top_topics:

                for phrase, count in top_topics[:7]:

                    st.markdown(
                        f"""
                        <div class="ske-topic">
                            <div>{html.escape(phrase.title())}</div>
                            <small>{count} occurrences</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.info(
                    "No repeated two-word phrases were found."
                )

        st.markdown("### Word Frequency")
        st.caption("Distribution of the eight most frequent terms.")

        frequency_data = dict(word_counts.most_common(8))

        if frequency_data:
            chart_data = pd.DataFrame({
                "Word": list(frequency_data.keys()),
                "Frequency": list(frequency_data.values()),
            })

            chart_data = chart_data.sort_values(
                "Frequency",
                ascending=True,
            )

            fig = px.bar(
                chart_data,
                x="Frequency",
                y="Word",
                orientation="h",
                text="Frequency",
                color_discrete_sequence=["#76252A"],
            )

            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Frequency: %{x}"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                template="plotly_white",
                height=360,
                autosize=True,
                showlegend=False,
                bargap=0.42,
                dragmode=False,
                margin=dict(l=12, r=42, t=20, b=38),
                font=dict(
                    family="Arial, sans-serif",
                    size=13,
                    color="#342624",
                ),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis=dict(
                    title=None,
                    showgrid=True,
                    gridcolor="#EEE9E3",
                    zeroline=False,
                    fixedrange=True,
                    rangemode="tozero",
                    tickfont=dict(
                        family="Arial, sans-serif",
                        size=12,
                        color="#342624",
                    ),
                ),
                yaxis=dict(
                    title=None,
                    showgrid=False,
                    ticks="",
                    fixedrange=True,
                    tickfont=dict(
                        family="Arial, sans-serif",
                        size=13,
                        color="#342624",
                    ),
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "scrollZoom": False,
                    "displayModeBar": False,
                    "doubleClick": False,
                    "responsive": False,
                },
            )

        else:
            st.info("No word frequency data is available.")

        st.markdown("---")
        sentence_col, question_col = st.columns([1, 1], gap="medium")

        with sentence_col:

            st.markdown("### Key Sentences to Review")

            for sentence, score in scored_sentences[:5]:

                st.markdown(
                    f"""
                    <div class="ske-question">
                        {html.escape(sentence)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if not scored_sentences:
                st.info("No key sentences were identified.")

        with question_col:

            st.markdown("### Revision Questions")

            st.caption(
                "Use these prompts to guide your studying."
            )

            for index, (keyword, score) in enumerate(
                top_keywords[:5],
                start=1
            ):

                safe_keyword = html.escape(keyword)

                st.markdown(
                    f"""
                    <div class="ske-question">
                        <b style="color:#76252A;">
                            {index:02}
                        </b>
                        &nbsp; Find the definition or explanation
                        of <b>{safe_keyword}</b> in your document.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    elif page == "Keywords":

        st.markdown("### Extracted Keywords")

        st.caption(
            "Scores are frequency-based heuristics, not "
            "probabilities or measures of semantic understanding."
        )

        keyword_rows = [
            {
                "Rank": index,
                "Keyword or Phrase": keyword,
                "Score": round(score, 2)
            }
            for index, (keyword, score) in enumerate(
                top_keywords,
                start=1
            )
        ]

        st.dataframe(
            keyword_rows,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.markdown("### 🔬 Compare Keyword Extraction Methods")

        st.caption(
            "Compare keywords identified by Frequency, TF-IDF, "
            "and TextRank using the same document."
        )

        if st.button(
            "Compare Methods",
            key="compare_keyword_methods_button"
        ):
            with st.spinner("Comparing keyword extraction methods..."):
                try:
                    comparison = compare_keyword_methods(
                        st.session_state["ske_document"],
                        top_n=limit
                    )

                    comparison_rows = []

                    for method, keywords in comparison.items():
                        for rank, (keyword, score) in enumerate(
                            keywords,
                            start=1
                        ):
                            comparison_rows.append({
                                "Method": method,
                                "Rank": rank,
                                "Keyword": keyword,
                                "Score": round(float(score), 4)
                            })

                    if comparison_rows:
                        st.dataframe(
                            comparison_rows,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info(
                            "No keywords were found by the selected methods."
                        )

                except Exception as error:
                    st.error(
                        f"Keyword comparison failed: {error}"
                    )

    elif page == "Word Frequency":

        st.markdown("### Word Frequency Analysis")

        st.caption(
            "See which individual words occur most often."
        )

        frequency_data = dict(
            word_counts.most_common(limit)
        )

        chart_data = pd.DataFrame(
            {
                "Word": list(frequency_data.keys()),
                "Frequency": list(frequency_data.values()),
            }
        ).set_index("Word")

        st.bar_chart(
            chart_data,
            color="#76252A",
            height=280,
            horizontal=True,
        )

        st.dataframe(
            [
                {
                    "Word": word,
                    "Occurrences": count
                }
                for word, count in word_counts.most_common(limit)
            ],
            use_container_width=True,
            hide_index=True
        )

    elif page == "Topics":

        st.markdown("### Potential Important Topics")

        st.caption(
            "Key phrases extracted using word sequences. "
            "These are candidate topics, not verified semantic groups."
        )

        try:
            topic_candidates = extract_keyphrases(
                st.session_state["ske_document"],
                top_n=15,
                ngram_range=(2, 3)
            )

            if topic_candidates:

                for index, (phrase, score) in enumerate(
                    topic_candidates,
                    start=1
                ):

                    st.html(
                        f"""
                        <div class="ske-card">
                            <div class="ske-label">
                                TOPIC CANDIDATE {index:02}
                            </div>

                            <div style="
                                font-family:Georgia,serif;
                                font-size:1.35rem;
                                color:#35191A;
                                margin:10px 0;">
                                {html.escape(phrase.title())}
                            </div>

                            <div class="ske-description">
                                Phrase frequency: {score}
                            </div>
                        </div>
                        """
                    )

            else:

                st.info(
                    "No key phrases were found. "
                    "Try analysing a longer document."
                )

        except Exception as error:

            st.error(
                f"Topic extraction failed: {error}"
            )

    elif page == "Key Sentences":

        st.markdown("### Sentences to Review")

        st.caption(
            "Sentences are ranked using the sentence-ranking "
            "algorithm to help identify important information."
        )

        try:
            ranked_sentences = rank_sentences(
                st.session_state["ske_document"],
                top_n=15
            )

            if ranked_sentences:

                for index, (sentence, score) in enumerate(
                    ranked_sentences,
                    start=1
                ):

                    st.html(
                        f"""
                        <div class="ske-card">
                            <div class="ske-label">
                                SENTENCE {index:02}
                            </div>

                            <div style="
                                font-family:Arial,sans-serif;
                                color:#403431;
                                line-height:1.7;
                                margin-top:10px;">
                                {html.escape(sentence)}
                            </div>

                            <div class="ske-description"
                                 style="margin-top:10px;">
                                Ranking Score: {score:.4f}
                            </div>
                        </div>
                        """
                    )

            else:

                st.info(
                    "No key sentences were found. "
                    "Try analysing a longer document."
                )

        except Exception as error:

            st.error(
                f"Sentence ranking failed: {error}"
            )

    elif page == "Revision Notes":

        st.markdown("### Revision Notes")

        st.caption(
            "Questions are generated from supported linguistic "
            "patterns in your document. Each question includes "
            "its original source sentence."
        )

        document_text = st.session_state.get(
            "ske_document",
            ""
        )

        try:
            revision_questions = generate_questions(
                document_text,
                max_questions=10
            )

            if revision_questions:

                for index, item in enumerate(
                    revision_questions,
                    start=1
                ):

                    question = html.escape(item["question"])
                    answer_source = html.escape(
                        item["answer_source"]
                    )

                    st.html(
                        f"""
                        <div class="ske-card">
                            <div class="ske-label">
                                REVISION QUESTION {index:02}
                            </div>

                            <div style="
                                font-family:Georgia,serif;
                                color:#35191A;
                                font-size:1.2rem;
                                line-height:1.6;
                                margin:12px 0;">
                                {question}
                            </div>

                            <div class="ske-question">
                                <b>Source sentence</b><br>
                                {answer_source}
                            </div>
                        </div>
                        """
                    )

            else:

                st.info(
                    "No suitable questions could be generated "
                    "from this document. Try a document containing "
                    "clear definitions and explanatory sentences."
                )

        except Exception as error:

            st.error(
                f"Question generation failed: {error}"
            )

    elif page == "Download":

        st.markdown("### Download Your Analysis")

        st.caption(
            "Export a detailed analysis report for your revision "
            "or project records."
        )

        document_text = st.session_state.get("ske_document", "")

        report = (
            "SMART KEYWORD EXTRACTOR\n"
            "ACADEMIC DOCUMENT ANALYSIS REPORT\n"
            + "=" * 45
            + "\n\n"
        )

        report += "DOCUMENT OVERVIEW\n"
        report += "-" * 30 + "\n"

        report += f"Total words: {len(document_text.split())}\n"
        report += f"Unique processed terms: {len(word_counts)}\n"
        report += f"Number of sentences: {len(scored_sentences)}\n\n"

        report += "TOP KEYWORDS\n"
        report += "-" * 30 + "\n"

        for keyword, score in top_keywords:
            report += f"{keyword}: {score:.2f}\n"

        report += "\nWORD FREQUENCY\n"
        report += "-" * 30 + "\n"

        for word, count in word_counts.most_common(limit):
            report += f"{word}: {count}\n"

        report += "\nPOTENTIAL TOPICS\n"
        report += "-" * 30 + "\n"

        for phrase, count in top_topics:
            report += f"{phrase}: {count} occurrences\n"

        report += "\nKEY SENTENCES TO REVIEW\n"
        report += "-" * 30 + "\n"

        try:
            ranked_sentences = rank_sentences(
                document_text,
                top_n=15
            )

            if ranked_sentences:
                for index, (sentence, score) in enumerate(
                    ranked_sentences, start=1
                ):
                    report += (
                        f"{index}. {sentence}\n"
                        f"   Ranking score: {score:.4f}\n\n"
                    )
            else:
                report += "No key sentences were found.\n"

        except Exception as error:
            report += f"Sentence ranking unavailable: {error}\n"

        report += "\nKEYWORD EXTRACTION METHOD COMPARISON\n"
        report += "-" * 40 + "\n"

        try:
            comparison = compare_keyword_methods(
                document_text,
                top_n=limit
            )

            for method, keywords in comparison.items():
                report += f"\n{method}\n"

                if keywords:
                    for keyword, score in keywords:
                        report += f"  {keyword}: {float(score):.4f}\n"
                else:
                    report += "  No keywords found.\n"

        except Exception as error:
            report += f"Comparison unavailable: {error}\n"

        report += "\nGENERATED REVISION QUESTIONS\n"
        report += "-" * 35 + "\n"

        try:
            revision_questions = generate_questions(
                document_text,
                max_questions=10
            )

            if revision_questions:
                for index, item in enumerate(
                    revision_questions, start=1
                ):
                    report += f"\nQuestion {index}: "
                    report += item["question"] + "\n"
                    report += (
                        "Source sentence: "
                        + item["answer_source"]
                        + "\n"
                    )
            else:
                report += (
                    "No suitable questions could be generated "
                    "from this document.\n"
                )

        except Exception as error:
            report += f"Question generation unavailable: {error}\n"

        st.download_button(
            label="↓  Download Analysis Report",
            data=report,
            file_name="SKE_Analysis_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.info(
            "Your report includes document statistics, keywords, "
            "word frequency, potential topics, ranked sentences, "
            "keyword method comparisons, and generated revision "
            "questions with their source sentences."
        )

    elif page == "About":

        st.markdown("### About SKE")

        st.html(
            f"""
            <div class="ske-card">
            <div class="ske-eyebrow">
                PROJECT OVERVIEW
            </div>

            <h2 style="margin-top:15px;">
                Smart Keyword Extractor
            </h2>

            <p style="line-height:1.8;">
                SKE is a small academic text-analysis application
                designed to help students identify frequently
                occurring terms, explore potential topics, and
                prepare revision prompts from their notes.
            </p>

            <hr>

            <h3>Technologies</h3>

            <p>
                Python · Streamlit · NLTK
            </p>

            <h3>Natural Language Processing Techniques</h3>

            <p>
                Tokenization · Stopword Removal · Word Frequency ·
                Phrase Extraction · Sentence Ranking
            </p>

            <h3>Important limitation</h3>

            <p style="line-height:1.8;">
                The current application uses frequency-based
                heuristics. It does not fully understand document
                meaning, verify concepts, or generate factual answers.
                Repeated phrases are only candidate topics.
            </p>
        </div>
        """)

st.markdown("""
<div class="ske-footer">
    S K E &nbsp; · &nbsp; SMART KEYWORD EXTRACTOR
    <br><br>
    READ WITH INTENTION.
</div>
""", unsafe_allow_html=True)
