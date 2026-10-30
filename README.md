# Smart Keyword Extractor (SKE)

A Python-based text analysis application built with Streamlit to help users identify important keywords, analyze word frequency, explore topics, and extract meaningful information from text.

## Overview

Smart Keyword Extractor (SKE) is an NLP-based application designed to simplify text analysis. It provides an interactive interface for exploring text and identifying important information that can support reading, revision, and study.

## Features

- **Keyword Extraction:** Identify important keywords and keyphrases from text.
- **Word Frequency Analysis:** Examine frequently occurring words.
- **TF-IDF Analysis:** Identify terms based on their importance within a document.
- **TextRank:** Rank important terms or sentences using graph-based relationships.
- **Key Sentence Extraction:** Identify sentences that best represent the input text.
- **Topic Exploration:** Explore important themes in the text.
- **Revision Notes:** Generate concise study-oriented notes.
- **Question Generation:** Generate questions using rule-based patterns.
- **Download Options:** Access supported output and download features through the interface.
- **Interactive Dashboard:** Explore results through a Streamlit web application.

## Technologies Used

- Python
- Streamlit
- Natural Language Processing (NLP)
- NLTK
- Scikit-learn
- NumPy
- Pandas
- NetworkX
- Plotly

## Project Structure

```text
smart-keyword-extractor/
├── app.py
├── src/
├── tests/
├── style.css
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kaushikprijul/smart-keyword-extractor.git
cd smart-keyword-extractor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application should open in your browser.

## Testing

Run the automated tests with:

```bash
python -m pytest -v
```

## Purpose

The project demonstrates the application of NLP techniques and text-analysis methods through an interactive Python web application.

## Author

**Kaushik Prijul**

GitHub: [@kaushikprijul](https://github.com/kaushikprijul)

## License

No license has been specified yet.
