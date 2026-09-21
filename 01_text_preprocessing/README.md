# 1. Text Preprocessing Pipeline

This topic turns messy review text into a sequence of meaningful, normalized tokens. The point is not to make the fanciest cleaner; it is to make each transformation visible so you can inspect what information is removed or changed.

## Dataset

The download script targets Kaggle's **IMDB Dataset of 50K Movie Reviews** (`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`). It contains 50,000 labeled rows in `IMDB Dataset.csv` with two columns:

- `review`: HTML-containing movie review text
- `sentiment`: `positive` or `negative`

The dataset is used here for realistic input examples. Preprocessing itself is unsupervised, so labels are not consumed by the pipeline. By default the scripts process a small sample (`--limit 200`) to stay quick on a laptop.

### Kaggle credentials

You need Kaggle credentials only to download the data. Create an API token in Kaggle, place `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/macOS), or configure Kaggle environment variables. The script first tries `kagglehub` and then falls back to the Kaggle CLI. If you cannot configure credentials yet, all scripts run on built-in example sentences.

## What is implemented

1. **Regex cleaning** — unescape HTML, remove tags/URLs, normalize punctuation, and collapse whitespace.
2. **Tokenization** — a small word-and-number tokenizer that preserves contractions such as `isn't`.
3. **Stopword removal** — a documented built-in English stopword set; no hidden corpus download.
4. **Porter stemming** — the main Porter suffix rules (steps 1a–5) implemented with helper functions for vowel checks, consonant endings, and the measure `m`.
5. **Rule-based lemmatization** — transparent morphology rules plus a small irregular-word dictionary. This is deliberately not a dictionary-backed linguistic lemmatizer; the limitations are part of the lesson.

## Code map

- `regex_cleaner.py` — HTML/URL/punctuation cleanup
- `tokenizer.py` — token extraction
- `stopwords.py` — explicit stopword set
- `porter_stemmer.py` — Porter algorithm from scratch
- `lemmatizer.py` — explainable suffix and irregular rules
- `from_scratch.py` — pipeline orchestration and preprocessing statistics
- `baseline_library.py` — NLTK comparison implementation
- `compare_results.py` — timing and output comparison
- `data/download_data.py` — Kaggle download helper

## Run it

```bash
cd 01_text_preprocessing
pip install -r requirements.txt
python data/download_data.py
python from_scratch.py --csv "data/IMDB Dataset.csv" --limit 200
python compare_results.py --csv "data/IMDB Dataset.csv" --limit 50
```

Without a downloaded CSV, omit `--csv` and the scripts use a few built-in review-like sentences.

## What to look for

- Cleaning can remove markup while preserving the words inside it.
- Stopword removal changes token counts and can remove grammatical context.
- Stemming is aggressive (`studies` and `studying` may collapse to a rough stem).
- Rule-based lemmatization is more readable but less linguistically complete than a trained or dictionary-backed tool.
- The library baseline is usually faster or linguistically broader because it packages many edge cases; the scratch version is intentionally easier to trace.

## Metric report

The runner reports input characters, raw token count, cleaned token count, stopword-removal rate, vocabulary sizes, and average tokens per document. These are appropriate diagnostics for a preprocessing pipeline rather than a classification accuracy.
