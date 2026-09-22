# 2. Bag of Words and N-grams from Scratch

This topic converts documents into count-based representations. The implementation makes vocabulary construction, document-term matrices, and contiguous n-gram counting explicit.

## Dataset

The download script targets Kaggle's IMDB Dataset of 50K Movie Reviews (`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`). It contains 50,000 rows with `review` and `sentiment` columns. The default experiments use 500 reviews so the output remains easy to inspect and runs quickly in Codespaces.

Kaggle credentials are required only for downloading the CSV. Place `kaggle.json` at `~/.kaggle/kaggle.json`, configure Kaggle environment variables, or use the built-in example documents.

## Core ideas

A vocabulary maps each distinct token to an integer column. A Bag-of-Words matrix stores how many times vocabulary token `j` occurs in document `i`:

`X[i, j] = count(token_j in document_i)`

An n-gram is a consecutive sequence of `n` tokens. Unigrams contain one token, bigrams contain two, and trigrams contain three. The implementation preserves repeated n-grams, so it can represent frequency.

## Code map

- `ngrams.py` generates contiguous n-grams.
- `bag_of_words.py` builds a vocabulary and dense count matrix.
- `from_scratch.py` loads reviews, builds unigram/bigram/trigram counts, and prints diagnostics.
- `baseline_library.py` uses scikit-learn's `CountVectorizer` for comparison.
- `compare_results.py` compares vocabulary size, matrix counts, sparsity, and runtime.
- `data/download_data.py` downloads the Kaggle dataset.

## Run in Codespaces

```bash
cd /workspaces/NLP_REC/02_bag_of_words_and_ngrams
python -m pip install -r requirements.txt
python from_scratch.py
python compare_results.py
```

After downloading the dataset:

```bash
python data/download_data.py
python from_scratch.py --csv "data/IMDB Dataset.csv" --limit 500
python compare_results.py --csv "data/IMDB Dataset.csv" --limit 100
```

## What to inspect

- How vocabulary order affects matrix columns.
- Why a dense matrix becomes mostly zero as vocabulary grows.
- How bigrams capture short phrases such as `not good` that unigrams separate.
- Why the library baseline is usually faster and more feature-rich.
- Whether the from-scratch and library unigram counts agree on the same tokenization.
