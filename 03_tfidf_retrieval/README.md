# 3. TF-IDF and Document Retrieval from Scratch

This topic turns documents into weighted vectors and retrieves the documents most similar to a text query. The implementation exposes vocabulary construction, document frequency, inverse document frequency, vector normalization, and cosine similarity without using a vectorizer.

## Dataset

The download script targets Kaggle's News Category Dataset (`rmisra/news-category-dataset`). It contains approximately 210,000 news records stored as JSON Lines. Important fields include `headline`, `short_description`, `category`, `link`, `authors`, and `date`.

The experiment combines `headline` and `short_description` into one searchable document. The default run uses the first 5000 records so it remains comfortable in Codespaces. Use `--limit 0` to process every available record.

Kaggle credentials are required only for downloading the data. Place `kaggle.json` at `~/.kaggle/kaggle.json`, configure Kaggle environment variables, or run the built-in examples.

## Mathematics

For a term `t` in document `d`, this project uses raw term frequency:

`tf(t,d) = count(t,d)`

The smoothed inverse document frequency is:

`idf(t) = log((1 + N) / (1 + df(t))) + 1`

The unnormalized TF-IDF weight is:

`tfidf(t,d) = tf(t,d) * idf(t)`

Each document vector is L2-normalized. A query is transformed using the same vocabulary and weights. Cosine similarity is:

`cosine(q,d) = (q dot d) / (||q|| * ||d||)`

With normalized vectors, retrieval is a matrix-vector dot product.

## Code map

- `text_utils.py` tokenizes the news text consistently for scratch and baseline experiments.
- `tfidf.py` builds the vocabulary, document frequencies, IDF vector, TF-IDF matrix, and cosine scores.
- `retrieval.py` converts a query into a vector and ranks documents.
- `from_scratch.py` loads data, builds the index, prints metrics, and displays top matches.
- `baseline_library.py` uses scikit-learn's `TfidfVectorizer` for comparison.
- `compare_results.py` compares vocabulary, matrix shape, runtime, scores, and top results.
- `data/download_data.py` downloads the Kaggle JSONL dataset.

## Run in Codespaces

```bash
cd /workspaces/NLP_REC/03_tfidf_retrieval
python -m pip install -r requirements.txt
python from_scratch.py
python compare_results.py
```

After downloading the real dataset:

```bash
python data/download_data.py
python from_scratch.py --json "data/News_Category_Dataset_v3.json" --limit 5000
python compare_results.py --json "data/News_Category_Dataset_v3.json" --limit 1000
```

Use a custom query:

```bash
python from_scratch.py --json "data/News_Category_Dataset_v3.json" --limit 5000 --query "climate change renewable energy policy"
```

## What to inspect

- Common words have lower IDF because they appear in many documents.
- A query containing rare, topical words can retrieve more focused results.
- Cosine similarity ignores document length after normalization.
- The scratch implementation is intentionally explicit and can be slower than scikit-learn's optimized sparse implementation.
- The baseline may differ slightly when the dataset contains punctuation or Unicode tokenization edge cases.
