# 5. Statistical N-gram Language Model from Scratch

This topic learns the probability of a word from the words immediately before it. The implementation supports unigram, bigram, and trigram models, reports perplexity, generates text, and compares Laplace smoothing with interpolated Kneser-Ney smoothing.

## Dataset

The download script uses Kaggle's News Category Dataset (`rmisra/news-category-dataset`). It contains approximately 210,000 JSONL news records with fields including `headline`, `short_description`, `category`, `link`, `authors`, and `date`.

The experiment joins each headline and short description, splits the text into sentences, and uses the first 5000 records by default. Rare words are mapped to `<UNK>` using `--min-count` so evaluation can score unseen words. A fixed random seed creates an 80/20 train/test split.

## N-gram probability

For an order-`n` model:

`P(w_i | w_(i-n+1), ..., w_(i-1))`

The maximum-likelihood estimate is:

`count(history, word) / count(history)`

## Laplace smoothing

Add-alpha smoothing assigns probability to unseen n-grams:

`P_Laplace(word | history) = (count(history, word) + alpha) / (count(history) + alpha * |V|)`

## Interpolated Kneser-Ney smoothing

The discounted higher-order probability is:

`max(count(history, word) - D, 0) / count(history)`

The remaining probability mass is:

`lambda(history) = D * unique_followers(history) / count(history)`

It is combined recursively with a lower-order model. The unigram base uses continuation probability:

`P_cont(word) = unique_predecessors(word) / unique_bigram_types`

## Perplexity

For `M` predicted tokens:

`perplexity = exp(-sum(log P(word | history)) / M)`

Lower perplexity means the model assigns higher probability to the held-out text.

## Code map

- `tokenization.py` turns article text into tokenized sentences.
- `counts.py` builds n-gram, context, follower, and predecessor counts.
- `laplace.py` implements add-alpha smoothing.
- `kneser_ney.py` implements recursive interpolated Kneser-Ney smoothing.
- `language_model.py` evaluates perplexity and generates text.
- `from_scratch.py` trains and evaluates the scratch model.
- `baseline_library.py` uses NLTK language-model classes.
- `compare_results.py` compares scratch and NLTK perplexity.
- `data/download_data.py` downloads the Kaggle dataset.

## Run in Codespaces

```bash
cd /workspaces/NLP_REC/05_ngram_language_model
python -m pip install -r requirements.txt
python from_scratch.py
python compare_results.py
```

Download and use the real dataset:

```bash
python data/download_data.py
python from_scratch.py --json "data/News_Category_Dataset_v3.json" --limit 5000 --order 3
python compare_results.py --json "data/News_Category_Dataset_v3.json" --limit 2000 --order 3
```

Useful options:

```bash
python from_scratch.py --order 2 --alpha 1.0 --discount 0.75 --generate-length 30
```

## What to inspect

- Higher-order models use more context but produce more unseen n-grams.
- Laplace smoothing is simple but spreads probability across the entire vocabulary.
- Kneser-Ney uses continuation counts, so words appearing after many different contexts receive stronger lower-order probability.
- Perplexity should be compared only when tokenization, vocabulary, split, and prediction set are consistent.
- Generated text from a small statistical model is locally plausible but lacks long-range coherence.
