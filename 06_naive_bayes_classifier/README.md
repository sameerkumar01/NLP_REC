# 6. Multinomial Naive Bayes Text Classifier from Scratch

This topic builds a complete spam classifier using manually created count features and a Multinomial Naive Bayes model. The implementation exposes class priors, token likelihoods, Laplace smoothing, log-space prediction, and classification metrics.

## Dataset

The download script uses Kaggle's SMS Spam Collection Dataset (`uciml/sms-spam-collection-dataset`). It contains 5572 English SMS messages. The original CSV commonly uses `v1` for the `ham` or `spam` label and `v2` for the message text.

The default command processes up to 5000 messages. A deterministic stratified split keeps both classes represented in training and test data. The vocabulary is learned from training messages only.

## Multinomial Naive Bayes

For class `c` and document `d`:

`P(c | d) proportional to P(c) * product(P(word | c) ^ count(word,d))`

Products of small probabilities can underflow, so prediction uses logs:

`log P(c | d) = log P(c) + sum(count(word,d) * log P(word | c))`

The class prior is:

`P(c) = documents_in_class_c / total_documents`

With Laplace smoothing:

`P(word | c) = (count(word,c) + alpha) / (total_tokens_in_c + alpha * vocabulary_size)`

The predicted class is the class with the greatest log score.

## Metrics

The project reports accuracy, spam precision, spam recall, spam F1, and the confusion matrix. These metrics matter because the dataset contains more ham messages than spam messages.

## Code map

- `text_utils.py` tokenizes SMS messages.
- `vectorizer.py` creates the training vocabulary and count matrix manually.
- `naive_bayes.py` implements Multinomial Naive Bayes in NumPy.
- `metrics.py` implements classification metrics from predicted and true labels.
- `from_scratch.py` loads, splits, trains, evaluates, and prints example predictions.
- `baseline_library.py` uses scikit-learn's `CountVectorizer` and `MultinomialNB`.
- `compare_results.py` evaluates both implementations on the same split.
- `data/download_data.py` downloads the Kaggle dataset.

## Run in Codespaces

```bash
cd /workspaces/NLP_REC/06_naive_bayes_classifier
python -m pip install -r requirements.txt
python from_scratch.py
python compare_results.py
```

Download and use the real dataset:

```bash
python data/download_data.py
python from_scratch.py --csv "data/spam.csv" --limit 5000
python compare_results.py --csv "data/spam.csv" --limit 5000
```

Useful options:

```bash
python from_scratch.py --csv "data/spam.csv" --alpha 1.0 --min-df 2 --test-size 0.2
```

## What to inspect

- The model adds token evidence to the log prior for each class.
- Laplace smoothing prevents unseen class-token pairs from receiving zero probability.
- Spam precision measures how many predicted spam messages are actually spam.
- Spam recall measures how many real spam messages are found.
- The conditional-independence assumption is unrealistic but often effective for count-based text classification.
