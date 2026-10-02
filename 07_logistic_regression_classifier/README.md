# 7. Logistic Regression for Text Classification from Scratch

This topic trains a binary spam classifier with manually implemented TF-IDF features, sigmoid probabilities, binary cross-entropy, gradients, and gradient descent. It uses the same SMS dataset and deterministic split strategy as Topic 6 so Naive Bayes and logistic regression can be compared directly.

## Dataset

The download script uses Kaggle's SMS Spam Collection Dataset (`uciml/sms-spam-collection-dataset`). It contains 5572 English SMS messages. The original CSV commonly stores the `ham` or `spam` label in `v1` and the message text in `v2`.

The vocabulary and IDF values are learned only from the training split. The default experiment processes up to 5000 messages, uses a deterministic stratified 80/20 split, and removes terms that appear in fewer than two training documents.

## TF-IDF

The implementation uses raw term count and smoothed inverse document frequency:

`tf(t,d) = count(t,d)`

`idf(t) = log((1 + N) / (1 + df(t))) + 1`

`tfidf(t,d) = tf(t,d) * idf(t)`

Each document vector is L2-normalized.

## Logistic regression

For feature vector `x`, weights `w`, and bias `b`:

`z = x dot w + b`

`sigmoid(z) = 1 / (1 + exp(-z))`

The binary cross-entropy loss is:

`L = -mean(y * log(p) + (1-y) * log(1-p))`

With L2 regularization:

`L_total = L + lambda * sum(w^2) / 2`

The gradients are:

`dL/dw = X.T dot (p-y) / m + lambda * w`

`dL/db = mean(p-y)`

Gradient descent updates the parameters using:

`w = w - learning_rate * dL/dw`

`b = b - learning_rate * dL/db`

## Code map

- `text_utils.py` tokenizes SMS messages.
- `tfidf_vectorizer.py` implements vocabulary, IDF, transformation, and normalization.
- `logistic_regression.py` implements sigmoid, loss, gradients, training, and prediction.
- `metrics.py` implements accuracy, precision, recall, F1, and confusion counts.
- `from_scratch.py` loads data, creates the split, trains the model, and reports metrics.
- `baseline_library.py` uses scikit-learn TF-IDF and logistic regression.
- `compare_results.py` evaluates both approaches on the same split.
- `data/download_data.py` downloads the Kaggle dataset.

## Run in Codespaces

```bash
cd /workspaces/NLP_REC/07_logistic_regression_classifier
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
python from_scratch.py --csv "data/spam.csv" --epochs 300 --learning-rate 0.5 --l2 0.001 --threshold 0.5
```

## What to inspect

- TF-IDF reduces the influence of words appearing in many messages.
- The sigmoid turns a linear score into a spam probability.
- Cross-entropy strongly penalizes confident incorrect predictions.
- L2 regularization discourages extremely large weights.
- Logistic regression learns a discriminative boundary, while Naive Bayes models class-conditional token probabilities.
