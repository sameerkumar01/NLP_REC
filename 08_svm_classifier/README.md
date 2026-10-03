# 8. Support Vector Machine for Text Classification from Scratch

This topic trains a linear Support Vector Machine on TF-IDF features using manually implemented hinge loss and gradient descent. It uses the same SMS dataset and deterministic split strategy as Topics 6 and 7 so Naive Bayes, logistic regression, and SVM results can be compared directly.

## Dataset

The download script uses Kaggle's SMS Spam Collection Dataset (`uciml/sms-spam-collection-dataset`). It contains 5572 English SMS messages, commonly with the `ham` or `spam` label in `v1` and message text in `v2`.

The default experiment processes up to 5000 messages, performs the same stratified 80/20 split used in Topics 6 and 7, and learns the TF-IDF vocabulary from training messages only.

## Linear SVM

Binary labels are represented as `-1` for ham and `+1` for spam. The linear decision score is:

`score(x) = w dot x + b`

The geometric margin is determined by the distance from points to the separating hyperplane. SVM training attempts to find a wide margin while penalizing examples that fall inside the margin or are misclassified.

The hinge loss for one example is:

`hinge(y, score) = max(0, 1 - y * score)`

The primal objective used here is:

`L = lambda * ||w||^2 / 2 + C * mean(max(0, 1 - y * (Xw + b)))`

For examples violating the margin, the gradients are:

`dL/dw = lambda * w - C * mean(y_i * x_i)`

`dL/db = -C * mean(y_i)`

For correctly classified examples outside the margin, only the regularization gradient remains.

## The kernel trick

A linear SVM computes a hyperplane directly in the TF-IDF feature space. A kernel SVM computes similarity through a kernel function without explicitly creating the transformed high-dimensional features.

The RBF kernel is:

`K(x,z) = exp(-gamma * ||x-z||^2)`

Efficient kernel SVM training normally uses dual optimization, support-vector storage, kernel caching, and algorithms such as SMO. The from-scratch implementation stays focused on the transparent linear primal objective. The library baseline includes both linear and RBF SVMs so their behavior can still be compared.

## Code map

- `text_utils.py` tokenizes SMS messages.
- `tfidf_vectorizer.py` creates normalized TF-IDF features manually.
- `linear_svm.py` implements hinge loss, gradients, learning-rate decay, and prediction.
- `metrics.py` implements accuracy, precision, recall, F1, and confusion counts.
- `from_scratch.py` loads, splits, trains, evaluates, and displays margins.
- `baseline_library.py` uses scikit-learn linear and RBF SVMs.
- `compare_results.py` evaluates scratch linear, library linear, and library RBF models.
- `data/download_data.py` downloads the Kaggle dataset.

## Run in Codespaces

```bash
cd /workspaces/NLP_REC/08_svm_classifier
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
python from_scratch.py --csv "data/spam.csv" --epochs 500 --learning-rate 0.1 --regularization 0.01 --c 1.0
```

## What to inspect

- Examples with `y * score >= 1` do not contribute hinge-loss gradients.
- Examples inside or across the margin directly change the separating hyperplane.
- Regularization controls the trade-off between margin width and training fit.
- TF-IDF text data is high-dimensional and often works well with a linear boundary.
- RBF SVMs can model nonlinear boundaries but are slower and more memory-intensive on large text datasets.
