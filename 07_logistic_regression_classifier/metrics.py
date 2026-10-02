from __future__ import annotations


def classification_metrics(y_true, y_pred):
    pairs = list(zip(y_true, y_pred))
    true_positive = sum(actual == 1 and predicted == 1 for actual, predicted in pairs)
    true_negative = sum(actual == 0 and predicted == 0 for actual, predicted in pairs)
    false_positive = sum(actual == 0 and predicted == 1 for actual, predicted in pairs)
    false_negative = sum(actual == 1 and predicted == 0 for actual, predicted in pairs)
    total = len(pairs)
    accuracy = (true_positive + true_negative) / max(total, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
