from __future__ import annotations

import numpy as np


class MultinomialNaiveBayesScratch:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes = None
        self.log_class_prior = None
        self.log_feature_probability = None

    def fit(self, matrix, labels):
        labels = np.asarray(labels)
        self.classes, class_counts = np.unique(labels, return_counts=True)
        self.log_class_prior = np.log(class_counts / len(labels))
        feature_probabilities = []
        for label in self.classes:
            feature_counts = matrix[labels == label].sum(axis=0).astype(np.float64)
            smoothed_counts = feature_counts + self.alpha
            probabilities = smoothed_counts / smoothed_counts.sum()
            feature_probabilities.append(np.log(probabilities))
        self.log_feature_probability = np.vstack(feature_probabilities)
        return self

    def predict_log_scores(self, matrix):
        return matrix @ self.log_feature_probability.T + self.log_class_prior

    def predict(self, matrix):
        scores = self.predict_log_scores(matrix)
        return self.classes[np.argmax(scores, axis=1)]
