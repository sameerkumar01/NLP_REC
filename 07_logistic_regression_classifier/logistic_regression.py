from __future__ import annotations

import numpy as np


class LogisticRegressionScratch:
    def __init__(self, learning_rate: float = 0.5, epochs: int = 300, l2: float = 0.001, tolerance: float = 1e-7):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.tolerance = tolerance
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def sigmoid(values):
        clipped = np.clip(values, -500, 500)
        return 1 / (1 + np.exp(-clipped))

    def loss(self, matrix, labels):
        probabilities = self.sigmoid(matrix @ self.weights + self.bias)
        probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
        cross_entropy = -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
        regularization = 0.5 * self.l2 * np.sum(self.weights ** 2)
        return cross_entropy + regularization

    def fit(self, matrix, labels):
        labels = np.asarray(labels, dtype=np.float64)
        self.weights = np.zeros(matrix.shape[1], dtype=np.float64)
        self.bias = 0.0
        previous_loss = float("inf")
        sample_count = max(len(labels), 1)
        for _ in range(self.epochs):
            probabilities = self.sigmoid(matrix @ self.weights + self.bias)
            errors = probabilities - labels
            weight_gradient = matrix.T @ errors / sample_count + self.l2 * self.weights
            bias_gradient = np.mean(errors)
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient
            current_loss = self.loss(matrix, labels)
            self.loss_history.append(current_loss)
            if abs(previous_loss - current_loss) < self.tolerance:
                break
            previous_loss = current_loss
        return self

    def predict_proba(self, matrix):
        return self.sigmoid(matrix @ self.weights + self.bias)

    def predict(self, matrix, threshold: float = 0.5):
        return (self.predict_proba(matrix) >= threshold).astype(np.int64)
