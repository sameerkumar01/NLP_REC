from __future__ import annotations

import numpy as np


class LinearSVMScratch:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 500, regularization: float = 0.01, c: float = 1.0, decay: float = 0.001, tolerance: float = 1e-7):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.c = c
        self.decay = decay
        self.tolerance = tolerance
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def decision_function(self, matrix):
        return matrix @ self.weights + self.bias

    def loss(self, matrix, labels):
        margins = labels * self.decision_function(matrix)
        hinge = np.maximum(0, 1 - margins)
        return 0.5 * self.regularization * np.sum(self.weights ** 2) + self.c * np.mean(hinge)

    def fit(self, matrix, labels):
        labels = np.asarray(labels, dtype=np.float64)
        self.weights = np.zeros(matrix.shape[1], dtype=np.float64)
        self.bias = 0.0
        previous_loss = float("inf")
        sample_count = max(len(labels), 1)
        for epoch in range(self.epochs):
            margins = labels * self.decision_function(matrix)
            active = margins < 1
            weight_gradient = self.regularization * self.weights
            bias_gradient = 0.0
            if np.any(active):
                weight_gradient -= self.c * (matrix[active].T @ labels[active]) / sample_count
                bias_gradient = -self.c * np.sum(labels[active]) / sample_count
            current_rate = self.learning_rate / (1 + self.decay * epoch)
            self.weights -= current_rate * weight_gradient
            self.bias -= current_rate * bias_gradient
            current_loss = self.loss(matrix, labels)
            self.loss_history.append(current_loss)
            if abs(previous_loss - current_loss) < self.tolerance:
                break
            previous_loss = current_loss
        return self

    def predict(self, matrix):
        return np.where(self.decision_function(matrix) >= 0, 1, -1)
