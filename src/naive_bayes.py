"""Gaussian Naive Bayes classifier, written from scratch with NumPy.

Naive Bayes is a Bayesian network where the class is the only parent of every
feature. It assumes the features are independent once the class is known:

    P(class | x)  is proportional to  P(class) * product over features of P(x_i | class)

Each P(x_i | class) is a Gaussian (bell curve) with a mean and variance learned
from the training rows of that class. We add logarithms instead of multiplying
probabilities, which avoids numbers so small that they round to zero."""
import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        if var_smoothing < 0:
            raise ValueError("var_smoothing cannot be negative")
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (rows are examples)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(X) == 0:
            raise ValueError("there is no training data")

        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            rows = X[y == c]
            self.priors_[i] = len(rows) / len(X)          # P(class)
            self.means_[i] = rows.mean(axis=0)            # mean of each feature
            self.vars_[i] = rows.var(axis=0)              # variance of each feature

        # A tiny extra variance stops division by zero when a feature is
        # constant inside a class.
        self.vars_ += self.var_smoothing * max(X.var(axis=0).max(), 1e-12)
        return self

    def _log_joint(self, X):
        """log P(class) + sum of log P(x_i | class), for every row and class."""
        if not hasattr(self, "classes_"):
            raise RuntimeError("call fit() before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.means_.shape[1]:
            raise ValueError(f"X must have {self.means_.shape[1]} features per row")
        scores = []
        for i in range(len(self.classes_)):
            log_gauss = -0.5 * (
                np.log(2 * np.pi * self.vars_[i])
                + (X - self.means_[i]) ** 2 / self.vars_[i]
            )
            scores.append(np.log(self.priors_[i]) + log_gauss.sum(axis=1))
        return np.column_stack(scores)

    def predict_proba(self, X):
        """P(class | x) for each row, as a matrix with one column per class."""
        log_joint = self._log_joint(X)
        log_joint -= log_joint.max(axis=1, keepdims=True)  # numerical safety
        probs = np.exp(log_joint)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        log_joint = self._log_joint(X)  # raises RuntimeError if not fitted
        return self.classes_[np.argmax(log_joint, axis=1)]

    def score(self, X, y):
        """Accuracy: the fraction of correct predictions."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
