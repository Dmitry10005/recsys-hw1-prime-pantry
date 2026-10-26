import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, reg=100.):
        self.reg = reg

    def fit(self, positive):
        self.history = positive.astype(np.float64)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::len(gram) + 1] += self.reg
        precision = cho_solve(cho_factor(gram, lower=True), np.eye(len(gram)))
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0.)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
