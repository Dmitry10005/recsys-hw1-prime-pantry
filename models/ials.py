import numpy as np


class IALS:
    def __init__(self, factors=32, reg=10., alpha=20., iterations=12, seed=2026):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def _update(self, targets, fixed):
        result = np.zeros((targets.shape[0], self.factors))
        base = fixed.T @ fixed + self.reg * np.eye(self.factors)
        for row in range(targets.shape[0]):
            a, b = targets.indptr[row:row + 2]
            indices, values = targets.indices[a:b], targets.data[a:b]
            if len(indices):
                y = fixed[indices]
                result[row] = np.linalg.solve(
                    base + self.alpha * (y.T @ y),
                    (1 + self.alpha) * (y.T @ values),
                )
        return result

    def fit(self, targets):
        # Явные нули — это наблюдавшиеся низкие оценки, их удалять нельзя.
        targets = targets.astype(np.float64).tocsr()
        transpose = targets.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, .01, (targets.shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._update(targets, self.item_factors)
            self.item_factors = self._update(transpose, self.user_factors)
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
