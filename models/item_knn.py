import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=50, shrinkage=10):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, positive):
        self.history = positive.astype(np.float64)
        overlap = (self.history.T @ self.history).toarray()
        norm = np.sqrt(np.diag(overlap))
        denom = norm[:, None] * norm[None, :]
        similarity = np.divide(overlap, denom, out=np.zeros_like(overlap), where=denom > 0)
        if self.shrinkage:
            similarity *= overlap / (overlap + self.shrinkage)
        np.fill_diagonal(similarity, 0)
        # В каждом столбце оставляем ближайших соседей целевого товара.
        sources = np.argsort(-similarity, axis=0, kind="stable")[:self.neighbors]
        targets = np.broadcast_to(np.arange(similarity.shape[1]), sources.shape)
        self.weights = sparse.csr_matrix(
            (similarity[sources, targets].ravel(), (sources.ravel(), targets.ravel())),
            shape=similarity.shape,
        )
        self.weights.eliminate_zeros()
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
