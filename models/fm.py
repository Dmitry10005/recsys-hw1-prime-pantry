import numpy as np
import torch
from torch import nn


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, factors=32, lr=.01, reg=1e-5, epochs=15,
                 batch_size=4096, seed=2026, metadata=True):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Embedding(n_features, 1)
        self.factors = nn.Embedding(n_features, factors)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=.02)
        self.bias = nn.Parameter(torch.zeros(()))
        self.lr, self.reg, self.epochs = lr, reg, epochs
        self.batch_size, self.seed, self.metadata = batch_size, seed, metadata

    def forward(self, ids, values):
        if not self.metadata:
            # Оставляем только поля user ID и item ID.
            ids, values = ids[:, [0, 3]], values[:, [0, 3]]
        embedded = self.factors(ids) * values.unsqueeze(-1)
        interaction = .5 * (embedded.sum(1).square() - embedded.square().sum(1)).sum(1)
        return self.bias + (self.linear(ids).squeeze(-1) * values).sum(1) + interaction

    def fit(self, examples, history, cutoff):
        ids, values, labels = map(torch.as_tensor, examples)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        generator = torch.Generator().manual_seed(self.seed)
        self.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(labels), generator=generator)
            for batch in order.split(self.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self(ids[batch], values[batch]), labels[batch])
                loss.backward()
                optimizer.step()
        self.eval()
        self._prepare_scoring(history, cutoff)
        return self

    def _prepare_scoring(self, history, cutoff):
        """Считаем общие части FM заранее для быстрой выдачи по всему каталогу."""
        groups = history.groups(np.arange(history.n_users), np.arange(history.n_items), cutoff)
        vectors, offsets = [], []
        with torch.no_grad():
            for group_number, (ids, values) in enumerate(groups):
                if not self.metadata:
                    if group_number == 2:
                        values = np.zeros_like(values)
                    else:
                        values[:, 1:] = 0
                ids = torch.as_tensor(ids, dtype=torch.long)
                values = torch.as_tensor(values, dtype=torch.float32)
                emb = self.factors(ids) * values[..., None]
                vector = emb.sum(1)
                offset = (self.linear(ids).squeeze(-1) * values).sum(1)
                offset += .5 * (vector.square() - emb.square().sum(1)).sum(1)
                vectors.append(vector.numpy())
                offsets.append(offset.numpy())
        u, i, context = vectors
        bu, bi, bc = offsets
        self.user_vectors, self.item_vectors = u, i
        self.user_bias = bu + (u @ context.T).ravel()
        self.item_bias = bi + (i @ context.T).ravel()
        self.constant = float(self.bias.detach()) + bc[0]

    def score(self, users):
        return (self.user_vectors[users] @ self.item_vectors.T + self.user_bias[users, None]
                + self.item_bias[None, :] + self.constant)
