import numpy as np
from scipy import sparse
from .ials import IALS


def preference(rating, timestamp, cutoff, half_life=180., floor=.5):
    age_days = np.maximum(0, (cutoff - np.asarray(timestamp)) / 86400.)
    return ((np.asarray(rating) - 3.) / 2.) * (
        floor + (1 - floor) * np.exp2(-age_days / half_life)
    )


class CustomIALS(IALS):
    def __init__(self, half_life=180., floor=.5, **kwargs):
        super().__init__(**kwargs)
        self.half_life, self.floor = half_life, floor

    def fit_events(self, events, shape, cutoff):
        values = preference(events.rating, events.timestamp, cutoff, self.half_life, self.floor)
        targets = sparse.csr_matrix((values, (events.u, events.i)), shape=shape)
        return self.fit(targets)
