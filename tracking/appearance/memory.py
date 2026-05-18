from collections import deque

import numpy as np


class AppearanceMemory:
    def __init__(self, bank_size=20, ema_alpha=0.9):
        self.features = deque(maxlen=bank_size)
        self.smooth_feat = None
        self.ema_alpha = ema_alpha

    def update(self, feat):
        if feat is None:
            return

        feat = feat.astype(np.float32)
        norm = np.linalg.norm(feat)

        if norm < 1e-6:
            return

        feat = feat / norm
        self.features.append(feat)

        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
            return

        self.smooth_feat = (
            self.ema_alpha * self.smooth_feat
            + (1.0 - self.ema_alpha) * feat
        )

        self.smooth_feat = self.smooth_feat / max(
            np.linalg.norm(self.smooth_feat),
            1e-6,
        )

    def similarity(self, feat):
        if feat is None:
            return -1.0

        feat = feat.astype(np.float32)
        feat = feat / max(np.linalg.norm(feat), 1e-6)

        smooth_sim = -1.0
        bank_sim = -1.0

        if self.smooth_feat is not None:
            smooth_sim = float(np.dot(self.smooth_feat, feat))

        if len(self.features) > 0:
            bank_sim = max(float(np.dot(f, feat)) for f in self.features)

        if smooth_sim < 0 and bank_sim < 0:
            return -1.0

        if smooth_sim < 0:
            return bank_sim

        if bank_sim < 0:
            return smooth_sim

        return 0.7 * smooth_sim + 0.3 * bank_sim


class AppearanceStore:
    def __init__(self, bank_size=20, ema_alpha=0.9):
        self.bank_size = bank_size
        self.ema_alpha = ema_alpha
        self.memories = {}

    def ensure(self, track_id):
        track_id = int(track_id)

        if track_id not in self.memories:
            self.memories[track_id] = AppearanceMemory(
                bank_size=self.bank_size,
                ema_alpha=self.ema_alpha,
            )

        return self.memories[track_id]

    def update(self, track_id, feat):
        if feat is None:
            return

        memory = self.ensure(track_id)
        memory.update(feat)

    def similarity(self, track_id, feat):
        memory = self.memories.get(int(track_id))

        if memory is None:
            return -1.0

        return memory.similarity(feat)

    def remove(self, track_id):
        self.memories.pop(int(track_id), None)