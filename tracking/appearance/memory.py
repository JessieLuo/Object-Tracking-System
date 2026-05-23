from collections import deque

import numpy as np


class AppearanceMemory:
    def __init__(
            self,
            bank_size=20,
            ema_alpha=0.9,
    ):
        self.features = deque(maxlen=bank_size)
        self.smooth_feat = None
        self.ema_alpha = ema_alpha

    def update(self, feat: np.ndarray):
        if feat is None:
            return

        feat = feat.astype(np.float32)

        self.features.append(feat)

        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
            return

        self.smooth_feat = (
                self.ema_alpha * self.smooth_feat
                + (1.0 - self.ema_alpha) * feat
        ).astype(np.float32)

        norm = np.linalg.norm(self.smooth_feat)

        if norm > 1e-6:
            self.smooth_feat = (
                self.smooth_feat / norm
            ).astype(np.float32)

    def similarity(self, feat: np.ndarray) -> float:
        if feat is None:
            return -1.0

        feat = feat.astype(np.float32)

        smooth_sim = -1.0
        bank_sim = -1.0

        if self.smooth_feat is not None:
            smooth_sim = float(
                np.dot(self.smooth_feat, feat)
            )

        if len(self.features) > 0:
            bank_sim = max(
                float(np.dot(f, feat))
                for f in self.features
            )

        if smooth_sim < 0 and bank_sim < 0:
            return -1.0

        if smooth_sim < 0:
            return bank_sim

        if bank_sim < 0:
            return smooth_sim

        return 0.7 * smooth_sim + 0.3 * bank_sim


class AppearanceStore:
    def __init__(
            self,
            bank_size=20,
            ema_alpha=0.9,
    ):
        self.bank_size = bank_size
        self.ema_alpha = ema_alpha
        self.memories = {}

    def ensure(self, track_id) -> AppearanceMemory:
        track_id = int(track_id)

        if track_id not in self.memories:
            self.memories[track_id] = AppearanceMemory(
                bank_size=self.bank_size,
                ema_alpha=self.ema_alpha,
            )

        return self.memories[track_id]

    def initiate(self, track, observation):
        feat = observation.feat

        if feat is None:
            return

        memory = self.ensure(track.id)
        memory.update(feat)

    def update(self, track, observation):
        feat = observation.feat

        if feat is None:
            return

        memory = self.ensure(track.id)
        memory.update(feat)

    def similarity(self, track, feat: np.ndarray) -> float:
        memory = self.memories.get(int(track.id))

        if memory is None:
            return -1.0

        return memory.similarity(feat)

    def remove(self, track):
        self.memories.pop(int(track.id), None)