# tracking/motion/model.py

import numpy as np

from tracking.motion.kalman import KalmanFilterXYAH


class KalmanMotion:
    def __init__(self):
        self.filters = {}

    def predict(self, tracks):
        for track in tracks:
            kf = self.filters.get(track.id)

            if kf is not None:
                kf.predict()

    def initiate(self, track, observation):
        self.filters[track.id] = KalmanFilterXYAH(
            observation.box,
        )

    def update(self, track, observation):
        kf = self.filters.get(track.id)

        if kf is not None:
            kf.update(observation.box)

    def remove(self, track):
        self.filters.pop(track.id, None)

    def gating_distance(self, track, boxes: np.ndarray):
        kf = self.filters.get(track.id)

        if kf is None:
            return np.zeros(
                (len(boxes),),
                dtype=np.float32,
            )

        return kf.gating_distance(boxes)