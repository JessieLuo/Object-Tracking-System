import numpy as np

from .matching import tlbr_to_xyah, xyah_to_tlbr


class KalmanBox:
    def __init__(self, box):
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0

        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)

        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4, dtype=np.float32) * 5.0

        self.x[:4, 0] = tlbr_to_xyah(box)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return xyah_to_tlbr(self.x[:, 0])

    def update(self, box):
        z = tlbr_to_xyah(box).reshape(4, 1)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P

        return xyah_to_tlbr(self.x[:, 0])

    def box(self):
        return xyah_to_tlbr(self.x[:, 0])