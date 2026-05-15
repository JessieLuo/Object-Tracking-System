import numpy as np
import scipy.linalg

from .matching import tlbr_to_xyah, xyah_to_tlbr

EPS = 1e-6  # inherited from ultralytics


class KalmanFilterXYAH:
    """
    Stateful Kalman filter wrapper for one bounding box track.

    External box format:
        TopLeftBottomRight = [x1, y1, x2, y2]

    Internal state format:
        [cx, cy, a, h, vx, vy, va, vh]

    Internally it follows the ByteTrack / DeepSORT XYAH Kalman idea:
        - constant velocity model
        - detector observation is [cx, cy, a, h]
        - noise Q (process) & R (measurement) is dynamically scaled by current box height
    """

    def __init__(
            self,
            box,
            std_weight_position: float = 1.0 / 20,
            std_weight_velocity: float = 1.0 / 160,
    ):
        ndim, dt = 4, 1.0

        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # F: state transition matrix
        # Predict next state using constant velocity model
        self._motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # H: observation matrix
        # Map 8D Kalman state -> 4D detector measurement space.
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # initially acquire measurement
        measurement = tlbr_to_xyah(box).astype(np.float32)
        measurement[3] = max(float(measurement[3]), EPS)

        # x: state mean
        # P: state covariance (state uncertainty)
        self.mean, self.covariance = self._initiate(measurement)

    def _initiate(self, measurement):
        """
        Initialize a new track from one detector measurement.
        measurement:
            [cx, cy, a, h]
        """
        mean_pos = measurement.astype(np.float32)
        mean_vel = np.zeros_like(mean_pos, dtype=np.float32)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)

        h = max(float(mean_pos[3]), EPS)

        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                1e-2,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                1e-5,
                10 * self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )

        covariance = np.diag(std * std).astype(np.float32)
        return mean, covariance

    def predict(self):
        """
        Predict current track state forward by one frame.

        Returns:
            Predicted TLBR box.
        """
        h = max(float(self.mean[3]), EPS)

        std_pos = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )

        std_vel = np.array(
            [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )

        # Q: process noise covariance
        # Motion uncertainty of the prediction model
        motion_cov = np.diag(
            np.square(np.r_[std_pos, std_vel])
        ).astype(np.float32)

        # x' = F x
        self.mean = np.dot(self.mean, self._motion_mat.T).astype(np.float32)
        # P' = F P F.T + Q
        self.covariance = (
                np.linalg.multi_dot(
                    (self._motion_mat, self.covariance, self._motion_mat.T)
                )
                + motion_cov
        ).astype(np.float32)

        self.covariance = self._symmetrize(self.covariance)
        self.mean[3] = max(float(self.mean[3]), EPS)

        return xyah_to_tlbr(self.mean[:4]).astype(np.float32)

    def _project(self):
        """
        Project 8D state distribution to 4D measurement space.

        Returns:
                projected_mean [cx, cy, a, h]

                projected_cov HPH.T + R
        """
        h = max(float(self.mean[3]), EPS)

        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )

        # R: measurement noise covariance
        # Detector observation uncertainty.
        innovation_cov = np.diag(std * std).astype(np.float32)

        # H x
        projected_mean = np.dot(self._update_mat, self.mean).astype(np.float32)

        # HPH.T
        projected_cov = np.linalg.multi_dot(
            (self._update_mat, self.covariance, self._update_mat.T)
        ).astype(np.float32)

        # S = HPH.T + R
        projected_cov = projected_cov + innovation_cov
        projected_cov = self._symmetrize(projected_cov)

        return projected_mean, projected_cov

    def update(self, box):
        """
        Correct current track state using one detector box.

        Args:
            box:
                TLBR box = [x1, y1, x2, y2]

        Returns:
            Updated TLBR box.
        """
        # z from cam sensor + det result
        measurement = tlbr_to_xyah(box).astype(np.float32)

        if not np.isfinite(measurement).all():
            return xyah_to_tlbr(self.mean[:4]).astype(np.float32)

        measurement[3] = max(float(measurement[3]), EPS)

        projected_mean, projected_cov = self._project()

        try:
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov,
                lower=True,
                check_finite=False,
            )

            # K = P'H.T@inv(S)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower),
                np.dot(self.covariance, self._update_mat.T).T,
                check_finite=False,
            ).T
            # cho_solve(...) a potential greate optimizer that increase infer

        except (np.linalg.LinAlgError, ValueError):
            return xyah_to_tlbr(self.mean[:4]).astype(np.float32)

        # y = z - Hx
        innovation = measurement - projected_mean

        # x = x' + K y
        self.mean = (
                self.mean + np.dot(innovation, kalman_gain.T)
        ).astype(np.float32)
        # P = P' - K@S@K.T
        # TODO: why not P = (I - K@H)P' OR P = P' - P'@K@H
        self.covariance = (
                self.covariance
                - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        ).astype(np.float32)

        self.covariance = self._symmetrize(self.covariance)
        self.mean[3] = max(float(self.mean[3]), EPS)

        return xyah_to_tlbr(self.mean[:4]).astype(np.float32)

    def gating_distance(self, boxes, only_position=False, metric="maha"):
        """
        Compute distance between this track state and candidate TLBR boxes.

        This can be used before matching to reject impossible candidates.

        Args:
            boxes:
                Nx4 TLBR boxes.

            only_position:
                If True, use only [cx, cy].

            metric:
                "maha" or "gaussian".

        Returns:
            distances:
                shape = (N,)
        """
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.float32)

        measurements = np.asarray(
            [tlbr_to_xyah(box) for box in boxes],
            dtype=np.float32,
        )

        measurements[:, 3] = np.maximum(measurements[:, 3], EPS)

        mean, covariance = self._project()

        if only_position:
            mean = mean[:2]
            covariance = covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean

        if metric == "gaussian":
            return np.sum(d * d, axis=1).astype(np.float32)

        if metric == "maha":
            try:
                cholesky_factor = np.linalg.cholesky(covariance)
                z = scipy.linalg.solve_triangular(
                    cholesky_factor,
                    d.T,
                    lower=True,
                    check_finite=False,
                    overwrite_b=True,
                )
                return np.sum(z * z, axis=0).astype(np.float32)

            except (np.linalg.LinAlgError, ValueError):
                inv_cov = np.linalg.pinv(covariance)
                return np.einsum("ij,jk,ik->i", d, inv_cov, d).astype(np.float32)

        raise ValueError("invalid distance metric")

    def box(self):
        """
        Return current state as TLBR box.
        """
        return xyah_to_tlbr(self.mean[:4]).astype(np.float32)

    @staticmethod
    def _symmetrize(mat):
        return ((mat + mat.T) * 0.5).astype(np.float32)


class _DummyKalman:
    """Ablation Test / Fake Kalman to
    verify calculation pressure."""

    def __init__(self, box, *args, **kwargs):
        self._box = np.asarray(box, dtype=np.float32)

    def predict(self):
        return self._box

    def update(self, box):
        self._box = np.asarray(box, dtype=np.float32)
        return self._box

    def gating_distance(self, boxes, only_position=False, metric="maha"):
        return np.zeros((len(boxes),), dtype=np.float32)

    def box(self):
        return self._box
