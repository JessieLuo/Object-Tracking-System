# tracking/interfaces.py

from typing import Protocol
import numpy as np


class MotionModule(Protocol):
    def predict(self, tracks) -> None:
        ...

    def initiate(self, track, observation) -> None:
        ...

    def update(self, track, observation) -> None:
        ...

    def remove(self, track) -> None:
        ...

    def gating_distance(self, track, boxes: np.ndarray) -> np.ndarray:
        ...


class AppearanceModule(Protocol):
    def begin_frame(self, frame, frame_id: int) -> None:
        ...

    def initiate(self, track, observation) -> None:
        ...

    def update(self, track, observation) -> None:
        ...

    def remove(self, track) -> None:
        ...

    def similarity(self, track, feat: np.ndarray) -> float:
        ...