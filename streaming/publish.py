import numpy as np
from abc import ABC, abstractmethod


class BaseWriter(ABC):
    """
    Unified interface for all video output backends (RTSP, file, etc.)
    """

    def __init__(self, dst: str):
        self.dst = dst
        self._opened = False

    @abstractmethod
    def open(self) -> None:
        self._opened = True

    @abstractmethod
    def write(self, frame: np.ndarray) -> None:
        if not self._opened:
            raise RuntimeError("Writer not opened")

    @abstractmethod
    def close(self) -> None:
        self._opened = False

    def is_open(self) -> bool:
        return self._opened