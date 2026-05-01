# vid_pub.py
import cv2 as cv
from typing import Optional, Tuple

from publish import BaseWriter


class VideoFileWriter(BaseWriter):
    def __init__(self, path: Optional[str], size_wh: Tuple[int, int], fps: float):
        self.path = path
        self.size_wh = size_wh
        self.fps = fps
        self.writer: cv.VideoWriter | None = None

    def open(self) -> None:
        if not self.path:
            return

        fourcc = cv.VideoWriter_fourcc(*"mp4v")

        self.writer = cv.VideoWriter(
            self.path,
            fourcc,
            self.fps,
            self.size_wh,
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {self.path}")

    def write(self, frame) -> None:
        if self.writer is not None:
            self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None