import time
from typing import Tuple, Optional

import av
import cv2 as cv
import numpy as np
from av.utils import Fraction

from streaming.publish import BaseWriter


class RtspWriter(BaseWriter):

    def __init__(
            self,
            url: str,
            size_wh: Tuple[int, int],
            fps: float = 15.0,
            preset: str = "ultrafast",
            gop: Optional[int] = 10,
            bitrate_k: int = 500,
    ):
        super().__init__(url)
        self.size_wh = size_wh
        self.fps = fps
        self.preset = preset
        self.gop = gop
        self.bitrate_k = bitrate_k
        self.container = None
        self.stream = None

        # output throttling
        self.frame_interval = 1.0 / fps
        self.last_write = 0.0

    def open(self) -> None:
        """create an RTSP output channel"""
        w, h = self.size_wh

        self.container = av.open(
            self.dst,
            mode="w",
            format="rtsp"
        )

        self.stream = self.container.add_stream(
            "libx264",
            rate=Fraction(int(self.fps), 1)
        )

        self.stream.width = w
        self.stream.height = h
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "bf": "0", "g": "10",
            "threads": "1",
            "rc-lookahead": "0",
            "sc_threshold": "0",
            "b": f"{self.bitrate_k}k",
            "maxrate": f"{self.bitrate_k}k",
            "bufsize": f"{self.bitrate_k // 2}k",
        }

        super().open()

    def write(self, frame: np.ndarray) -> None:
        """The core logic render the processed frame to video stream"""
        if not self._opened:
            return

        now = time.time()

        if now - self.last_write < self.frame_interval:
            return

        self.last_write = now

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        h, w = frame.shape[:2]
        exp_w, exp_h = self.size_wh
        if (w, h) != (exp_w, exp_h):
            frame = cv.resize(
                frame,
                (exp_w, exp_h),
                interpolation=cv.INTER_LINEAR
            )

        frame = np.ascontiguousarray(frame)

        av_frame = av.VideoFrame.from_ndarray(
            frame,
            format="bgr24"
        )

        packets = self.stream.encode(av_frame)

        for packet in packets:
            try:
                self.container.mux(packet)

            except Exception:
                return

    def close(self) -> None:
        if self.container and self.stream:
            try:
                packets = self.stream.encode(None)
                for packet in packets:
                    self.container.mux(packet)
            except Exception:
                pass

            try:
                self.container.close()
            except Exception:
                pass

        self.container = None
        self.stream = None

        super().close()
