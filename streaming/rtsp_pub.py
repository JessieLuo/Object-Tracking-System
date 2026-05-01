# rtsp_pub.py
from typing import Tuple, Optional

import cv2 as cv
import numpy as np
import av
from av.utils import Fraction

from publish import BaseWriter


class RtspWriter(BaseWriter):
    def __init__(
        self,
        url: str,
        size_wh: Tuple[int, int],
        fps: float = 30.0,
        preset: str = "veryfast",
        gop: Optional[int] = None,
        bitrate_k: int = 1000,
    ):
        super().__init__(url)

        self.size_wh = size_wh
        self.fps = fps
        self.preset = preset
        self.gop = gop if gop is not None else int(max(2, round(fps)))
        self.bitrate_k = bitrate_k

        self.container = None
        self.stream = None

    def open(self) -> None:
        w, h = self.size_wh

        self.container = av.open(self.dst, mode="w", format="rtsp")

        self.stream = self.container.add_stream(
            "libx264", rate=Fraction(int(self.fps), 1)
        )
        self.stream.width = w
        self.stream.height = h
        self.stream.pix_fmt = "yuv420p"

        # low latency tuning
        self.stream.options = {
            "preset": self.preset,
            "tune": "zerolatency",
            "profile": "baseline",
            "g": str(self.gop),
            "bf": "0",
            "b": f"{self.bitrate_k}k",
        }

        super().open()

    def write(self, frame: np.ndarray) -> None:
        if not self._opened:
            return

        # ffmpeg encoding format check (e.g., uint8, 3 channels, no alpha)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        # process alpha if exists (e.g., RGBA -> RGB)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        h, w = frame.shape[:2]
        exp_w, exp_h = self.size_wh

        # only resize when necessary (e.g., for RTSP streaming stability)
        if (w, h) != (exp_w, exp_h):
            frame = cv.resize(frame, (exp_w, exp_h),
                              interpolation=cv.INTER_LINEAR)

        frame = np.ascontiguousarray(frame)

        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")

        # encode may return multiple packets for keyframes), so mux all of them
        packets = self.stream.encode(av_frame)
        for packet in packets:
            self.container.mux(packet)

    def close(self) -> None:
        if self.container and self.stream:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            except Exception:
                pass

            self.container.close()

        self.container = None
        self.stream = None
        super().close()
