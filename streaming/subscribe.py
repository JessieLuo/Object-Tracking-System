# subscribe.py
import time
from typing import Iterator, Optional, Tuple

import av
import numpy as np


class FrameSource:
    def __init__(
        self,
        src: str,
        transport: str = "tcp",
        size: Optional[Tuple[int, int]] = None,
    ):
        self.src = src
        self.transport = transport
        self.size = size

        self.container: Optional[av.container.InputContainer] = None
        self.stream: Optional[av.video.stream.VideoStream] = None

    def open(self) -> None:
        if not isinstance(self.src, str):
            raise ValueError("Only string sources (RTSP/file) are supported")

        options = {}

        # RTSP low-latency tuning
        if self.src.startswith("rtsp"):
            options = {
                "rtsp_transport": "tcp" if self.transport.lower() == "tcp" else "udp",
                "fflags": "nobuffer",
                "flags": "low_delay",
                "max_delay": "0",
                "reorder_queue_size": "0",
                "buffer_size": "102400",
                "stimeout": "5000000",  # 5s timeout (us)
            }

        self.container = av.open(self.src, mode="r", options=options)
        self.stream = self.container.streams.video[0]

        # safe threading multiplexing (e.g., for multiple subscribers)
        self.stream.thread_type = "AUTO"

    def frames(self) -> Iterator[np.ndarray]:
        if self.container is None or self.stream is None:
            self.open()

        while True:
            try:
                # demux -> avoids unnecessary decoding of non-video packets
                for packet in self.container.demux(self.stream):
                    for frame in packet.decode():
                        img: np.ndarray = frame.to_ndarray(format="bgr24")
                        yield img
                        break  # drop packet overflowed frames

            except Exception:
                # reconnect
                self.release()
                time.sleep(0.5)
                try:
                    self.open()
                except Exception:
                    time.sleep(1)

    def get_fps(self) -> float:
        if self.stream is None:
            return 0.0

        if self.stream.average_rate:
            return float(self.stream.average_rate)
        if self.stream.base_rate:
            return float(self.stream.base_rate)

        return 30.0

    def get_hw(self) -> Tuple[int, int]:
        if self.stream and self.stream.width and self.stream.height:
            return self.stream.height, self.stream.width
        return 0, 0

    def release(self) -> None:
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
        self.container = None
        self.stream = None
