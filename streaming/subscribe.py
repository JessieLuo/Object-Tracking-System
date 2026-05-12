import time
import threading

from typing import Optional, Tuple

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

        self.container = None
        self.stream = None

        self.running = False

        self.latest_frame = None

        self.lock = threading.Lock()

        self.thread = None

    # =====================================================
    # OPEN
    # =====================================================

    def open(self):

        options = {}

        if self.src.startswith("rtsp"):

            options = {
                "rtsp_transport":
                    "tcp"
                    if self.transport.lower() == "tcp"
                    else "udp",

                # low latency
                "fflags": "nobuffer",
                "flags": "low_delay",

                "max_delay": "0",

                "reorder_queue_size": "0",

                "buffer_size": "102400",

                # reconnect
                "stimeout": "5000000",
            }

        self.container = av.open(
            self.src,
            mode="r",
            options=options
        )

        self.stream = self.container.streams.video[0]

        # IMPORTANT
        # frame queue latency killer
        self.stream.thread_type = "NONE"

    # =====================================================
    # BACKGROUND READER
    # =====================================================

    def _reader_loop(self):

        while self.running:

            try:

                for packet in self.container.demux(self.stream):

                    if not self.running:
                        break

                    for frame in packet.decode():

                        img = frame.to_ndarray(
                            format="bgr24"
                        )

                        # resize if needed
                        if self.size is not None:

                            h, w = img.shape[:2]

                            tw, th = self.size

                            if (w, h) != (tw, th):

                                import cv2

                                img = cv2.resize(
                                    img,
                                    (tw, th)
                                )

                        # overwrite old frame
                        with self.lock:

                            self.latest_frame = img

                        # IMPORTANT:
                        # drop old decode queue
                        break

            except Exception:
                try:
                    if self.container is not None:
                        self.container.close()

                except Exception:
                    pass

                self.container = None
                self.stream = None

                time.sleep(0.5)

                try:
                    self.open()

                except Exception:
                    time.sleep(1)

    # =====================================================
    # PUBLIC
    # =====================================================

    def frames(self):

        if self.running:
            return

        self.open()

        self.running = True

        self.thread = threading.Thread(
            target=self._reader_loop,
            daemon=True
        )

        self.thread.start()

        while self.running:

            frame = None

            with self.lock:

                if self.latest_frame is not None:

                    frame = self.latest_frame.copy()

            if frame is None:

                time.sleep(0.001)

                continue

            yield frame

    # =====================================================
    # UTILS
    # =====================================================

    def get_fps(self):

        if self.stream is None:
            return 0.0

        if self.stream.average_rate:
            return float(self.stream.average_rate)

        if self.stream.base_rate:
            return float(self.stream.base_rate)

        return 30.0

    def get_hw(self):

        if (
            self.stream
            and self.stream.width
            and self.stream.height
        ):

            return (
                self.stream.height,
                self.stream.width
            )

        return 0, 0

    # =====================================================
    # CLOSE
    # =====================================================

    def release(self):

        self.running = False

        if self.thread is not None:
            import threading
            if threading.current_thread() != self.thread:
                self.thread.join(timeout=1)

        if self.container is not None:
            try:
                self.container.close()

            except Exception:
                pass

        self.container = None
        self.stream = None
