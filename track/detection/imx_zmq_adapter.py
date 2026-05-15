import json
import zmq
import time
import numpy as np


class IMXZmqAdapter:
    """
    Receive IMX500 frame + detections from ZMQ.

    Output:
        frame: uint8, shape (H, W, 3)
        dets : float32, shape (N, 6)
               [x1, y1, x2, y2, score, cls]
    """

    def __init__(
        self,
        addr="tcp://127.0.0.1:5555",
        print_latency=True,
        latency_log_interval=2.0,
    ):
        self.addr = addr
        self.print_latency = print_latency
        self.latency_log_interval = latency_log_interval

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)

        # 关键：接收队列小，避免积压
        self.sock.setsockopt(zmq.RCVHWM, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")

        self.sock.connect(addr)

        self.last_log_t = time.perf_counter()
        self.recv_count = 0
        self.last_latency_ms = 0.0
        self.last_frame_id = -1

    def _recv_latest_multipart(self):
        """
        Block until at least one message arrives.
        Then drain all queued messages and keep only the latest.

        This avoids old-frame backlog.
        """
        msg = self.sock.recv_multipart()

        while True:
            try:
                msg = self.sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        return msg

    def read(self):
        header_bytes, frame_bytes = self._recv_latest_multipart()

        now_ns = time.perf_counter_ns()

        header = json.loads(header_bytes.decode("utf-8"))

        shape = tuple(header["shape"])

        frame = np.frombuffer(
            frame_bytes,
            dtype=np.uint8,
        ).reshape(shape)

        # publisher sends XRGB8888: (H, W, 4)
        # use first 3 channels for tracker/ReID.
        frame = np.ascontiguousarray(frame[:, :, :3])

        objects = header.get("objects", [])

        if objects:
            dets = np.array(
                [
                    [
                        obj["x1"],
                        obj["y1"],
                        obj["x2"],
                        obj["y2"],
                        obj["score"],
                        obj["cls"],
                    ]
                    for obj in objects
                ],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        send_ts_ns = header.get("send_ts_ns", None)

        if send_ts_ns is not None:
            self.last_latency_ms = (now_ns - int(send_ts_ns)) / 1e6

        self.last_frame_id = int(header.get("frame_id", -1))
        self.recv_count += 1

        det_valid = bool(header.get("det_valid", True))
        now = time.perf_counter()
        if self.print_latency and now - self.last_log_t >= self.latency_log_interval:
            print(
                f"[IMX-ZMQ] recv_fps={self.recv_count / (now - self.last_log_t):.1f} "
                f"latency={self.last_latency_ms:.1f}ms "
                f"frame_id={self.last_frame_id} "
                f"dets={len(dets)} "
                f"det_valid={det_valid}"
            )

            self.recv_count = 0
            self.last_log_t = now

        return frame, dets, det_valid


    def close(self):
        self.sock.close()