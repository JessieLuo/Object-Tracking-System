# backend/zmq/track_zmq_pub.py

import json
import zmq

from backend.publisher import TrackPublisher


class TrackZmqPublisher(TrackPublisher):
    def __init__(
            self,
            bind_addr: str,
            topic: str,
    ):
        self.topic = topic

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)

        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)

        self.socket.bind(bind_addr)

    def publish(self, payload: dict):
        data = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")

        self.socket.send_multipart([
            self.topic.encode("utf-8"),
            data,
        ])

    def close(self):
        self.socket.close()