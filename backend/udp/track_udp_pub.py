# backend/udp/track_udp_pub.py

import json
import socket

from backend.publisher import TrackPublisher


class TrackUdpPublisher(TrackPublisher):
    def __init__(
            self,
            host: str,
            port: int,
    ):
        self.addr = (host, int(port))

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

    def publish(self, payload: dict):
        data = json.dumps(
            payload,
            separators=(",", ":"),
        ).encode("utf-8")

        self.sock.sendto(
            data,
            self.addr,
        )

    def close(self):
        self.sock.close()