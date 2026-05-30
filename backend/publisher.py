# backend/publisher.py

from abc import ABC, abstractmethod


class TrackPublisher(ABC):
    @abstractmethod
    def publish(self, payload: dict):
        pass

    def close(self):
        pass


class NullTrackPublisher(TrackPublisher):
    def publish(self, payload: dict):
        return


def track_publisher(
        transport: str,
        camera_id: str,
        udp_host: str | None = None,
        udp_port: int | None = None,
        zmq_bind: str | None = None,
):
    transport = transport.lower()

    if transport == "none":
        return NullTrackPublisher()

    if transport == "udp":
        from backend.udp.track_udp_pub import TrackUdpPublisher

        if udp_host is None or udp_port is None:
            raise ValueError("udp transport requires udp_host and udp_port")

        return TrackUdpPublisher(
            host=udp_host,
            port=udp_port,
        )

    if transport == "zmq":
        from backend.zmq.track_zmq_pub import TrackZmqPublisher

        if zmq_bind is None:
            raise ValueError("zmq transport requires zmq_bind")

        return TrackZmqPublisher(
            bind_addr=zmq_bind,
            topic=f"tracks.{camera_id}",
        )

    raise ValueError(f"unknown transport: {transport}")