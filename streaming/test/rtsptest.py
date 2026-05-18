import time
from queue import Queue
from threading import Thread

from streaming.subscribe import FrameSource
from streaming.rtsp_pub import RtspWriter


q = Queue(maxsize=1)
STOP = object()


def producer(source: FrameSource):
    try:
        for frame in source.frames():
            if not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    pass
            q.put(frame)
    finally:
        q.put(STOP)


def consumer(writer: RtspWriter):
    writer_opened = False

    interval = 1.0 / writer.fps
    start = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            frame = q.get()
            if frame is STOP:
                break

            if not writer_opened:
                h, w = frame.shape[:2]
                writer.size_wh = (w, h)
                writer.open()
                writer_opened = True

            # Core logic: make sure the frames are sent at the correct intervals
            target_ts = start + frame_idx * interval
            now = time.perf_counter()

            if now < target_ts:
                time.sleep(target_ts - now)

            writer.write(frame)
            frame_idx += 1

    finally:
        if writer_opened:
            writer.close()


if __name__ == "__main__":
    src = "rtsp://127.0.0.1:8554/maccam"

    source = FrameSource(src)

    writer = RtspWriter(
        "rtsp://127.0.0.1:8554/out",
        size_wh=(640, 480),
        fps=30,
    )

    t1 = Thread(target=producer, args=(source,))
    t2 = Thread(target=consumer, args=(writer,))

    t1.start()
    t2.start()

    try:
        t1.join()
        t2.join()

    except KeyboardInterrupt:
        q.put(STOP)
        t2.join()