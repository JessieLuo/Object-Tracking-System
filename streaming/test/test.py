# run_cv.py
import argparse
import time
from queue import Queue, Full, Empty
from threading import Thread

from streaming.subscribe import FrameSource
from streaming.rtsp_pub import RtspWriter
from streaming.vid_pub import VideoFileWriter
from streaming.test.cvworker import CvWorker


STOP = object()


def is_rtsp(src: str) -> bool:
    return src.startswith("rtsp://")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", required=True)
    parser.add_argument("--out-rtsp", default=None)
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--fps", type=float, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    source = FrameSource(args.src)
    source.open()

    fps = args.fps if args.fps > 0 else source.get_fps()
    h, w = source.get_hw()

    worker = CvWorker()

    # --- writer ---
    rtsp_writer = None
    file_writer = None

    if args.out_rtsp:
        rtsp_writer = RtspWriter(args.out_rtsp, (w, h), fps)
    if args.out_file:
        file_writer = VideoFileWriter(args.out_file, (w, h), fps)

    # ========= CASE 1: RTSP（真实实时）=========
    if is_rtsp(args.src):
        q = Queue(maxsize=1)

        def producer():
            try:
                for frame in source.frames():
                    try:
                        q.put(frame, block=False)
                    except Full:
                        try:
                            q.get_nowait()  # drop old
                        except Empty:
                            pass
                        q.put(frame, block=False)
            finally:
                q.put(STOP)

        def consumer():
            opened = False
            interval = 1.0 / fps if fps > 0 else 0
            next_ts = time.time()

            while True:
                frame = q.get()
                if frame is STOP:
                    break

                if not opened:
                    if rtsp_writer:
                        rtsp_writer.open()
                    if file_writer:
                        file_writer.open()
                    opened = True

                frame = worker.process(frame)

                # fps pacing（关键）
                if interval > 0:
                    now = time.time()
                    if now < next_ts:
                        time.sleep(next_ts - now)
                    next_ts += interval

                if rtsp_writer:
                    rtsp_writer.write(frame)
                if file_writer:
                    file_writer.write(frame)

            if opened:
                if rtsp_writer:
                    rtsp_writer.close()
                if file_writer:
                    file_writer.close()

        t1 = Thread(target=producer, daemon=True)
        t2 = Thread(target=consumer, daemon=True)

        t1.start()
        t2.start()

        try:
            while t1.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    # ========= CASE 2: 本地文件 =========
    else:
        interval = 1.0 / fps if fps > 0 else 0
        next_ts = time.time()

        if rtsp_writer:
            rtsp_writer.open()
        if file_writer:
            file_writer.open()

        try:
            for frame in source.frames():
                frame = worker.process(frame)

                if interval > 0:
                    now = time.time()
                    if now < next_ts:
                        time.sleep(next_ts - now)
                    next_ts += interval

                if rtsp_writer:
                    rtsp_writer.write(frame)
                if file_writer:
                    file_writer.write(frame)

        except KeyboardInterrupt:
            pass

        finally:
            if rtsp_writer:
                rtsp_writer.close()
            if file_writer:
                file_writer.close()


if __name__ == "__main__":
    main()