import time
import argparse
import cv2

from streaming.rtsp_pub import RtspWriter
from streaming.vid_pub import VideoFileWriter
from streaming.subscribe import FrameSource

from track.utils.fps import FpsMeter, draw_fps

from track.detection.yolo_det import UltralyticsYoloDetector
from track.trackers.osnet_sort import ReIDTracker


def load_detector(model):
    return UltralyticsYoloDetector(
        model,
        conf=0.1,
        classes=[0],
    )


def load_tracker(model_name, weights):
    return ReIDTracker(
        iou_thresh=0.3,
        low_iou_thresh=0.2,
        high_thresh=0.4,
        low_thresh=0.1,
        reid_thresh=0.50,
        max_lost=60,
        min_hits=2,
        reid_interval=15,
        bank_size=30,
        ema_alpha=0.9,
        reid_model_name=model_name,
        reid_weights=weights,
    )


def build_writer(args):

    if args.writer == "rtsp":

        writer = RtspWriter(
            url=args.output,
            size_wh=(args.width, args.height),
            fps=args.fps,
        )

    elif args.writer == "video":

        writer = VideoFileWriter(
            path=args.output,
            size_wh=(args.width, args.height),
            fps=args.fps,
        )

    else:
        raise ValueError(f"unknown writer: {args.writer}")

    writer.open()

    return writer


def run(args):

    source = FrameSource(args.source)

    detector = load_detector(args.model)

    tracker = load_tracker(args.reid_model_name, args.reid_weights)

    writer = build_writer(args)

    fps_meter = FpsMeter()

    out_interval = 1.0 / args.fps

    next_write_t = time.perf_counter()

    try:

        for frame in source.frames():

            dets = detector.inference(frame)

            tracks, vis = tracker.update(frame, dets)

            if len(tracks) > 0:
                print(tracks[:, :5])

            fps = fps_meter.tick()

            if fps is not None:
                vis = draw_fps(vis, fps)

            now = time.perf_counter()

            if now >= next_write_t:

                writer.write(vis)

                next_write_t += out_interval

                if next_write_t < now - out_interval:
                    next_write_t = now + out_interval

            if args.show:

                cv2.imshow("track", vis)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:

        print("\nexiting...")

    finally:

        source.release()

        writer.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)

    parser.add_argument("--output", required=True)

    parser.add_argument("--writer", default="rtsp")

    parser.add_argument("--model", required=True)

    parser.add_argument("--reid_model_name", required=True)
    
    parser.add_argument("--reid_weights", required=True)

    parser.add_argument("--width", type=int, default=320)

    parser.add_argument("--height", type=int, default=320)

    parser.add_argument("--fps", type=float, default=25)

    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    run(args)
