import argparse
import time

import cv2

from detection.imx_zmq_adapter import IMXZmqAdapter
from detection.yolo_det import UltralyticsYoloDetector
from streaming.rtsp_pub import RtspWriter
from streaming.subscribe import FrameSource
from streaming.vid_pub import VideoFileWriter
from tracking.appearance.osnet import OSNetReID
from tracking.association.strategy import MotionAppearanceAssociation
from tracking.motion.kalman import KalmanFilterXYAH
from tracking.tracker import Tracker
from tracking.utils.fps import FpsMeter, draw_fps


def load_detector(det_model):
    return UltralyticsYoloDetector(det_model, conf=0.1, classes=[0], )


def load_tracker(model_name, weights):
    reid = OSNetReID(
        model_name=model_name,
        weights_path=weights,
        min_h=80,
    )

    association = MotionAppearanceAssociation(
        iou_thresh=0.3,
        low_iou_thresh=0.2,
        appearance_thresh=0.50,
        use_low_score_rescue=True,
        lost_association="appearance",
    )

    return Tracker(
        motion_factory=KalmanFilterXYAH,
        appearance_model=reid,
        association=association,
        max_lost=60,
        min_hits=2,
        appearance_interval=15,
        bank_size=30,
        ema_alpha=0.9,
        high_thresh=0.4,
        low_thresh=0.1,
        max_appearance_per_frame=2,
    )


def build_writer(args):
    if args.writer == "rtsp":
        writer = RtspWriter(url=args.output, size_wh=(args.width, args.height), fps=args.fps, )

    elif args.writer == "video":
        writer = VideoFileWriter(path=args.output, size_wh=(args.width, args.height), fps=args.fps, )

    else:
        raise ValueError(f"unknown writer: {args.writer}")

    writer.open()

    return writer


def run(args):
    if args.detector == "yolo":
        if args.det_model is None:
            raise ValueError("--det_model is required when --detector yolo")

        source = FrameSource(args.source)
        detector = load_detector(args.det_model)
        frame_iter = source.frames()

    elif args.detector == "imx_zmq":
        source = IMXZmqAdapter(addr=args.zmq_addr, print_latency=True, )
        detector = None
        frame_iter = None

    else:
        raise ValueError(f"unknown detector: {args.detector}")

    tracker = load_tracker(args.reid_model_name, args.reid_weights)

    writer = build_writer(args)

    fps_meter = FpsMeter()

    out_interval = 1.0 / args.fps

    next_write_t = time.perf_counter()

    try:
        while True:
            if args.detector == "yolo":
                frame = next(frame_iter)
                dets = detector.inference(frame)

            else:
                frame, dets, det_valid = source.read()
                if not det_valid:
                    continue

            tracks, vis = tracker.update(frame, dets)

            if len(tracks) > 0:
                print(tracks[:, :5])

            fps = fps_meter.tick()

            if fps is not None:
                xvis = draw_fps(vis, fps)

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
        if args.detector == "yolo":
            source.release()
        elif args.detector == "imx_zmq":
            source.close()

        writer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # If result not yolo-style no need sub streaming
    parser.add_argument("--source", required=None)

    parser.add_argument("--output", required=True)

    parser.add_argument("--writer", default="rtsp")

    parser.add_argument("--detector", choices=["yolo", "imx_zmq"], default="yolo", )

    parser.add_argument("--det_model", required=None)

    parser.add_argument("--zmq_addr", default="tcp://127.0.0.1:5555", )

    parser.add_argument("--reid_model_name", required=True)

    parser.add_argument("--reid_weights", required=True)

    parser.add_argument("--width", type=int, default=320)

    parser.add_argument("--height", type=int, default=320)

    parser.add_argument("--fps", type=float, default=25)

    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    run(args)
