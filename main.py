import argparse

import cv2
import numpy as np
import time

from detection.imx_zmq_adapter import IMXZmqAdapter
from detection.yolo_det import UltralyticsYoloDetector

from streaming.subscribe import FrameSource
from streaming.rtsp_pub import RtspWriter
from streaming.vid_pub import VideoFileWriter

from tracking.builder import build_tracker
from tracking.observation import Observation
from tracking.utils.draw import draw_tracks
from tracking.utils.boxes import nms_dets
from tracking.utils.fps import FpsMeter, draw_fps

from backend.json.track_encoder import encode_tracks
from backend.publisher import track_publisher


def load_detector(det_model):
    return UltralyticsYoloDetector(
        det_model,
        conf=0.1,
        classes=[0],
    )


def build_observations(dets):
    observations = []

    for det in dets:
        observations.append(
            Observation(
                box=det[:4].astype(np.float32),
                score=float(det[4]),
                feat=None,
            )
        )

    return observations


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
    if args.detector == "yolo":
        if args.det_model is None:
            raise ValueError("--det_model is required when --detector yolo")

        source = FrameSource(args.source)
        detector = load_detector(args.det_model)
        frame_iter = source.frames()

    elif args.detector == "imx_zmq":
        source = IMXZmqAdapter(
            addr=args.zmq_addr,
            print_latency=True,
        )

        detector = None
        frame_iter = None

    else:
        raise ValueError(f"unknown detector: {args.detector}")

    tracker = build_tracker(args)

    track_pub = None
    if args.emit_tracks:
        track_pub = track_publisher(
            transport=args.transport,
            camera_id=args.camera_id,
            udp_host=args.track_udp_host,
            udp_port=args.track_udp_port,
            zmq_bind=args.track_zmq_bind,
        )

    writer = build_writer(args)
    fps_meter = FpsMeter()

    try:
        while True:
            if args.detector == "yolo":
                frame = next(frame_iter)
                """获取到当前帧进入处理流程时的时间戳。该时间戳为系统时间"""
                frame_timestamp = time.time() 

                dets = detector.inference(frame)
                dets = nms_dets(dets, iou_thr=0.5,)

            else:
                frame, dets, det_valid = source.read()

                if not det_valid:
                    continue
                frame_timestamp = time.time() # align yolo timestamp style

            observations = build_observations(dets)

            tracks = tracker.update(
                frame,
                observations,
            )

            if track_pub is not None:
                payload = encode_tracks(
                    camera_id=args.camera_id,
                    frame_id=tracker.frame_id,
                    frame_timestamp=frame_timestamp,
                    tracks=tracks,
                )
                track_pub.publish(payload)

            vis = draw_tracks(
                frame,
                tracks,
            )

            fps = fps_meter.tick()

            if fps is not None:
                vis = draw_fps(
                    vis,
                    fps,
                )

            writer.write(vis)

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
        
        if track_pub is not None:
            track_pub.close()

        writer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=None)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--writer",
        choices=["rtsp", "video"],
        default="rtsp",
    )

    parser.add_argument(
        "--detector",
        choices=["yolo", "imx_zmq"],
        default="yolo",
    )

    parser.add_argument("--det_model", required=None)

    parser.add_argument(
        "--zmq_addr",
        default="tcp://127.0.0.1:5555",
    )

    parser.add_argument("--reid_model_name", required=True)
    parser.add_argument("--reid_weights", required=True)

    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=25)

    parser.add_argument("--show", action="store_true")

    parser.add_argument("--emit_tracks", action="store_true")

    parser.add_argument(
        "--transport",
        choices=["none", "udp", "zmq"],
        default="none",
    )
    parser.add_argument(
        "--track_udp_host",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--track_udp_port",
        type=int,
        default=5000,
    )
    parser.add_argument("--track_zmq_bind", type=str, default=None,)

    parser.add_argument(
        "--camera_id",
        type=str,
        default="cam0",
    )

    args = parser.parse_args()

    run(args)