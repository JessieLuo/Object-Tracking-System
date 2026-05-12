from streaming.rtsp_pub import RtspWriter
from streaming.subscribe import FrameSource

from track.utils.fps import FpsMeter, draw_fps

from track.detection.yolo_det import UltralyticsYoloDetector
from track.trackers.osnet_sort import ReIDTracker


if __name__ == "__main__":
    source = FrameSource("rtsp://127.0.0.1:8554/maccam")

    detector = UltralyticsYoloDetector(
        "models/yolo26n_ncnn_model",
        conf=0.4,
        classes=[0]
    )

    tracker = ReIDTracker(
        iou_thresh=0.3,
        reid_thresh=0.75,
        max_lost=30,
        reid_interval=5,
        reid_weights="weights/osnet_x0_25_msmt17.pt"
    )

    writer = RtspWriter(
        url="rtsp://127.0.0.1:8554/out",
        size_wh=(640, 480),
        fps=15
    )
    writer.open()
    fps_meter = FpsMeter()

    try:
        for frame in source.frames():
            fps = fps_meter.tick()
            dets = detector.inference(frame)
            tracks, vis = tracker.update(frame, dets)
            print(tracks[:, :5])

            if fps is not None:
                vis = draw_fps(vis, fps)

            writer.write(vis)

    except KeyboardInterrupt:
        print("\nexiting...")

    finally:
        source.release()
        writer.close()
