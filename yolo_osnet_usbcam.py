import time

from streaming.rtsp_pub import RtspWriter
from streaming.subscribe import FrameSource

from track.utils.fps import FpsMeter, draw_fps

from track.detection.yolo_det import UltralyticsYoloDetector
from track.trackers.osnet_sort import ReIDTracker


if __name__ == "__main__":
    source = FrameSource("rtsp://127.0.0.1:8554/usbcam")

    detector = UltralyticsYoloDetector(
        "models/yolo26n.mnn",
        conf=0.1,
        classes=[0],
    )

    tracker = ReIDTracker(
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
        reid_weights="weights/osnet_x0_25_msmt17.pt",
    )

    OUT_FPS = 25.0
    OUT_INTERVAL = 1.0 / OUT_FPS

    writer = RtspWriter(
        url="rtsp://127.0.0.1:8554/usbdet",
        size_wh=(320, 320),
        fps=OUT_FPS,
    )

    writer.open()

    fps_meter = FpsMeter()

    last_vis = None
    next_write_t = time.perf_counter()

    try:
        for frame in source.frames():
            dets = detector.inference(frame)
            tracks, vis = tracker.update(frame, dets)
            print(tracks[:, :5])

            fps = fps_meter.tick()
            if fps is not None:
                vis = draw_fps(vis, fps)

            last_vis = vis

            now = time.perf_counter()

            if now >= next_write_t:
                writer.write(last_vis)

                # 固定输出节奏，不跟随处理耗时乱跳
                next_write_t += OUT_INTERVAL

                # 如果处理太慢，直接追到当前时间附近，避免越积越延迟
                if next_write_t < now - OUT_INTERVAL:
                    next_write_t = now + OUT_INTERVAL

    except KeyboardInterrupt:
        print("\nexiting...")

    finally:
        source.release()
        writer.close()