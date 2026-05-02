

from streaming.rtsp_pub import RtspWriter
from streaming.subscribe import FrameSource
from track.det_track import YOLOTracker
from track.utils.fps import FpsMeter, draw_fps


if __name__ == "__main__":
    source = FrameSource("rtsp://127.0.0.1:8554/maccam")

    tracker = YOLOTracker(
        model_path="models/yolo26n.mnn",
        conf=0.3,
        classes=[0],  # person class only
    )

    writer = RtspWriter(
        url="rtsp://127.0.0.1:8554/out",
        size_wh=(640, 480),
        fps=30
    )
    writer.open()
    fps_meter = FpsMeter()

    try:
        for frame in source.frames():
            fps = fps_meter.tick()
            dets, vis = tracker.infer(frame)
            print(dets[:3])

            if fps is not None:
                vis = draw_fps(vis, fps)

            writer.write(vis)

    except KeyboardInterrupt:
        print("\nexiting...")

    finally:
        source.release()
        writer.close()
