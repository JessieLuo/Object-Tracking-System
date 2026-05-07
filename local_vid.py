from streaming.vid_pub import VideoFileWriter
from streaming.subscribe import FrameSource
from track.det_track import YOLOTracker
from track.utils.fps import FpsMeter, draw_fps

import cv2


if __name__ == "__main__":
    source = FrameSource("track/data/test_640.avi")

    tracker = YOLOTracker(
        model_path="models/yolo26n.mnn",
        conf=0.3,
        classes=[0],  # person class only
    )

    writer = VideoFileWriter(
        path="track/data/det.mp4",
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

            cv2.imshow("det", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\nexiting...")

    finally:
        source.release()
        writer.close()
        cv2.destroyAllWindows()
