from streaming.vid_pub import VideoFileWriter
from streaming.subscribe import FrameSource

from track.utils.fps import FpsMeter, draw_fps

from track.detection.yolo_det import UltralyticsYoloDetector
from track.trackers.osnet_sort import ReIDTracker

import cv2


if __name__ == "__main__":

    source = FrameSource("track/data/test_640.mp4")

    detector = UltralyticsYoloDetector(
        "models/yolo26n_ncnn_model",
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
        reid_interval=3,
        bank_size=30,
        ema_alpha=0.9,
        reid_weights="weights/osnet_x1_0_market.pth",
    )

    writer = VideoFileWriter(
        path="track/data/det.mp4",
        size_wh=(640, 480),
        fps=25,
    )
    writer.open()

    fps_meter = FpsMeter()

    try:

        for frame in source.frames():

            fps = fps_meter.tick()

            dets = detector.inference(frame)

            tracks, vis = tracker.update(frame, dets)

            if len(tracks) > 0:
                print(tracks[:, :5])

            if fps is not None:
                vis = draw_fps(vis, fps)

            writer.write(vis)

            cv2.imshow("track", vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:

        print("\nexiting...")

    finally:

        source.release()

        writer.close()

        cv2.destroyAllWindows()